"""Model-specific one-launch BF16 GDN-2 decode short convolution.

The public tensors retain the packed module layout.  The fixed CCE kernel sees
zero-copy flattened views, computes all 6144 width-four depthwise channels,
applies SiLU, and publishes the next cache in the same launch.
"""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch


GDN2_SHORT_CONV_BLOCK_DIM = 8
GDN2_SHORT_CONV_CHANNELS = 6144
GDN2_SHORT_CONV_WIDTH = 4


def _unit_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "kernels/projects/a5/gdn2_short_conv_decode"
    if not (root / "kernels/step.py").is_file():
        raise FileNotFoundError(f"missing repository-owned GDN-2 short-conv unit: {root}")
    return root


@functools.lru_cache(maxsize=1)
def gdn2_short_conv_decode_kernel() -> Any:
    path = _unit_root() / "kernels/step.py"
    spec = importlib.util.spec_from_file_location("_afla_gdn2_short_conv_decode_step", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.gdn2_short_conv_decode_kernel


@functools.lru_cache(maxsize=1)
def _compiled(device: str, block_dim: int) -> Any:
    from ...runtime.compile import compile_kernel

    return compile_kernel(
        gdn2_short_conv_decode_kernel(),
        device=device,
        block_dim=block_dim,
        backend="cce",
    )


def prepare(
    *, device: str = "a5", block_dim: int = GDN2_SHORT_CONV_BLOCK_DIM
) -> None:
    """Build/register the short-conv vendor tree before any aclnn execution."""
    if device != "a5":
        raise ValueError(f"GDN-2 short-conv decode requires device='a5', got {device!r}")
    if block_dim != GDN2_SHORT_CONV_BLOCK_DIM:
        raise ValueError(
            "GDN-2 short-conv decode fixes "
            f"block_dim={GDN2_SHORT_CONV_BLOCK_DIM}, got {block_dim}"
        )
    _compiled(device, block_dim)


def _check(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    *,
    device: str,
    block_dim: int,
) -> None:
    if device != "a5" or block_dim != GDN2_SHORT_CONV_BLOCK_DIM:
        raise ValueError(
            "GDN-2 short-conv decode requires device='a5' and "
            f"block_dim={GDN2_SHORT_CONV_BLOCK_DIM}, got {device!r}/{block_dim}"
        )
    expected = {
        "x": (x, (1, 1, GDN2_SHORT_CONV_CHANNELS)),
        "cache": (
            cache,
            (1, GDN2_SHORT_CONV_CHANNELS, GDN2_SHORT_CONV_WIDTH),
        ),
        "weight": (
            weight,
            (GDN2_SHORT_CONV_CHANNELS, 1, GDN2_SHORT_CONV_WIDTH),
        ),
    }
    for name, (tensor, shape) in expected.items():
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must be contiguous bfloat16 {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
        if tensor.device.type != "npu":
            raise ValueError(f"{name} must be on NPU, got {tensor.device}")
        if tensor.device != x.device:
            raise ValueError(f"all inputs must be on {x.device}, got {name} on {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")


def fused_short_conv_decode_gdn2(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    *,
    device: str = "a5",
    block_dim: int = GDN2_SHORT_CONV_BLOCK_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SiLU(depthwise_conv4(x, cache)) and the updated packed cache."""
    if torch.is_grad_enabled():
        raise RuntimeError("GDN-2 short-conv decode is inference-only")
    _check(x, cache, weight, device=device, block_dim=block_dim)

    y = torch.empty_like(x)
    new_cache = torch.empty_like(cache)
    _compiled(device, block_dim)(
        {
            "x": x.view(1, GDN2_SHORT_CONV_CHANNELS),
            "cache": cache.view(GDN2_SHORT_CONV_CHANNELS, GDN2_SHORT_CONV_WIDTH),
            "weight": weight.view(GDN2_SHORT_CONV_CHANNELS, GDN2_SHORT_CONV_WIDTH),
        },
        {},
        {
            "y": y.view(1, GDN2_SHORT_CONV_CHANNELS),
            "new_cache": new_cache.view(
                GDN2_SHORT_CONV_CHANNELS, GDN2_SHORT_CONV_WIDTH
            ),
        },
    )
    return y, new_cache


__all__ = [
    "GDN2_SHORT_CONV_BLOCK_DIM",
    "GDN2_SHORT_CONV_CHANNELS",
    "GDN2_SHORT_CONV_WIDTH",
    "fused_short_conv_decode_gdn2",
    "gdn2_short_conv_decode_kernel",
    "prepare",
]
