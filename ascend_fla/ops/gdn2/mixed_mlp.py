"""Fixed-shape mixed RMSNorm2/W1/W2/SwiGLU inference boundary."""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch


GDN2_MIXED_MLP_BLOCK_DIM = 28
GDN2_MIXED_MLP_HIDDEN_SIZE = 2304
GDN2_MIXED_MLP_INTERMEDIATE_SIZE = 6208
GDN2_MIXED_MLP_ITEMS = 97
GDN2_MIXED_MLP_PAIR_N = 128


def _unit_root() -> Path:
    root = (
        Path(__file__).resolve().parents[3]
        / "kernels/projects/a5/gdn2_norm2_w12_swiglu"
    )
    if not (root / "kernels/step.py").is_file():
        raise FileNotFoundError(f"missing repository-owned GDN-2 mixed MLP unit: {root}")
    return root


@functools.lru_cache(maxsize=1)
def gdn2_mixed_mlp_kernel() -> Any:
    path = _unit_root() / "kernels/step.py"
    spec = importlib.util.spec_from_file_location("_afla_gdn2_mixed_mlp_step", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.gdn2_norm2_w12_swiglu_kernel


@functools.lru_cache(maxsize=1)
def _compiled(device: str, block_dim: int) -> Any:
    from ...runtime.compile import compile_kernel

    return compile_kernel(
        gdn2_mixed_mlp_kernel(),
        device=device,
        block_dim=block_dim,
        backend="cce",
    )


def prepare(
    *, device: str = "a5", block_dim: int = GDN2_MIXED_MLP_BLOCK_DIM
) -> None:
    """Build/register the mixed MLP vendor tree before any aclnn execution."""
    if device != "a5":
        raise ValueError(f"GDN-2 mixed MLP requires device='a5', got {device!r}")
    if block_dim != GDN2_MIXED_MLP_BLOCK_DIM:
        raise ValueError(
            "GDN-2 mixed MLP fixes "
            f"block_dim={GDN2_MIXED_MLP_BLOCK_DIM}, got {block_dim}"
        )
    _compiled(device, block_dim)


def _check(
    x: torch.Tensor,
    gamma: torch.Tensor,
    paired_weight: torch.Tensor,
    *,
    device: str,
    block_dim: int,
) -> None:
    if device != "a5" or block_dim != GDN2_MIXED_MLP_BLOCK_DIM:
        raise ValueError(
            "GDN-2 mixed MLP requires device='a5' and "
            f"block_dim={GDN2_MIXED_MLP_BLOCK_DIM}, got {device!r}/{block_dim}"
        )
    expected = {
        "x": (x, (1, GDN2_MIXED_MLP_HIDDEN_SIZE)),
        "gamma": (gamma, (GDN2_MIXED_MLP_HIDDEN_SIZE,)),
        "paired_weight": (
            paired_weight,
            (
                GDN2_MIXED_MLP_ITEMS,
                GDN2_MIXED_MLP_PAIR_N,
                GDN2_MIXED_MLP_HIDDEN_SIZE,
            ),
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


def fused_norm2_w12_swiglu_gdn2(
    x: torch.Tensor,
    gamma: torch.Tensor,
    paired_weight: torch.Tensor,
    *,
    device: str = "a5",
    block_dim: int = GDN2_MIXED_MLP_BLOCK_DIM,
) -> torch.Tensor:
    """Return the fixed B=T=1 BF16 SwiGLU hidden activation before W3."""
    if torch.is_grad_enabled():
        raise RuntimeError("GDN-2 mixed MLP is inference-only")
    _check(x, gamma, paired_weight, device=device, block_dim=block_dim)
    hidden = torch.empty(
        (1, GDN2_MIXED_MLP_INTERMEDIATE_SIZE),
        dtype=torch.bfloat16,
        device=x.device,
    )
    _compiled(device, block_dim)(
        {"x": x, "gamma": gamma, "paired_weight": paired_weight},
        {},
        {"hidden": hidden},
    )
    return hidden


__all__ = [
    "GDN2_MIXED_MLP_BLOCK_DIM",
    "GDN2_MIXED_MLP_HIDDEN_SIZE",
    "GDN2_MIXED_MLP_INTERMEDIATE_SIZE",
    "GDN2_MIXED_MLP_ITEMS",
    "GDN2_MIXED_MLP_PAIR_N",
    "fused_norm2_w12_swiglu_gdn2",
    "gdn2_mixed_mlp_kernel",
    "prepare",
]
