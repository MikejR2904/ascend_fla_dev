"""Model-specific one-launch BF16 GDN-2 decode boundary.

This entry is deliberately narrower than :mod:`fused_recurrent`: it accepts the
released model's raw BF16 projection outputs and owns gate activation, q/k norm,
FP32 recurrence, output RMSNorm+swish and the final BF16 store.  The standalone
FP32 recurrent ABI remains available as the semantic/control path.
"""
from __future__ import annotations

import functools
import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch


GDN2_DECODE_BLOCK_DIM = 8
GDN2_DECODE_HEADS = 16
HEAD_DIM = 128
VALUE_DIM = 128


def _unit_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "kernels/projects/a5/gdn2_fused_decode"
    if not (root / "kernels/step.py").is_file():
        raise FileNotFoundError(f"missing repository-owned GDN-2 fused decode unit: {root}")
    return root


@functools.lru_cache(maxsize=1)
def gdn2_fused_decode_kernel() -> Any:
    path = _unit_root() / "kernels/step.py"
    spec = importlib.util.spec_from_file_location("_afla_gdn2_fused_decode_step", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.gdn2_fused_decode_kernel


@functools.lru_cache(maxsize=1)
def _compiled(device: str, block_dim: int) -> Any:
    from ...runtime.compile import compile_kernel

    return compile_kernel(
        gdn2_fused_decode_kernel(),
        device=device,
        block_dim=block_dim,
        backend="cce",
    )


def prepare(*, device: str = "a5", block_dim: int = GDN2_DECODE_BLOCK_DIM) -> None:
    """Build/register the fused decode vendor tree before any aclnn execution."""
    if device != "a5":
        raise ValueError(f"GDN-2 fused decode requires device='a5', got {device!r}")
    if block_dim != GDN2_DECODE_BLOCK_DIM:
        raise ValueError(
            f"GDN-2 fused decode fixes block_dim={GDN2_DECODE_BLOCK_DIM}, got {block_dim}"
        )
    _compiled(device, block_dim)


def _check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    f_raw: torch.Tensor,
    b_raw: torch.Tensor,
    w_raw: torch.Tensor,
    output_gate: torch.Tensor,
    decay_rate: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    device: str,
    block_dim: int,
) -> None:
    if device != "a5" or block_dim != GDN2_DECODE_BLOCK_DIM:
        raise ValueError(
            "GDN-2 fused decode requires device='a5' and "
            f"block_dim={GDN2_DECODE_BLOCK_DIM}, got {device!r}/{block_dim}"
        )
    shape = (1, 1, GDN2_DECODE_HEADS, HEAD_DIM)
    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "f_raw": f_raw,
        "b_raw": b_raw,
        "w_raw": w_raw,
        "output_gate": output_gate,
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"{name} must be contiguous bfloat16 {shape}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
    fp32_shapes = {
        "decay_rate": (GDN2_DECODE_HEADS, HEAD_DIM),
        "dt_bias": (GDN2_DECODE_HEADS, HEAD_DIM),
        "norm_weight": (VALUE_DIM,),
    }
    fp32_tensors = {
        "decay_rate": decay_rate,
        "dt_bias": dt_bias,
        "norm_weight": norm_weight,
    }
    for name, shape_fp32 in fp32_shapes.items():
        tensor = fp32_tensors[name]
        if tuple(tensor.shape) != shape_fp32 or tensor.dtype != torch.float32:
            raise ValueError(
                f"{name} must be contiguous float32 {shape_fp32}, got "
                f"{tensor.dtype} {tuple(tensor.shape)}"
            )
    if initial_state is not None and (
        tuple(initial_state.shape) != (1, GDN2_DECODE_HEADS, HEAD_DIM, VALUE_DIM)
        or initial_state.dtype != torch.float32
    ):
        raise ValueError(
            "initial_state must be contiguous float32 "
            f"{(1, GDN2_DECODE_HEADS, HEAD_DIM, VALUE_DIM)}"
        )
    all_tensors = [*tensors.values(), *fp32_tensors.values()]
    if initial_state is not None:
        all_tensors.append(initial_state)
    for name, tensor in zip(
        [*tensors, *fp32_tensors, "initial_state"], all_tensors
    ):
        if tensor.device.type != "npu":
            raise ValueError(f"{name} must be on NPU, got {tensor.device}")
        if tensor.device != q.device:
            raise ValueError(f"all inputs must be on {q.device}, got {name} on {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")


def fused_decode_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    f_raw: torch.Tensor,
    b_raw: torch.Tensor,
    w_raw: torch.Tensor,
    output_gate: torch.Tensor,
    decay_rate: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    device: str = "a5",
    block_dim: int = GDN2_DECODE_BLOCK_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fixed B1T1H16 BF16 fused decode and return BF16 output/FP32 state."""
    if torch.is_grad_enabled():
        raise RuntimeError("GDN-2 fused decode is inference-only")
    _check(
        q,
        k,
        v,
        f_raw,
        b_raw,
        w_raw,
        output_gate,
        decay_rate,
        dt_bias,
        norm_weight,
        initial_state,
        device=device,
        block_dim=block_dim,
    )
    state = initial_state
    if state is None:
        state = torch.zeros(
            1,
            GDN2_DECODE_HEADS,
            HEAD_DIM,
            VALUE_DIM,
            dtype=torch.float32,
            device="cpu",
        ).to(q.device)
    output = torch.empty(
        1, 1, GDN2_DECODE_HEADS, VALUE_DIM, dtype=torch.bfloat16, device=q.device
    )
    final_state = torch.empty(
        1,
        GDN2_DECODE_HEADS,
        HEAD_DIM,
        VALUE_DIM,
        dtype=torch.float32,
        device=q.device,
    )
    _compiled(device, block_dim)(
        {
            "q": q,
            "k": k,
            "v": v,
            "f_raw": f_raw,
            "b_raw": b_raw,
            "w_raw": w_raw,
            "output_gate": output_gate,
            "decay_rate": decay_rate,
            "dt_bias": dt_bias,
            "norm_weight": norm_weight,
            "initial_state": state,
        },
        {},
        {"o": output, "final_state": final_state},
    )
    return output, final_state


__all__ = [
    "GDN2_DECODE_BLOCK_DIM",
    "GDN2_DECODE_HEADS",
    "fused_decode_gdn2",
    "gdn2_fused_decode_kernel",
    "prepare",
]
