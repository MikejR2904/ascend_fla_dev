"""CCE-backed GDN-2 recurrent forward for short inference sequences.

The public tensors stay token-major. The repository-owned Ascriptor kernel performs
q/k L2 normalization and the full activated-gate recurrence in FP32, while this
wrapper validates the narrow first ABI, converts floating inputs to FP32, allocates
outputs and invokes the generated aclnn operator in-process. There is no Torch
fallback for this backend.
"""
from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

import torch


HEAD_DIM = 128
VALUE_DIM = 128
GDN2_RECURRENT_T_MAX = 16
GDN2_RECURRENT_BLOCK_DIM = 8
SUPPORTED_HEADS = (1, 16)
SUPPORTED_INPUT_DTYPES = (torch.bfloat16, torch.float32)
QK_NORM_EPS = 1e-6
ATTENTION_SCALE = HEAD_DIM**-0.5


def _unit_root() -> Path:
    root = Path(__file__).resolve().parents[3] / "kernels/projects/a5/gdn2_fused_recurrent"
    if not (root / "kernels/step.py").is_file():
        raise FileNotFoundError(f"missing repository-owned GDN-2 recurrent unit: {root}")
    return root


@functools.lru_cache(maxsize=1)
def gdn2_fused_recurrent_kernel() -> Any:
    path = _unit_root() / "kernels/step.py"
    spec = importlib.util.spec_from_file_location("_afla_gdn2_recurrent_step", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.gdn2_fused_recurrent_kernel


@functools.lru_cache(maxsize=1)
def _compiled(device: str, block_dim: int) -> Any:
    from ...runtime.compile import compile_kernel

    return compile_kernel(
        gdn2_fused_recurrent_kernel(),
        device=device,
        block_dim=block_dim,
        backend="cce",
    )


def prepare(*, device: str = "a5", block_dim: int = GDN2_RECURRENT_BLOCK_DIM) -> None:
    """Build and register the CCE vendor tree before the process executes any op."""
    if device != "a5":
        raise ValueError(f"GDN-2 recurrent first scope requires device='a5', got {device!r}")
    if block_dim != GDN2_RECURRENT_BLOCK_DIM:
        raise ValueError(
            f"GDN-2 recurrent first scope fixes block_dim={GDN2_RECURRENT_BLOCK_DIM}, "
            f"got {block_dim}"
        )
    _compiled(device, block_dim)


def _check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    scale: float,
    use_qk_l2norm: bool,
    qk_norm_eps: float,
    device: str,
    block_dim: int,
) -> tuple[int, int, int]:
    if device != "a5":
        raise ValueError(f"GDN-2 recurrent first scope requires device='a5', got {device!r}")
    if block_dim != GDN2_RECURRENT_BLOCK_DIM:
        raise ValueError(
            f"block_dim must be {GDN2_RECURRENT_BLOCK_DIM}, got {block_dim}; "
            "other launch partitions have no contract evidence"
        )
    if not use_qk_l2norm:
        raise ValueError("CCE recurrent first scope requires q/k L2 normalization")
    if not math.isclose(qk_norm_eps, QK_NORM_EPS, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"qk_norm_eps must be {QK_NORM_EPS}, got {qk_norm_eps}")
    if not math.isclose(scale, ATTENTION_SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"attention scale must be {ATTENTION_SCALE}, got {scale}")
    if q.ndim != 4:
        raise ValueError(f"q must be [B,T,H,128], got {tuple(q.shape)}")
    batch, time, heads, width = q.shape
    expected = (batch, time, heads, HEAD_DIM)
    if batch != 1:
        raise ValueError(f"first CCE recurrent scope fixes B=1, got B={batch}")
    if not 1 <= time <= GDN2_RECURRENT_T_MAX:
        raise ValueError(
            f"CCE recurrent supports 1<=T<={GDN2_RECURRENT_T_MAX}, got T={time}; "
            "longer sequences require a separately validated chunk path"
        )
    if heads not in SUPPORTED_HEADS or width != HEAD_DIM:
        raise ValueError(
            f"CCE recurrent requires H in {SUPPORTED_HEADS} and K=128, got H={heads} K={width}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("b", b), ("w", w)):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}")
        if tensor.dtype not in SUPPORTED_INPUT_DTYPES:
            raise ValueError(
                f"{name} must be bfloat16 or float32, got {tensor.dtype}"
            )
    if g.shape != expected or g.dtype != torch.float32:
        raise ValueError(f"g must be float32 {expected}, got {g.dtype} {tuple(g.shape)}")
    dtypes = {tensor.dtype for tensor in (q, k, v, b, w)}
    if len(dtypes) != 1:
        raise ValueError(f"q/k/v/b/w must share one dtype, got {sorted(map(str, dtypes))}")
    state_shape = (batch, heads, HEAD_DIM, VALUE_DIM)
    if initial_state is not None and (
        tuple(initial_state.shape) != state_shape or initial_state.dtype != torch.float32
    ):
        raise ValueError(
            f"initial_state must be float32 {state_shape}, got "
            f"{initial_state.dtype} {tuple(initial_state.shape)}"
        )
    tensors = [q, k, v, g, b, w]
    if initial_state is not None:
        tensors.append(initial_state)
    for name, tensor in zip(
        ("q", "k", "v", "g", "b", "w", "initial_state"), tensors
    ):
        if tensor.device.type != "npu":
            raise ValueError(f"{name} must be on NPU, got {tensor.device}")
        if tensor.device != q.device:
            raise ValueError(f"all inputs must be on {q.device}, got {name} on {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")
    return batch, time, heads


def fused_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    scale: float = ATTENTION_SCALE,
    use_qk_l2norm: bool = True,
    qk_norm_eps: float = QK_NORM_EPS,
    device: str = "a5",
    block_dim: int = GDN2_RECURRENT_BLOCK_DIM,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the CCE GDN-2 recurrence and return token-major output plus FP32 state."""
    batch, time, heads = _check(
        q,
        k,
        v,
        g,
        b,
        w,
        initial_state,
        scale=scale,
        use_qk_l2norm=use_qk_l2norm,
        qk_norm_eps=qk_norm_eps,
        device=device,
        block_dim=block_dim,
    )
    output_dtype = v.dtype
    inputs_fp32 = [tensor if tensor.dtype == torch.float32 else tensor.float() for tensor in (q, k, v, b, w)]
    q_fp32, k_fp32, v_fp32, b_fp32, w_fp32 = inputs_fp32
    state = initial_state
    if state is None:
        state = torch.zeros(
            batch,
            heads,
            HEAD_DIM,
            VALUE_DIM,
            dtype=torch.float32,
            device="cpu",
        ).to(q.device)
    output_fp32 = torch.empty(
        batch, time, heads, VALUE_DIM, dtype=torch.float32, device=q.device
    )
    final_state = torch.empty(
        batch, heads, HEAD_DIM, VALUE_DIM, dtype=torch.float32, device=q.device
    )
    _compiled(device, block_dim)(
        {
            "q": q_fp32,
            "k": k_fp32,
            "v": v_fp32,
            "g": g,
            "erase_gate": b_fp32,
            "w": w_fp32,
            "initial_state": state,
        },
        {"B": batch, "T": time, "H": heads},
        {"o": output_fp32, "final_state": final_state},
    )
    output = output_fp32 if output_dtype == torch.float32 else output_fp32.to(output_dtype)
    return output, final_state if output_final_state else None
