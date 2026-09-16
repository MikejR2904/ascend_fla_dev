"""Generated inputs and independent CPU formula for fused GDN-2 decode."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


HEADS = 16
HEAD_DIM = 128
VALUE_DIM = 128
QK_EPS = 1e-6
NORM_EPS = 1e-5
Q_SCALE = HEAD_DIM**-0.5


def make_inputs(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    parameters = case["parameters"]
    state_scale = float(parameters.get("state_scale", 0.1))
    raw_scale = float(parameters.get("raw_scale", 3.0))
    decay_min = float(parameters.get("decay_min", 1.0))
    decay_max = float(parameters.get("decay_max", 16.0))
    generator = torch.Generator().manual_seed(int(case["seed"]))
    shape = (1, 1, HEADS, HEAD_DIM)

    def raw(scale: float = raw_scale) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    q = raw(0.5)
    k = raw(0.5)
    v = raw(0.25)
    f_raw = raw()
    b_raw = raw()
    w_raw = raw()
    output_gate = raw()
    decay_head = -torch.linspace(decay_min, decay_max, HEADS, dtype=torch.float32)
    decay_rate = decay_head[:, None].expand(HEADS, HEAD_DIM).contiguous()
    dt_bias = (
        torch.rand((HEADS, HEAD_DIM), generator=generator, dtype=torch.float32) * 4.0 - 5.0
    )
    norm_weight = (
        torch.randn(VALUE_DIM, generator=generator, dtype=torch.float32) * 0.1 + 1.0
    ).to(torch.bfloat16).float()
    if state_scale == 0.0:
        initial_state = torch.zeros(1, HEADS, HEAD_DIM, VALUE_DIM)
    else:
        initial_state = (
            torch.randn(
                (1, HEADS, HEAD_DIM, VALUE_DIM),
                generator=generator,
                dtype=torch.float32,
            )
            * state_scale
        )
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "f_raw": f_raw.contiguous(),
        "b_raw": b_raw.contiguous(),
        "w_raw": w_raw.contiguous(),
        "output_gate": output_gate.contiguous(),
        "decay_rate": decay_rate,
        "dt_bias": dt_bias.contiguous(),
        "norm_weight": norm_weight.contiguous(),
        "initial_state": initial_state.contiguous(),
    }


def validate_inputs(
    inputs: dict[str, torch.Tensor], case: dict[str, Any] | None = None
) -> None:
    del case
    shape = (1, 1, HEADS, HEAD_DIM)
    for name in ("q", "k", "v", "f_raw", "b_raw", "w_raw", "output_gate"):
        tensor = inputs[name]
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must be contiguous bfloat16 {shape}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    fp32_shapes = {
        "decay_rate": (HEADS, HEAD_DIM),
        "dt_bias": (HEADS, HEAD_DIM),
        "norm_weight": (VALUE_DIM,),
        "initial_state": (1, HEADS, HEAD_DIM, VALUE_DIM),
    }
    for name, shape_fp32 in fp32_shapes.items():
        tensor = inputs[name]
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape_fp32:
            raise ValueError(f"{name} must be contiguous float32 {shape_fp32}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if not all(bool(torch.isfinite(tensor.float()).all()) for tensor in inputs.values()):
        raise ValueError("all inputs must be finite")
    if bool((inputs["decay_rate"] >= 0).any()):
        raise ValueError("decay_rate must be strictly negative")
    if not torch.equal(inputs["decay_rate"], inputs["decay_rate"][:, :1].expand_as(inputs["decay_rate"])):
        raise ValueError("each decay_rate head row must be one broadcast scalar")


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    validate_inputs(inputs)
    q = inputs["q"].float()
    k = inputs["k"].float()
    v = inputs["v"].float()
    f_raw = inputs["f_raw"].float()
    b = torch.sigmoid(inputs["b_raw"].float())
    w = torch.sigmoid(inputs["w_raw"].float())
    gate = F.silu(inputs["output_gate"].float())

    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + QK_EPS) * Q_SCALE
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + QK_EPS)
    g = inputs["decay_rate"].view(1, 1, HEADS, HEAD_DIM) * F.softplus(
        f_raw + inputs["dt_bias"].view(1, 1, HEADS, HEAD_DIM)
    )
    state = inputs["initial_state"].clone()
    state = state * torch.exp(g[:, 0]).unsqueeze(-1)
    erase = torch.matmul((b[:, 0] * k[:, 0]).unsqueeze(-2), state).squeeze(-2)
    delta = w[:, 0] * v[:, 0] - erase
    state = state + k[:, 0].unsqueeze(-1) * delta.unsqueeze(-2)
    recurrent = torch.matmul(q[:, 0].unsqueeze(-2), state).squeeze(-2).unsqueeze(1)
    normalized = recurrent * torch.rsqrt(
        recurrent.square().mean(dim=-1, keepdim=True) + NORM_EPS
    )
    normalized = normalized * inputs["norm_weight"].view(1, 1, 1, VALUE_DIM)
    output = (normalized * gate).to(torch.bfloat16)
    return {"o": output.contiguous(), "final_state": state.contiguous()}


def validate_reference(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    case: dict[str, Any] | None = None,
) -> None:
    del inputs, case
    expected = {
        "o": (torch.bfloat16, (1, 1, HEADS, VALUE_DIM)),
        "final_state": (torch.float32, (1, HEADS, HEAD_DIM, VALUE_DIM)),
    }
    if set(outputs) != set(expected):
        raise ValueError(f"reference outputs must be {sorted(expected)}")
    for name, (dtype, shape) in expected.items():
        tensor = outputs[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} {shape}")
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"{name} contains non-finite values")
