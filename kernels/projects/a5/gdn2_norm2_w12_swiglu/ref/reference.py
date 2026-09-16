"""Input generation and independent FP32 formula for the mixed MLP unit."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


K = 2304
INTERMEDIATE = 6208
CHANNELS_PER_ITEM = 64
ITEMS = INTERMEDIATE // CHANNELS_PER_ITEM
PAIR_N = 2 * CHANNELS_PER_ITEM
EPS = 1e-5


def make_inputs(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    parameters = case["parameters"]
    generator = torch.Generator().manual_seed(int(case["seed"]))
    x_scale = float(parameters.get("x_scale", 1.0))
    weight_scale = float(parameters.get("weight_scale", 0.02))
    x = (torch.randn((1, K), generator=generator) * x_scale).to(torch.bfloat16)
    gamma = (1.0 + 0.1 * torch.randn((K,), generator=generator)).to(torch.bfloat16)
    weight = (
        torch.randn((ITEMS, PAIR_N, K), generator=generator) * weight_scale
    ).to(torch.bfloat16)
    if bool(parameters.get("structured", False)):
        x = (
            torch.arange(K, dtype=torch.float32).remainder(31).sub_(15).div_(16)
        ).view(1, K).to(torch.bfloat16)
        gamma = (
            1.0 + torch.arange(K, dtype=torch.float32).remainder(11) / 64.0
        ).to(torch.bfloat16)
        row = torch.arange(ITEMS * PAIR_N, dtype=torch.float32).remainder(17)
        col = torch.arange(K, dtype=torch.float32).remainder(13)
        weight = ((row[:, None] - 8) * (col[None, :] - 6) / 4096).reshape(
            ITEMS, PAIR_N, K
        ).to(torch.bfloat16)
    return {
        "x": x.contiguous(),
        "gamma": gamma.contiguous(),
        "paired_weight": weight.contiguous(),
    }


def validate_inputs(
    inputs: dict[str, torch.Tensor], case: dict[str, Any] | None = None
) -> None:
    del case
    expected = {
        "x": (torch.bfloat16, (1, K)),
        "gamma": (torch.bfloat16, (K,)),
        "paired_weight": (torch.bfloat16, (ITEMS, PAIR_N, K)),
    }
    if set(inputs) != set(expected):
        raise ValueError(f"inputs must be {sorted(expected)}")
    for name, (dtype, shape) in expected.items():
        tensor = inputs[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} {shape}")
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"{name} contains non-finite values")


def reference(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    validate_inputs(inputs)
    x = inputs["x"]
    gamma = inputs["gamma"]
    weight = inputs["paired_weight"]
    inverse = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + EPS)
    normalized = (x.float() * inverse * gamma.float()).to(torch.bfloat16)
    product = F.linear(
        normalized.float(), weight.flatten(0, 1).float()
    ).reshape(ITEMS, PAIR_N).to(torch.bfloat16)
    left = product[:, :CHANNELS_PER_ITEM]
    right = product[:, CHANNELS_PER_ITEM:]
    silu = F.silu(left).to(torch.bfloat16)
    hidden = (silu * right).reshape(1, INTERMEDIATE).to(torch.bfloat16)
    return {"hidden": hidden.contiguous()}


def validate_reference(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    case: dict[str, Any] | None = None,
) -> None:
    del inputs, case
    if set(outputs) != {"hidden"}:
        raise ValueError("reference output must be ['hidden']")
    hidden = outputs["hidden"]
    if (
        hidden.dtype != torch.bfloat16
        or tuple(hidden.shape) != (1, INTERMEDIATE)
        or not hidden.is_contiguous()
        or not bool(torch.isfinite(hidden.float()).all())
    ):
        raise ValueError("hidden must be finite contiguous bfloat16 [1,6208]")
