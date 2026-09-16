"""Generated inputs and independent CPU formula for GDN-2 decode short-conv."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


CHANNELS = 6144
WIDTH = 4


def make_inputs(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    parameters = case["parameters"]
    generator = torch.Generator().manual_seed(int(case["seed"]))
    x_scale = float(parameters.get("x_scale", 0.5))
    cache_scale = float(parameters.get("cache_scale", 0.5))
    weight_scale = float(parameters.get("weight_scale", 0.2))

    x = (torch.randn((1, CHANNELS), generator=generator) * x_scale).to(torch.bfloat16)
    cache = (
        torch.randn((CHANNELS, WIDTH), generator=generator) * cache_scale
    ).to(torch.bfloat16)
    weight = (
        torch.randn((CHANNELS, WIDTH), generator=generator) * weight_scale
    ).to(torch.bfloat16)
    if bool(parameters.get("zero_history", False)):
        cache.zero_()
    if bool(parameters.get("structured", False)):
        # Lane- and tap-distinct bit patterns expose cache permutation errors.
        lane = torch.arange(CHANNELS, dtype=torch.float32).remainder(97) / 32.0
        cache = torch.stack(
            (lane + 1.0, lane + 2.0, lane + 3.0, lane + 4.0), dim=-1
        ).to(torch.bfloat16)
        x = (lane.view(1, CHANNELS) - 1.25).to(torch.bfloat16)
        weight = torch.tensor(
            [0.125, -0.25, 0.5, 0.75], dtype=torch.bfloat16
        ).view(1, WIDTH).expand(CHANNELS, WIDTH).contiguous()
    return {"x": x.contiguous(), "cache": cache.contiguous(), "weight": weight.contiguous()}


def validate_inputs(
    inputs: dict[str, torch.Tensor], case: dict[str, Any] | None = None
) -> None:
    del case
    expected = {
        "x": (torch.bfloat16, (1, CHANNELS)),
        "cache": (torch.bfloat16, (CHANNELS, WIDTH)),
        "weight": (torch.bfloat16, (CHANNELS, WIDTH)),
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
    current = inputs["x"].transpose(0, 1)
    new_cache = torch.cat((inputs["cache"][:, 1:], current), dim=-1).contiguous()
    convolution = (
        new_cache.float() * inputs["weight"].float()
    ).sum(dim=-1, dtype=torch.float32)
    y = F.silu(convolution).to(torch.bfloat16).view(1, CHANNELS).contiguous()
    return {"y": y, "new_cache": new_cache}


def validate_reference(
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    case: dict[str, Any] | None = None,
) -> None:
    del inputs, case
    expected = {
        "y": (torch.bfloat16, (1, CHANNELS)),
        "new_cache": (torch.bfloat16, (CHANNELS, WIDTH)),
    }
    if set(outputs) != set(expected):
        raise ValueError(f"reference outputs must be {sorted(expected)}")
    for name, (dtype, shape) in expected.items():
        tensor = outputs[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} {shape}")
        if not bool(torch.isfinite(tensor.float()).all()):
            raise ValueError(f"{name} contains non-finite values")
