#!/usr/bin/env python3
"""Validate and time the fixed GDN-2 packed short-conv decode kernel.

Run each order in a fresh process.  The CCE path includes the public wrapper and
output allocation; the torch_npu baseline is the optimized T=1 composition that
shares its convolution window with the returned cache.  Both are compared with
an independent CPU FP32 formula before timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.ops.gdn2.short_conv_decode import (  # noqa: E402
    GDN2_SHORT_CONV_BLOCK_DIM,
    GDN2_SHORT_CONV_CHANNELS,
    GDN2_SHORT_CONV_WIDTH,
    fused_short_conv_decode_gdn2,
    prepare,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--order", choices=("cce-first", "torch-first"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.device.startswith("npu"):
        parser.error("this benchmark requires an NPU device")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations positive")
    return args


def _cpu_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn((1, 1, GDN2_SHORT_CONV_CHANNELS), generator=generator) * 0.75).to(
        torch.bfloat16
    )
    cache = (
        torch.randn(
            (1, GDN2_SHORT_CONV_CHANNELS, GDN2_SHORT_CONV_WIDTH),
            generator=generator,
        )
        * 0.75
    ).to(torch.bfloat16)
    weight = (
        torch.randn(
            (GDN2_SHORT_CONV_CHANNELS, 1, GDN2_SHORT_CONV_WIDTH),
            generator=generator,
        )
        * 0.3
    ).to(torch.bfloat16)
    return x.contiguous(), cache.contiguous(), weight.contiguous()


def _cpu_reference(
    x: torch.Tensor, cache: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    current = x.transpose(1, 2)
    new_cache = torch.cat((cache[..., 1:], current), dim=-1).contiguous()
    convolution = (
        new_cache[0].float() * weight[:, 0].float()
    ).sum(dim=-1, dtype=torch.float32)
    y = F.silu(convolution).to(torch.bfloat16).view_as(x).contiguous()
    return y, new_cache


def _torch_npu(
    x: torch.Tensor, cache: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    current = x.transpose(1, 2)
    window = torch.cat((cache[..., 1:], current), dim=-1)
    y = F.conv1d(
        window,
        weight,
        bias=None,
        groups=GDN2_SHORT_CONV_CHANNELS,
        padding=0,
    )
    return F.silu(y.transpose(1, 2)), window


def _relative_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    delta = got.float() - want.float()
    return float(delta.norm() / want.float().norm().clamp_min(1e-30))


def _comparison(
    got: tuple[torch.Tensor, torch.Tensor],
    want: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float | bool]:
    got_y, got_cache = (tensor.cpu() for tensor in got)
    want_y, want_cache = want
    delta = got_y.float() - want_y.float()
    return {
        "y_max_abs": float(delta.abs().max()),
        "y_relative_l2": _relative_l2(got_y, want_y),
        "cache_bitwise_equal": bool(torch.equal(got_cache, want_cache)),
        "cache_relative_l2": _relative_l2(got_cache, want_cache),
    }


def _time(
    function: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    last = None
    for _ in range(warmup):
        last = function()
    torch.npu.synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        last = function()
    torch.npu.synchronize(device)
    elapsed = time.perf_counter() - started
    del last
    return elapsed * 1e6 / iterations


def main() -> int:
    args = _parse_args()
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:  # pragma: no cover - remote-only benchmark
        raise RuntimeError("torch_npu is required") from error
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch_npu is required")

    # Register the custom OPP before the first NPU operator is resolved.
    prepare(device="a5", block_dim=GDN2_SHORT_CONV_BLOCK_DIM)
    device = torch.device(args.device)
    cpu_x, cpu_cache, cpu_weight = _cpu_inputs(args.seed)
    oracle = _cpu_reference(cpu_x, cpu_cache, cpu_weight)
    x, cache, weight = (
        tensor.to(device) for tensor in (cpu_x, cpu_cache, cpu_weight)
    )

    def cce() -> tuple[torch.Tensor, torch.Tensor]:
        return fused_short_conv_decode_gdn2(x, cache, weight)

    def torch_npu() -> tuple[torch.Tensor, torch.Tensor]:
        return _torch_npu(x, cache, weight)

    with torch.inference_mode():
        cce_result = cce()
        torch_result = torch_npu()
        torch.npu.synchronize(device)
        comparisons = {
            "cce_vs_cpu": _comparison(cce_result, oracle),
            "torch_npu_vs_cpu": _comparison(torch_result, oracle),
            "cce_vs_torch_npu": _comparison(
                cce_result,
                tuple(tensor.cpu() for tensor in torch_result),
            ),
        }
        timings: dict[str, float] = {}
        order = (
            (("cce", cce), ("torch_npu", torch_npu))
            if args.order == "cce-first"
            else (("torch_npu", torch_npu), ("cce", cce))
        )
        for name, function in order:
            timings[f"{name}_us"] = _time(
                function, device, args.warmup, args.iterations
            )

    cce_us = timings["cce_us"]
    torch_us = timings["torch_npu_us"]
    passed = (
        comparisons["cce_vs_cpu"]["y_relative_l2"] <= 0.001
        and comparisons["cce_vs_cpu"]["y_max_abs"] <= 0.015625
        and comparisons["cce_vs_cpu"]["cache_bitwise_equal"]
    )
    result = {
        "passed": passed,
        "shape": [1, 1, GDN2_SHORT_CONV_CHANNELS],
        "width": GDN2_SHORT_CONV_WIDTH,
        "dtype": "bfloat16",
        "block_dim": GDN2_SHORT_CONV_BLOCK_DIM,
        "seed": args.seed,
        "order": args.order,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "comparison": comparisons,
        "timing": {
            **timings,
            "speedup_torch_over_cce": torch_us / cce_us,
            "cce_calls_per_second": 1e6 / cce_us,
            "torch_npu_calls_per_second": 1e6 / torch_us,
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
