#!/usr/bin/env python3
"""Validate and time the Ascriptor CCE GDN-2 recurrent operator on an NPU.

The script builds/registers the custom OPP before any compute operator runs, compares
both FP32 and BF16-input cases with the independent CPU FP32-state recurrence, checks
state chaining, and measures the complete public call against the torch_npu baseline.
Physical card selection remains the caller's responsibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.ops.gdn2 import fused_recurrent_gdn2, prepare
from ascend_fla.reference.gdn2 import gdn2_recurrent_reference


HEAD_DIM = 128
SCALE = HEAD_DIM**-0.5
QK_EPS = 1e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations must be positive")
    return args


def _make_cpu(
    *,
    seed: int,
    time_steps: int,
    heads: int,
    dtype: torch.dtype,
    state_scale: float,
    gate_scale: float,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    shape = (1, time_steps, heads, HEAD_DIM)
    q = (torch.randn(shape, generator=generator) * 0.5).to(dtype)
    k = (torch.randn(shape, generator=generator) * 0.5).to(dtype)
    v = (torch.randn(shape, generator=generator) * 0.25).to(dtype)
    g = -torch.rand(shape, generator=generator) * gate_scale
    b = torch.rand(shape, generator=generator).to(dtype)
    w = torch.rand(shape, generator=generator).to(dtype)
    state = torch.randn(1, heads, HEAD_DIM, HEAD_DIM, generator=generator) * state_scale
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "g": g.contiguous(),
        "b": b.contiguous(),
        "w": w.contiguous(),
        "initial_state": state.contiguous(),
    }


def _reference(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return gdn2_recurrent_reference(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        inputs["initial_state"],
        scale=SCALE,
        use_qk_l2norm=True,
        qk_norm_eps=QK_EPS,
    )


def _device_inputs(
    inputs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _run_cce(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    output, state = fused_recurrent_gdn2(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["b"],
        inputs["w"],
        initial_state=inputs["initial_state"],
        output_final_state=True,
    )
    assert state is not None
    return output, state


def _diff(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    got_cpu = got.detach().cpu().float()
    want_cpu = want.detach().cpu().float()
    delta = got_cpu - want_cpu
    return {
        "max_abs_diff": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / want_cpu.norm().clamp_min(1e-30)),
    }


def _timed(
    function: Callable[[], Any],
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
    del last
    return (time.perf_counter() - started) * 1e6 / iterations


def main() -> None:
    args = _parse_args()
    if args.device.split(":", 1)[0] != "npu":
        raise ValueError(f"this verification requires an NPU, got {args.device}")
    import torch_npu  # noqa: F401

    build_started = time.perf_counter()
    prepare(block_dim=args.block_dim)
    build_seconds = time.perf_counter() - build_started
    device = torch.device(args.device)

    cases = (
        ("fp32_t1_h1_zero", 20260914, 1, 1, torch.float32, 0.0, 3.0),
        ("fp32_t4_h1_state", 20260915, 4, 1, torch.float32, 0.1, 6.0),
        ("fp32_t1_h16_real", 20260917, 1, 16, torch.float32, 0.1, 10.0),
        ("bf16_t6_h16_strong", 20260918, 6, 16, torch.bfloat16, 0.1, 60.0),
    )
    case_results: dict[str, Any] = {}
    passed = True
    for name, seed, time_steps, heads, dtype, state_scale, gate_scale in cases:
        cpu_inputs = _make_cpu(
            seed=seed,
            time_steps=time_steps,
            heads=heads,
            dtype=dtype,
            state_scale=state_scale,
            gate_scale=gate_scale,
        )
        expected_output, expected_state = _reference(cpu_inputs)
        device_inputs = _device_inputs(cpu_inputs, device)
        with torch.inference_mode():
            actual_output, actual_state = _run_cce(device_inputs)
        torch.npu.synchronize(device)
        output_diff = _diff(actual_output, expected_output)
        state_diff = _diff(actual_state, expected_state)
        output_budget = 1e-4 if dtype == torch.float32 else 1e-2
        state_budget = 1e-4
        case_passed = (
            output_diff["relative_l2"] <= output_budget
            and state_diff["relative_l2"] <= state_budget
        )
        passed &= case_passed
        case_results[name] = {
            "dtype": str(dtype),
            "shape": list(cpu_inputs["q"].shape),
            "output": output_diff,
            "final_state": state_diff,
            "output_relative_l2_budget": output_budget,
            "state_relative_l2_budget": state_budget,
            "passed": case_passed,
        }

    chain_cpu = _make_cpu(
        seed=20260919,
        time_steps=4,
        heads=16,
        dtype=torch.bfloat16,
        state_scale=0.1,
        gate_scale=20.0,
    )
    chain_device = _device_inputs(chain_cpu, device)
    with torch.inference_mode():
        full_output, full_state = _run_cce(chain_device)
        state = chain_device["initial_state"]
        pieces = []
        for index in range(4):
            step_inputs = {
                name: (
                    tensor[:, index:index + 1].contiguous()
                    if name != "initial_state"
                    else state
                )
                for name, tensor in chain_device.items()
            }
            step_output, state = _run_cce(step_inputs)
            pieces.append(step_output.detach().cpu())
    torch.npu.synchronize(device)
    chained_output = torch.cat(pieces, dim=1)
    chain_output_diff = _diff(chained_output, full_output)
    chain_state_diff = _diff(state, full_state)
    chain_passed = (
        chain_output_diff["relative_l2"] <= 1e-5
        and chain_state_diff["relative_l2"] <= 1e-5
    )
    passed &= chain_passed

    timing_cpu = _make_cpu(
        seed=20260920,
        time_steps=1,
        heads=16,
        dtype=torch.bfloat16,
        state_scale=0.1,
        gate_scale=10.0,
    )
    timing_inputs = _device_inputs(timing_cpu, device)
    with torch.inference_mode():
        cce_us = _timed(
            lambda: _run_cce(timing_inputs), device, args.warmup, args.iterations
        )
        torch_us = _timed(
            lambda: gdn2_recurrent_reference(
                timing_inputs["q"],
                timing_inputs["k"],
                timing_inputs["v"],
                timing_inputs["g"],
                timing_inputs["b"],
                timing_inputs["w"],
                timing_inputs["initial_state"],
                scale=SCALE,
                use_qk_l2norm=True,
                qk_norm_eps=QK_EPS,
            ),
            device,
            args.warmup,
            args.iterations,
        )

    result = {
        "device": str(device),
        "backend": "ascriptor-cce/aclnn",
        "block_dim": args.block_dim,
        "build_seconds": build_seconds,
        "cases": case_results,
        "state_chaining": {
            "output": chain_output_diff,
            "final_state": chain_state_diff,
            "relative_l2_budget": 1e-5,
            "passed": chain_passed,
        },
        "timing": {
            "shape": [1, 1, 16, 128],
            "dtype": "torch.bfloat16",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "cce_public_call_us": cce_us,
            "torch_npu_reference_us": torch_us,
            "speedup": torch_us / cce_us,
        },
        "passed": passed,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("GDN-2 CCE recurrent verification exceeded its numerical budget")


if __name__ == "__main__":
    main()
