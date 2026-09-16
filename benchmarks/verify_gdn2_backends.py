#!/usr/bin/env python3
"""Compare the full GDN-2 torch_npu and Ascriptor CCE recurrent backends.

Both models use the same checkpoint and projection layout and are loaded sequentially.
The CCE custom OPP is prepared before either model executes an NPU compute operator.
Correctness uses same-dtype relative-L2 budgets; bitwise and argmax are diagnostics only.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.models import GDN2Config, GDN2ForCausalLM, GDN2LayerCache
from ascend_fla.ops.gdn2 import prepare


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--projection-layout",
        choices=GDN2ForCausalLM.projection_layouts,
        default="packed-inference",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[1, 450, 7483, 310, 3444, 338],
    )
    parser.add_argument("--step-token", type=int, default=3681)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--order", choices=("torch-first", "cce-first"), default="torch-first"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= len(args.prompt_tokens) <= 16:
        parser.error("the CCE recurrent first scope requires 1..16 prompt tokens")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations must be positive")
    return args


def _sync(device: torch.device) -> None:
    torch.npu.synchronize(device)


def _timed(
    function: Callable[[], Any],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    last = None
    for _ in range(warmup):
        last = function()
    _sync(device)
    started = time.perf_counter()
    for _ in range(iterations):
        last = function()
    _sync(device)
    del last
    return (time.perf_counter() - started) * 1e6 / iterations


def _cache_cpu(cache: list[GDN2LayerCache | None]) -> list[dict[str, Any]]:
    copied = []
    for layer in cache:
        assert layer is not None
        conv = layer.conv_state
        if isinstance(conv, tuple):
            conv = tuple(tensor.detach().cpu() for tensor in conv)
        elif isinstance(conv, torch.Tensor):
            conv = conv.detach().cpu()
        copied.append(
            {
                "recurrent_state": layer.recurrent_state.detach().cpu(),
                "conv_state": conv,
                "offset": layer.offset,
            }
        )
    return copied


def _run_backend(
    args: argparse.Namespace, backend: str, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any]]:
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    started = time.perf_counter()
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint,
        config=GDN2Config.gdn2_1_3b(),
        device=device,
        dtype=dtype,
        strict=True,
        core_backend=backend,
        projection_layout=args.projection_layout,
    )
    _sync(device)
    load_seconds = time.perf_counter() - started
    prompt = torch.tensor([args.prompt_tokens], dtype=torch.long).to(device)
    step = torch.tensor([[args.step_token]], dtype=torch.long).to(device)
    with torch.inference_mode():
        prompt_logits, prompt_cache = model(prompt, return_cache=True)
        step_logits, step_cache = model(step, cache=prompt_cache, return_cache=True)
        decode_us = _timed(
            lambda: model(step, cache=prompt_cache, return_cache=True),
            device,
            args.warmup,
            args.iterations,
        )
    payload = {
        "prompt_logits": prompt_logits.detach().cpu(),
        "step_logits": step_logits.detach().cpu(),
        "prompt_cache": _cache_cpu(prompt_cache),
        "step_cache": _cache_cpu(step_cache),
    }
    metrics = {
        "core_backend": backend,
        "load_seconds": load_seconds,
        "decode_us": decode_us,
        "prompt_argmax": int(payload["prompt_logits"][:, -1].argmax(dim=-1)[0]),
        "step_argmax": int(payload["step_logits"][:, -1].argmax(dim=-1)[0]),
        "memory_allocated_bytes": torch.npu.memory_allocated(device),
        "memory_reserved_bytes": torch.npu.memory_reserved(device),
    }
    del model, prompt, step, prompt_logits, step_logits, prompt_cache, step_cache
    gc.collect()
    torch.npu.empty_cache()
    _sync(device)
    return metrics, payload


def _diff(got: torch.Tensor, want: torch.Tensor) -> dict[str, float | bool]:
    delta = got.float() - want.float()
    return {
        "bitwise_equal": bool(torch.equal(got, want)),
        "max_abs_diff": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / want.float().norm().clamp_min(1e-30)),
    }


def _compare_caches(
    torch_cache: list[dict[str, Any]], cce_cache: list[dict[str, Any]]
) -> dict[str, Any]:
    recurrent = []
    convolution = []
    offsets_equal = len(torch_cache) == len(cce_cache)
    for torch_layer, cce_layer in zip(torch_cache, cce_cache):
        recurrent.append(
            _diff(cce_layer["recurrent_state"], torch_layer["recurrent_state"])
        )
        torch_conv = torch_layer["conv_state"]
        cce_conv = cce_layer["conv_state"]
        if isinstance(torch_conv, tuple):
            torch_conv = torch.cat(torch_conv, dim=1)
        if isinstance(cce_conv, tuple):
            cce_conv = torch.cat(cce_conv, dim=1)
        assert isinstance(torch_conv, torch.Tensor)
        assert isinstance(cce_conv, torch.Tensor)
        convolution.append(_diff(cce_conv, torch_conv))
        offsets_equal &= torch_layer["offset"] == cce_layer["offset"]
    return {
        "recurrent_relative_l2_max": max(item["relative_l2"] for item in recurrent),
        "recurrent_max_abs_diff": max(item["max_abs_diff"] for item in recurrent),
        "conv_relative_l2_max": max(item["relative_l2"] for item in convolution),
        "conv_max_abs_diff": max(item["max_abs_diff"] for item in convolution),
        "recurrent_bitwise_equal": all(item["bitwise_equal"] for item in recurrent),
        "conv_bitwise_equal": all(item["bitwise_equal"] for item in convolution),
        "offsets_equal": offsets_equal,
    }


def main() -> None:
    args = _parse_args()
    if args.device.split(":", 1)[0] != "npu":
        raise ValueError(f"this verification requires an NPU, got {args.device}")
    import torch_npu  # noqa: F401

    # Register the custom OPP before even the torch-first order executes built-in ops.
    prepare(block_dim=8)
    device = torch.device(args.device)
    order = ("torch", "cce") if args.order == "torch-first" else ("cce", "torch")
    runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for backend in order:
        runs[backend] = _run_backend(args, backend, device)
        print(f"{backend}_complete", flush=True)

    torch_metrics, torch_payload = runs["torch"]
    cce_metrics, cce_payload = runs["cce"]
    comparison = {
        "prompt_logits": _diff(
            cce_payload["prompt_logits"], torch_payload["prompt_logits"]
        ),
        "step_logits": _diff(cce_payload["step_logits"], torch_payload["step_logits"]),
        "prompt_cache": _compare_caches(
            torch_payload["prompt_cache"], cce_payload["prompt_cache"]
        ),
        "step_cache": _compare_caches(
            torch_payload["step_cache"], cce_payload["step_cache"]
        ),
    }
    budget = 1e-3 if args.dtype == "float32" else 1e-2
    passed = all(
        (
            comparison["prompt_logits"]["relative_l2"] <= budget,
            comparison["step_logits"]["relative_l2"] <= budget,
            comparison["prompt_cache"]["recurrent_relative_l2_max"] <= budget,
            comparison["prompt_cache"]["conv_relative_l2_max"] <= budget,
            comparison["step_cache"]["recurrent_relative_l2_max"] <= budget,
            comparison["step_cache"]["conv_relative_l2_max"] <= budget,
            comparison["prompt_cache"]["offsets_equal"],
            comparison["step_cache"]["offsets_equal"],
        )
    )
    result = {
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "dtype": args.dtype,
        "projection_layout": args.projection_layout,
        "order": args.order,
        "torch": torch_metrics,
        "cce": cce_metrics,
        "decode_speedup": torch_metrics["decode_us"] / cce_metrics["decode_us"],
        "comparison": comparison,
        "acceptance": {
            "relative_l2_budget": budget,
            "rule": "same-dtype relative_l2 plus cache offsets; bitwise and argmax are diagnostic only",
        },
        "passed": passed,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("full-model CCE recurrent backend exceeded the numerical budget")


if __name__ == "__main__":
    main()
