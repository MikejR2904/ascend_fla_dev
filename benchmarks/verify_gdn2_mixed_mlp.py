#!/usr/bin/env python3
"""Compare vendor and mixed-CCE GDN-2 MLP decode paths on the real checkpoint.

Models are loaded and released sequentially.  The mixed custom OPP is prepared
before either model executes an NPU compute operator.  Prompt execution remains
the accepted vendor path in both variants; the token chain exercises only the
fixed B=T=1 mixed boundary and checks propagation through later layers/caches.
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
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[1, 450, 7483, 310, 3444, 338]
    )
    parser.add_argument(
        "--decode-tokens", type=int, nargs="+", default=[3681, 42, 900, 1234]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--order", choices=("vendor-first", "cce-first"), default="vendor-first"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= len(args.prompt_tokens) <= 16:
        parser.error("the CCE prompt route requires 1..16 prompt tokens")
    if not args.decode_tokens:
        parser.error("--decode-tokens must not be empty")
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations positive")
    return args


def _sync(device: torch.device) -> None:
    torch.npu.synchronize(device)


def _timed(
    function: Callable[[], Any], device: torch.device, warmup: int, iterations: int
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
        if layer is None or not isinstance(layer.conv_state, torch.Tensor):
            raise TypeError("mixed MLP verification requires populated packed caches")
        copied.append(
            {
                "recurrent_state": layer.recurrent_state.detach().cpu(),
                "conv_state": layer.conv_state.detach().cpu(),
                "offset": layer.offset,
            }
        )
    return copied


def _run(
    args: argparse.Namespace, backend: str, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint,
        config=GDN2Config.gdn2_1_3b(),
        device=device,
        dtype=torch.bfloat16,
        strict=True,
        core_backend="cce",
        projection_layout="packed-inference",
        mlp_backend="cce" if backend == "cce" else "torch",
    )
    _sync(device)
    load_seconds = time.perf_counter() - started
    prompt = torch.tensor([args.prompt_tokens], dtype=torch.long).to(device)
    steps = [torch.tensor([[token]], dtype=torch.long).to(device) for token in args.decode_tokens]
    with torch.inference_mode():
        prompt_logits, prompt_cache = model(prompt, return_cache=True)
        current = prompt_cache
        step_logits = []
        step_caches = []
        for step in steps:
            logits, current = model(step, cache=current, return_cache=True)
            step_logits.append(logits.detach().cpu())
            step_caches.append(_cache_cpu(current))
        decode_us = _timed(
            lambda: model(steps[0], cache=prompt_cache, return_cache=True),
            device,
            args.warmup,
            args.iterations,
        )
    derived_buffers = {
        name: tensor.numel() * tensor.element_size()
        for name, tensor in model.named_buffers()
        if name.endswith("paired_w12_weight")
    }
    payload = {
        "prompt_logits": prompt_logits.detach().cpu(),
        "prompt_cache": _cache_cpu(prompt_cache),
        "step_logits": step_logits,
        "step_caches": step_caches,
    }
    metrics = {
        "mlp_backend": model.mlp_backend,
        "load_seconds": load_seconds,
        "decode_us": decode_us,
        "parameter_count": model.parameter_count,
        "state_dict_entries": len(model.state_dict()),
        "derived_paired_weight_buffers": len(derived_buffers),
        "derived_paired_weight_bytes": sum(derived_buffers.values()),
        "memory_allocated_bytes": torch.npu.memory_allocated(device),
        "memory_reserved_bytes": torch.npu.memory_reserved(device),
        "prompt_argmax": int(payload["prompt_logits"][:, -1].argmax(dim=-1)[0]),
        "decode_argmax": [int(logits[:, -1].argmax(dim=-1)[0]) for logits in step_logits],
    }
    del model, prompt, steps, prompt_logits, prompt_cache, current, logits
    gc.collect()
    torch.npu.empty_cache()
    _sync(device)
    return metrics, payload


def _diff(got: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = got.float() - expected.float()
    return {
        "bitwise_equal": bool(torch.equal(got, expected)),
        "max_abs_diff": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-30)),
    }


def _compare_cache(
    got: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, float | bool]:
    recurrent = [_diff(g["recurrent_state"], e["recurrent_state"]) for g, e in zip(got, expected)]
    convolution = [_diff(g["conv_state"], e["conv_state"]) for g, e in zip(got, expected)]
    return {
        "recurrent_relative_l2_max": max(x["relative_l2"] for x in recurrent),
        "recurrent_max_abs_diff": max(x["max_abs_diff"] for x in recurrent),
        "conv_relative_l2_max": max(x["relative_l2"] for x in convolution),
        "conv_max_abs_diff": max(x["max_abs_diff"] for x in convolution),
        "offsets_equal": all(g["offset"] == e["offset"] for g, e in zip(got, expected)),
    }


def main() -> None:
    args = _parse_args()
    if args.device.split(":", 1)[0] != "npu":
        raise ValueError(f"verification requires an NPU, got {args.device}")
    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    # Claim the shared card immediately. Allocation-only torch.empty does not
    # resolve an operator, so custom OPP paths can still be registered safely.
    reservation = torch.empty((1,), dtype=torch.uint8, device=device)
    # CANN reads the custom OPP search path once.  Register every model kernel,
    # including the optional mixed MLP, before any model executes.
    prepare(block_dim=8, mixed_mlp=True)
    order = ("vendor", "cce") if args.order == "vendor-first" else ("cce", "vendor")
    runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for backend in order:
        runs[backend] = _run(args, backend, device)
        print(f"{backend}_complete", flush=True)

    vendor_metrics, vendor = runs["vendor"]
    cce_metrics, cce = runs["cce"]
    prompt_cache = _compare_cache(cce["prompt_cache"], vendor["prompt_cache"])
    step_logits = [
        _diff(got, expected)
        for got, expected in zip(cce["step_logits"], vendor["step_logits"])
    ]
    step_caches = [
        _compare_cache(got, expected)
        for got, expected in zip(cce["step_caches"], vendor["step_caches"])
    ]
    comparison = {
        "prompt_logits": _diff(cce["prompt_logits"], vendor["prompt_logits"]),
        "prompt_cache": prompt_cache,
        "step_logits": step_logits,
        "step_logits_relative_l2_max": max(x["relative_l2"] for x in step_logits),
        "step_caches": step_caches,
        "step_recurrent_relative_l2_max": max(
            x["recurrent_relative_l2_max"] for x in step_caches
        ),
        "step_conv_relative_l2_max": max(x["conv_relative_l2_max"] for x in step_caches),
        "all_offsets_equal": prompt_cache["offsets_equal"]
        and all(x["offsets_equal"] for x in step_caches),
    }
    budget = 1e-2
    passed = all(
        (
            comparison["prompt_logits"]["relative_l2"] <= budget,
            comparison["prompt_cache"]["recurrent_relative_l2_max"] <= budget,
            comparison["prompt_cache"]["conv_relative_l2_max"] <= budget,
            comparison["step_logits_relative_l2_max"] <= budget,
            comparison["step_recurrent_relative_l2_max"] <= budget,
            comparison["step_conv_relative_l2_max"] <= budget,
            comparison["all_offsets_equal"],
        )
    )
    result = {
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "dtype": "bfloat16",
        "core_backend": "cce",
        "projection_layout": "packed-inference",
        "order": args.order,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "vendor": vendor_metrics,
        "cce": cce_metrics,
        "eager_decode_speedup": vendor_metrics["decode_us"] / cce_metrics["decode_us"],
        "comparison": comparison,
        "acceptance": {
            "relative_l2_budget": budget,
            "rule": "same-dtype relative-L2 for logits and both cache kinds; offsets exact",
        },
        "passed": passed,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("mixed MLP whole-model path exceeded the BF16 numerical budget")
    assert reservation.numel() == 1


if __name__ == "__main__":
    main()
