#!/usr/bin/env python3
"""用同一 checkpoint 验证 GDN-2 canonical 与 packed-inference 的数值一致性和 decode 耗时。

两种布局顺序加载，避免为比较而在 NPU 上同时常驻两份模型。
正式闸使用同 dtype relative-L2 预算；bitwise 与 argmax 只作额外诊断。
物理卡选择由调用环境负责；脚本内部只看逻辑 ``--device``，不会写死共享机器信息。
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


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn: Callable[[], Any], device: torch.device, warmup: int, iterations: int) -> float:
    last = None
    for _ in range(warmup):
        last = fn()
    _sync(device)
    started = time.perf_counter()
    for _ in range(iterations):
        last = fn()
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


def _run_layout(
    args: argparse.Namespace,
    layout: str,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    started = time.perf_counter()
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint,
        config=GDN2Config.gdn2_1_3b(),
        device=device,
        dtype=dtype,
        strict=True,
        core_backend=args.core_backend,
        projection_layout=layout,
    )
    _sync(device)
    load_seconds = time.perf_counter() - started
    if args.synthetic_length is None:
        prompt_tokens = args.prompt_tokens
    else:
        prompt_tokens = [
            int(value)
            for value in ((torch.arange(args.synthetic_length) * 37 + 1) % 32_000)
        ]
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    step = torch.tensor([[args.step_token]], dtype=torch.long).to(device)
    with torch.inference_mode():
        prompt_logits, prompt_cache = model(input_ids, return_cache=True)
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
    metrics: dict[str, Any] = {
        "layout": model.projection_layout,
        "core_backend": model.core_backend,
        "dtype": str(dtype),
        "load_seconds": load_seconds,
        "decode_us": decode_us,
        "parameter_count": model.parameter_count,
        "state_dict_entries": len(model.state_dict()),
        "prompt_argmax": int(payload["prompt_logits"][:, -1].argmax(dim=-1)[0]),
        "step_argmax": int(payload["step_logits"][:, -1].argmax(dim=-1)[0]),
    }
    if device.type == "npu":
        metrics["memory_allocated_bytes"] = torch.npu.memory_allocated(device)
        metrics["memory_reserved_bytes"] = torch.npu.memory_reserved(device)
    del model, input_ids, step, prompt_logits, step_logits, prompt_cache, step_cache
    gc.collect()
    if device.type == "npu":
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
    canonical: list[dict[str, Any]],
    packed: list[dict[str, Any]],
) -> dict[str, Any]:
    recurrent_equal = True
    conv_equal = True
    offsets_equal = True
    recurrent_max_abs = 0.0
    conv_max_abs = 0.0
    recurrent_relative_l2_max = 0.0
    conv_relative_l2_max = 0.0
    for canonical_layer, packed_layer in zip(canonical, packed):
        canonical_state = canonical_layer["recurrent_state"]
        packed_state = packed_layer["recurrent_state"]
        recurrent_equal &= torch.equal(canonical_state, packed_state)
        recurrent_max_abs = max(
            recurrent_max_abs,
            float((canonical_state.float() - packed_state.float()).abs().max()),
        )
        recurrent_relative_l2_max = max(
            recurrent_relative_l2_max,
            float(
                (canonical_state.float() - packed_state.float()).norm()
                / canonical_state.float().norm().clamp_min(1e-30)
            ),
        )
        canonical_conv = canonical_layer["conv_state"]
        packed_conv = packed_layer["conv_state"]
        if isinstance(canonical_conv, tuple):
            canonical_conv = torch.cat(canonical_conv, dim=1)
        assert isinstance(canonical_conv, torch.Tensor)
        assert isinstance(packed_conv, torch.Tensor)
        conv_equal &= torch.equal(canonical_conv, packed_conv)
        conv_max_abs = max(
            conv_max_abs,
            float((canonical_conv.float() - packed_conv.float()).abs().max()),
        )
        conv_relative_l2_max = max(
            conv_relative_l2_max,
            float(
                (canonical_conv.float() - packed_conv.float()).norm()
                / canonical_conv.float().norm().clamp_min(1e-30)
            ),
        )
        offsets_equal &= canonical_layer["offset"] == packed_layer["offset"]
    return {
        "recurrent_state_bitwise_equal": recurrent_equal,
        "recurrent_state_max_abs_diff": recurrent_max_abs,
        "recurrent_state_relative_l2_max": recurrent_relative_l2_max,
        "conv_state_bitwise_equal": conv_equal,
        "conv_state_max_abs_diff": conv_max_abs,
        "conv_state_relative_l2_max": conv_relative_l2_max,
        "offsets_equal": offsets_equal,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--core-backend", choices=GDN2ForCausalLM.core_backends, default="torch"
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[1, 450, 7483, 310, 3444, 338],
    )
    parser.add_argument("--synthetic-length", type=int)
    parser.add_argument("--step-token", type=int, default=3681)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--order",
        choices=("canonical-first", "packed-first"),
        default="canonical-first",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.synthetic_length is not None and args.synthetic_length <= 0:
        parser.error("--synthetic-length 必须为正数")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup 必须非负，--iterations 必须为正数")
    return args


def main() -> None:
    args = _parse_args()
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu  # noqa: F401
    device = torch.device(args.device)

    layouts = (
        ("canonical", "packed-inference")
        if args.order == "canonical-first"
        else ("packed-inference", "canonical")
    )
    runs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for layout in layouts:
        runs[layout] = _run_layout(args, layout, device)
        print(f"{layout}_complete", flush=True)
    canonical_metrics, canonical = runs["canonical"]
    packed_metrics, packed = runs["packed-inference"]
    result = {
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "core_backend": args.core_backend,
        "order": args.order,
        "prompt_length": args.synthetic_length or len(args.prompt_tokens),
        "canonical": canonical_metrics,
        "packed": packed_metrics,
        "decode_speedup": canonical_metrics["decode_us"] / packed_metrics["decode_us"],
        "equivalence": {
            "prompt_logits": _diff(packed["prompt_logits"], canonical["prompt_logits"]),
            "step_logits": _diff(packed["step_logits"], canonical["step_logits"]),
            "prompt_cache": _compare_caches(
                canonical["prompt_cache"], packed["prompt_cache"]
            ),
            "step_cache": _compare_caches(canonical["step_cache"], packed["step_cache"]),
        },
    }
    relative_l2_budget = 1e-5 if args.dtype == "float32" else 1e-2
    passed = all(
        (
            result["equivalence"]["prompt_logits"]["relative_l2"] <= relative_l2_budget,
            result["equivalence"]["step_logits"]["relative_l2"] <= relative_l2_budget,
            result["equivalence"]["prompt_cache"]["recurrent_state_relative_l2_max"]
            <= relative_l2_budget,
            result["equivalence"]["prompt_cache"]["conv_state_relative_l2_max"]
            <= relative_l2_budget,
            result["equivalence"]["step_cache"]["recurrent_state_relative_l2_max"]
            <= relative_l2_budget,
            result["equivalence"]["step_cache"]["conv_state_relative_l2_max"]
            <= relative_l2_budget,
            result["equivalence"]["prompt_cache"]["offsets_equal"],
            result["equivalence"]["step_cache"]["offsets_equal"],
        )
    )
    result["acceptance"] = {
        "rule": "same-dtype layout-vs-layout relative_l2 plus cache offsets; bitwise and argmax are diagnostic only",
        "relative_l2_budget": relative_l2_budget,
    }
    result["passed"] = passed
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("canonical 与 packed-inference 超出数值一致性预算")


if __name__ == "__main__":
    main()
