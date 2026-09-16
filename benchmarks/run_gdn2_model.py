#!/usr/bin/env python3
"""严格加载 GDN-2 checkpoint，并用 torch / torch_npu 跑完整模型冒烟。

示例（路径由调用环境决定，不把机器路径固化进仓库）：

    python benchmarks/run_gdn2_model.py --checkpoint <checkpoint.pth> --device npu:0

默认只跑两个 token；逐 token reference scan 的目标是证明模型和调用链可用，不是测性能。
原始输出可用 ``--output`` 留作实验记录。
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.models import GDN2Config, GDN2ForCausalLM


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--core-backend", choices=GDN2ForCausalLM.core_backends, default="torch"
    )
    parser.add_argument(
        "--projection-layout",
        choices=GDN2ForCausalLM.projection_layouts,
        default="canonical",
    )
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu  # noqa: F401  # 注册 PrivateUse1/NPU backend
    device = torch.device(args.device)

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    checkpoint_size = args.checkpoint.stat().st_size
    started = time.perf_counter()
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint,
        config=GDN2Config.gdn2_1_3b(),
        device=device,
        dtype=dtype,
        strict=True,
        core_backend=args.core_backend,
        projection_layout=args.projection_layout,
    )
    _sync(device)
    load_seconds = time.perf_counter() - started

    input_ids = torch.tensor([args.tokens], dtype=torch.long, device="cpu").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits, cache = model(input_ids, return_cache=True)
    _sync(device)
    forward_seconds = time.perf_counter() - started
    logits_cpu = logits.detach().cpu().float()
    dtype_counts = Counter()
    for parameter in model.parameters():
        dtype_counts[str(parameter.dtype)] += parameter.numel()

    result: dict[str, object] = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_bytes": checkpoint_size,
        "device": str(device),
        "dtype": str(dtype),
        "projection_layout": model.projection_layout,
        "core_backend": model.core_backend,
        "parameter_count": model.parameter_count,
        "reported_parameter_count": model.reported_parameter_count,
        "checkpoint_metadata": model.checkpoint_metadata,
        "parameter_dtypes": dict(sorted(dtype_counts.items())),
        "input_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits_cpu).all()),
        "logits_min": float(logits_cpu.min()),
        "logits_max": float(logits_cpu.max()),
        "last_token_argmax": int(logits_cpu[:, -1].argmax(dim=-1)[0]),
        "cache_layers": len(cache),
        "recurrent_state_shape": list(cache[0]["recurrent_state"].shape),
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
    }
    if device.type == "npu":
        result["npu_memory"] = {
            "allocated_bytes": torch.npu.memory_allocated(device),
            "reserved_bytes": torch.npu.memory_reserved(device),
        }

    if args.check_cache:
        step_cache = None
        pieces = []
        started = time.perf_counter()
        with torch.inference_mode():
            for index in range(input_ids.shape[1]):
                step_logits, step_cache = model(
                    input_ids[:, index:index + 1], cache=step_cache, return_cache=True
                )
                pieces.append(step_logits)
        _sync(device)
        # 每个连续输出先完整 D2H，再在 CPU 上 cast/cat；兼容缺少 NPU Cast 的环境。
        stepped = torch.cat(
            [piece.detach().cpu().float() for piece in pieces], dim=1
        )
        diff = stepped - logits_cpu
        result["cache_check"] = {
            "max_abs_diff": float(diff.abs().max()),
            "relative_l2": float(diff.norm() / logits_cpu.norm().clamp_min(1e-30)),
            "seconds": time.perf_counter() - started,
        }

    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["logits_finite"]:
        raise RuntimeError("完整模型输出含 NaN/Inf")


if __name__ == "__main__":
    main()
