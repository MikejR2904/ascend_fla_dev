#!/usr/bin/env python3
"""严格加载 GDN-2 checkpoint，并用本地 tokenizer 做 torch_npu 文本生成与计时。

示例（设备与路径由调用环境指定）：

    python benchmarks/generate_gdn2_text.py \
      --checkpoint <checkpoint.pth> --tokenizer <tokenizer-dir> --device npu:0

``--decode-backend eager`` 保留逐 token 基线；``npu-graph`` 计入 prefill、capture、logits
D2H、CPU sampling 与 token H2D，并另外报告排除一次性 setup 后的 decode 调用吞吐。
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.models import (
    GDN2Config,
    GDN2ForCausalLM,
    GDN2_DECODE_BACKENDS,
    generate_tokens,
)


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--core-backend", choices=GDN2ForCausalLM.core_backends, default="torch"
    )
    parser.add_argument(
        "--projection-layout",
        choices=GDN2ForCausalLM.projection_layouts,
        default="canonical",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--decode-backend", choices=GDN2_DECODE_BACKENDS, default="eager"
    )
    parser.add_argument("--graph-warmup", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.decode_backend == "npu-graph":
        required = {
            "device": args.device.split(":", 1)[0] == "npu",
            "dtype": args.dtype == "bfloat16",
            "core_backend": args.core_backend == "cce",
            "projection_layout": args.projection_layout == "packed-inference",
        }
        failed = [name for name, matched in required.items() if not matched]
        if failed:
            parser.error(
                "--decode-backend npu-graph 要求 NPU/BF16/CCE/packed-inference；"
                f"不匹配：{failed}"
            )
    return args


def main() -> None:
    args = _parse_args()
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu  # noqa: F401  # 注册 PrivateUse1/NPU backend
    from transformers import AutoTokenizer

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.vocab_size != GDN2Config.gdn2_1_3b().vocab_size:
        raise RuntimeError(
            f"tokenizer vocab_size 应为 32000，收到 {tokenizer.vocab_size}"
        )
    encoded = tokenizer(
        args.prompt,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids
    if encoded.shape[1] == 0:
        raise ValueError("prompt 经 tokenizer 后为空")

    load_started = time.perf_counter()
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
    load_seconds = time.perf_counter() - load_started

    input_ids = encoded.to(device=device, dtype=torch.long)
    generation_started = time.perf_counter()
    result = generate_tokens(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=tokenizer.eos_token_id,
        seed=args.seed,
        decode_backend=args.decode_backend,
        graph_warmup=args.graph_warmup,
    )
    _sync(device)
    generation_seconds = time.perf_counter() - generation_started

    full_ids = result.token_ids[0].tolist()
    continuation_ids = full_ids[result.prompt_tokens:]
    dtype_counts = Counter()
    for parameter in model.parameters():
        dtype_counts[str(parameter.dtype)] += parameter.numel()
    record: dict[str, object] = {
        "checkpoint": args.checkpoint.name,
        "checkpoint_metadata": model.checkpoint_metadata,
        "device": str(device),
        "dtype": str(dtype),
        "projection_layout": model.projection_layout,
        "core_backend": model.core_backend,
        "decode_backend": result.decode_backend,
        "parameter_count": model.parameter_count,
        "parameter_dtypes": dict(sorted(dtype_counts.items())),
        "tokenizer": {
            "path_name": args.tokenizer.name,
            "class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "prompt_has_bos": bool(full_ids and full_ids[0] == tokenizer.bos_token_id),
        },
        "prompt": args.prompt,
        "prompt_token_ids": full_ids[:result.prompt_tokens],
        "continuation_token_ids": continuation_ids,
        "continuation": tokenizer.decode(continuation_ids, skip_special_tokens=True),
        "full_text": tokenizer.decode(full_ids, skip_special_tokens=True),
        "generated_tokens": result.generated_tokens,
        "stopped_on_eos": result.stopped_on_eos,
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "generated_tokens_per_second": (
            result.generated_tokens / generation_seconds if generation_seconds else None
        ),
    }
    if result.timings is not None:
        timing_record = asdict(result.timings)
        decode_calls = result.timings.decode_model_calls
        decode_seconds = result.timings.decode_loop_seconds
        timing_record["decode_model_calls_per_second"] = (
            decode_calls / decode_seconds if decode_seconds else None
        )
        record["generation_timings"] = timing_record
    if device.type == "npu":
        record["npu_memory"] = {
            "allocated_bytes": torch.npu.memory_allocated(device),
            "reserved_bytes": torch.npu.memory_reserved(device),
        }

    text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
