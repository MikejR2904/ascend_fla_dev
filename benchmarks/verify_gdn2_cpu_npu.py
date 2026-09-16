#!/usr/bin/env python3
"""用 tiny GDN-2 FP32 模型验证 torch_npu 基线对 CPU oracle 的数值误差。

CPU 与 NPU 可以采用不同规约路径；本脚本只以 relative-L2 预算验收 logits、recurrent
cache 和 short-conv cache，不要求逐位相同。物理卡选择由调用环境负责。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.models import GDN2Config, GDN2ForCausalLM, GDN2LayerCache


def _config() -> GDN2Config:
    return GDN2Config(
        vocab_size=32,
        padding_multiple=8,
        block_size=32,
        n_layer=2,
        n_embd=16,
        intermediate_size=24,
        num_heads=2,
        num_v_heads=2,
        head_dim=4,
        conv_size=3,
    )


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().float()


def _diff(got: torch.Tensor, want: torch.Tensor) -> dict[str, float]:
    got_cpu = _cpu(got)
    want_cpu = _cpu(want)
    delta = got_cpu - want_cpu
    return {
        "max_abs_diff": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / want_cpu.norm().clamp_min(1e-30)),
    }


def _conv_cpu(layer: GDN2LayerCache) -> torch.Tensor:
    assert isinstance(layer.conv_state, tuple)
    # 每份连续 tensor 先完整 D2H，再在 CPU 拼接；不要依赖 NPU 侧额外的 cat。
    return torch.cat([_cpu(tensor) for tensor in layer.conv_state], dim=1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--budget", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget 必须为正数")
    return args


def main() -> None:
    args = _parse_args()
    if args.device.split(":", 1)[0] != "npu":
        raise ValueError(f"本验证要求 torch_npu device，收到 {args.device}")
    import torch_npu  # noqa: F401  # 注册 torch.npu backend

    device = torch.device(args.device)
    if not torch.npu.is_available():
        raise RuntimeError("没有可用的 NPU")

    torch.manual_seed(2026)
    cpu_model = GDN2ForCausalLM(_config()).float().eval()
    device_model = GDN2ForCausalLM(_config()).float().eval()
    device_model.load_state_dict(cpu_model.state_dict())
    device_model = device_model.to(device)
    prompt_cpu = torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
    step_cpu = torch.tensor([[2]], dtype=torch.long)

    with torch.inference_mode():
        cpu_logits, cpu_cache = cpu_model(prompt_cpu, return_cache=True)
        cpu_step, cpu_next = cpu_model(step_cpu, cache=cpu_cache, return_cache=True)
        device_logits, device_cache = device_model(
            prompt_cpu.to(device), return_cache=True
        )
        device_step, device_next = device_model(
            step_cpu.to(device), cache=device_cache, return_cache=True
        )
    torch.npu.synchronize(device)

    checks: dict[str, dict[str, float]] = {
        "prompt_logits": _diff(device_logits, cpu_logits),
        "step_logits": _diff(device_step, cpu_step),
    }
    offsets_equal = True
    for index, (cpu_layer, device_layer, cpu_next_layer, device_next_layer) in enumerate(
        zip(cpu_cache, device_cache, cpu_next, device_next)
    ):
        assert cpu_layer is not None and device_layer is not None
        assert cpu_next_layer is not None and device_next_layer is not None
        checks[f"prompt_recurrent_{index}"] = _diff(
            device_layer.recurrent_state, cpu_layer.recurrent_state
        )
        checks[f"step_recurrent_{index}"] = _diff(
            device_next_layer.recurrent_state, cpu_next_layer.recurrent_state
        )
        checks[f"prompt_conv_{index}"] = _diff(
            _conv_cpu(device_layer), _conv_cpu(cpu_layer)
        )
        checks[f"step_conv_{index}"] = _diff(
            _conv_cpu(device_next_layer), _conv_cpu(cpu_next_layer)
        )
        offsets_equal &= device_layer.offset == cpu_layer.offset == prompt_cpu.shape[1]
        offsets_equal &= (
            device_next_layer.offset
            == cpu_next_layer.offset
            == prompt_cpu.shape[1] + 1
        )

    max_relative_l2 = max(check["relative_l2"] for check in checks.values())
    result: dict[str, Any] = {
        "device": str(device),
        "dtype": "torch.float32",
        "rule": "CPU-vs-torch_npu relative_l2 plus cache offsets; bitwise is not required",
        "relative_l2_budget": args.budget,
        "max_relative_l2": max_relative_l2,
        "offsets_equal": offsets_equal,
        "checks": checks,
        "passed": offsets_equal and max_relative_l2 <= args.budget,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError("torch_npu GDN-2 基线超出 CPU fp32 数值预算")


if __name__ == "__main__":
    main()
