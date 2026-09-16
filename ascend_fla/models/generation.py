"""GDN-2 模型的自回归解码辅助函数。

采样刻意放在 CPU 上，因此端到端计时包含每步 logits D2H、CPU 采样与下一 token H2D。
``decode_backend="eager"`` 是通用基线；``"npu-graph"`` 只接受 GDN-2 已验证的
B1/T1/BF16/packed/CCE 定尺，并用固定地址 cache 做 stateful replay。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
from torch import nn

from .gdn2_graph import GDN2NPUGraphDecodeRunner

__all__ = [
    "GDN2_DECODE_BACKENDS",
    "GDN2GenerationResult",
    "GDN2GenerationTimings",
    "GDN2NPUGraphDecodeRunner",
    "generate_tokens",
]

GDN2_DECODE_BACKENDS = ("eager", "npu-graph")


@dataclass(frozen=True)
class GDN2GenerationTimings:
    """生成内部的同步 wall-time 分解。"""

    prefill_seconds: float
    decode_setup_seconds: float
    decode_loop_seconds: float
    decode_model_calls: int


@dataclass(frozen=True)
class GDN2GenerationResult:
    """一次 batch=1 生成的 token 结果，``token_ids`` 常驻 CPU。"""

    token_ids: torch.Tensor
    prompt_tokens: int
    generated_tokens: int
    stopped_on_eos: bool
    decode_backend: str = "eager"
    timings: GDN2GenerationTimings | None = None


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """从一行 CPU fp32 logits 选下一个 token，返回 ``[1,1]`` long tensor。"""
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("生成 logits 含 NaN/Inf")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits / temperature
    if top_k:
        values = torch.topk(scores, k=min(top_k, scores.shape[-1]), dim=-1).values
        scores = scores.masked_fill(scores < values[:, -1:], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.inference_mode()
def generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    eos_token_id: int | None = None,
    seed: int = 1234,
    decode_backend: str = "eager",
    graph_warmup: int = 3,
) -> GDN2GenerationResult:
    """用一次 prefill 加逐 token recurrent cache 生成文本。

    当前明确只支持 batch=1。模型计算留在 ``input_ids`` 所在设备；每步仅把最后一行
    logits 搬到 CPU 做采样。NPU Graph 路径不满足窄契约时直接报错，不回退 eager。
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError(f"input_ids 应为非空的 [1,T]，收到 {tuple(input_ids.shape)}")
    if input_ids.dtype != torch.long:
        raise TypeError(f"input_ids 必须是 torch.long，收到 {input_ids.dtype}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens 必须大于等于 0，收到 {max_new_tokens}")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature 必须是有限非负数，收到 {temperature}")
    if top_k < 0:
        raise ValueError(f"top_k 必须大于等于 0，收到 {top_k}")
    if decode_backend not in GDN2_DECODE_BACKENDS:
        raise ValueError(
            f"decode_backend 只支持 {GDN2_DECODE_BACKENDS}，收到 {decode_backend!r}；"
            "不能静默回退"
        )
    if decode_backend == "npu-graph" and graph_warmup < 1:
        raise ValueError(f"graph_warmup 必须为正数，收到 {graph_warmup}")

    config = getattr(model, "config", None)
    vocab_size = getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("model.config.vocab_size 必须是正整数")
    if eos_token_id is not None and not 0 <= eos_token_id < vocab_size:
        raise ValueError(f"eos_token_id 必须位于 [0,{vocab_size})，收到 {eos_token_id}")

    prompt_cpu = input_ids.detach().cpu()
    min_token, max_token = int(prompt_cpu.min()), int(prompt_cpu.max())
    if min_token < 0 or max_token >= vocab_size:
        raise ValueError(
            f"input_ids 必须位于 [0,{vocab_size})，实际最小/最大值为 {min_token}/{max_token}"
        )
    if max_new_tokens == 0:
        return GDN2GenerationResult(
            prompt_cpu,
            prompt_cpu.shape[1],
            0,
            False,
            decode_backend=decode_backend,
            timings=GDN2GenerationTimings(0.0, 0.0, 0.0, 0),
        )

    model.eval()
    prefill_started = time.perf_counter()
    logits, cache = model(input_ids, use_cache=True, return_cache=True)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] < vocab_size:
        raise RuntimeError(
            f"模型 logits 应为 [1,T,V] 且 V>={vocab_size}，收到 {tuple(logits.shape)}"
        )
    # 先整块 D2H，再在 CPU 上切片/转型；兼容不具备 NPU Slice/Cast 内置包的环境。
    next_logits = logits.detach().cpu().float()[:, -1, :vocab_size]
    prefill_seconds = time.perf_counter() - prefill_started
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)
    pieces = [prompt_cpu]
    stopped_on_eos = False
    graph_runner: GDN2NPUGraphDecodeRunner | None = None
    decode_setup_seconds = 0.0
    decode_model_calls = 0
    decode_loop_started = time.perf_counter()

    for index in range(max_new_tokens):
        next_cpu = _sample_next_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            generator=cpu_generator,
        )
        pieces.append(next_cpu)
        if eos_token_id is not None and int(next_cpu.item()) == eos_token_id:
            stopped_on_eos = True
            break
        if index + 1 == max_new_tokens:
            break

        if decode_backend == "npu-graph":
            if graph_runner is None:
                setup_started = time.perf_counter()
                graph_runner = GDN2NPUGraphDecodeRunner(
                    model,
                    cache,
                    sample_input=next_cpu,
                    warmup=graph_warmup,
                )
                decode_setup_seconds += time.perf_counter() - setup_started
            logits = graph_runner.step(next_cpu)
        else:
            next_device = next_cpu.to(device=input_ids.device)
            logits, cache = model(
                next_device,
                cache=cache,
                use_cache=True,
                return_cache=True,
            )
        decode_model_calls += 1
        next_logits = logits.detach().cpu().float()[:, -1, :vocab_size]

    decode_loop_seconds = (
        time.perf_counter() - decode_loop_started - decode_setup_seconds
    )
    token_ids = torch.cat(pieces, dim=-1)
    return GDN2GenerationResult(
        token_ids=token_ids,
        prompt_tokens=prompt_cpu.shape[1],
        generated_tokens=token_ids.shape[1] - prompt_cpu.shape[1],
        stopped_on_eos=stopped_on_eos,
        decode_backend=decode_backend,
        timings=GDN2GenerationTimings(
            prefill_seconds=prefill_seconds,
            decode_setup_seconds=decode_setup_seconds,
            decode_loop_seconds=decode_loop_seconds,
            decode_model_calls=decode_model_calls,
        ),
    )
