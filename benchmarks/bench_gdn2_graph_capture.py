#!/usr/bin/env python3
"""Measure eager and NPU-graph GDN-2 decode with the real checkpoint.

The graph experiment has two deliberately separate modes:

``fixed-cache``
    Replays one decode from the same prompt cache.  This matches the existing
    steady-decode microbenchmark and isolates launch/host overhead, but it is not
    an autoregressive cache chain.

``stateful``
    Copies each layer's newly produced recurrent and convolution cache back into
    fixed-address input buffers at the end of the captured graph.  Replays therefore
    advance the state and are usable for autoregressive decode.  ``offset`` is Python
    metadata and is intentionally maintained by the caller rather than captured.

NPUGraph capture records work without producing valid outputs.  This benchmark never
reads capture-time buffers; it performs a replay first, then applies the same-dtype
relative-L2 checks used by the rest of the GDN-2 validation suite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.models import GDN2Config, GDN2ForCausalLM, GDN2LayerCache


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--mode", choices=("fixed-cache", "stateful"), required=True)
    parser.add_argument(
        "--short-conv-path",
        choices=("cce", "torch"),
        default="cce",
        help="benchmark-only A/B switch; production packed CCE decode defaults to cce",
    )
    parser.add_argument(
        "--block-norm-path",
        choices=("native", "decomposed"),
        default="native",
        help="benchmark-only A/B switch for block/final RMSNorm",
    )
    parser.add_argument(
        "--mlp-w12-path",
        choices=("packed", "separate"),
        default="packed",
        help="benchmark-only A/B switch using the same host-packed W1/W2 parameter",
    )
    parser.add_argument(
        "--mlp-boundary-path",
        choices=("vendor", "cce"),
        default="vendor",
        help="use the explicit B=T=1 mixed RMSNorm2/W12/SwiGLU CCE candidate",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--serial-iterations",
        type=int,
        default=50,
        help="iterations with a device synchronize after every token",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        nargs="+",
        default=[1, 450, 7483, 310, 3444, 338],
    )
    parser.add_argument(
        "--correctness-tokens",
        type=int,
        nargs="+",
        default=[3681, 42, 900, 1234],
        help="token sequence used to verify static-input updates and cache chaining",
    )
    parser.add_argument("--timing-token", type=int, default=3681)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device.split(":", 1)[0] != "npu":
        parser.error("NPU graph capture requires an NPU device")
    if args.warmup < 0 or args.iterations <= 0 or args.serial_iterations <= 0:
        parser.error(
            "--warmup must be nonnegative and both iteration counts must be positive"
        )
    if not 1 <= len(args.prompt_tokens) <= 16:
        parser.error("the CCE prompt path currently requires 1..16 tokens")
    if not args.correctness_tokens:
        parser.error("--correctness-tokens must not be empty")
    if args.mlp_boundary_path == "cce" and (
        args.block_norm_path != "native" or args.mlp_w12_path != "packed"
    ):
        parser.error(
            "--mlp-boundary-path=cce requires --block-norm-path=native "
            "and --mlp-w12-path=packed for an unambiguous control"
        )
    return args


def _sync(device: torch.device) -> None:
    torch.npu.synchronize(device)


def _select_short_conv_path(model: GDN2ForCausalLM, path: str) -> None:
    """Force the pre-fusion path only for a controlled benchmark process."""
    if path == "cce":
        return

    def disabled(_self: Any, _hidden: Any, _cache: Any, _use_cache: bool) -> bool:
        return False

    for block in model.transformer["h"]:
        block.attn._uses_fused_short_conv = MethodType(  # type: ignore[method-assign]
            disabled, block.attn
        )


def _select_norm_mlp_path(
    model: GDN2ForCausalLM,
    norm_path: str,
    w12_path: str,
) -> None:
    """Select the two independent pre-optimization controls for a 2x2 A/B."""
    if norm_path == "decomposed":
        norms = [model.transformer["ln_f"]]
        for block in model.transformer["h"]:
            norms.extend((block.norm_1, block.norm_2))
        for norm in norms:
            norm._use_native_npu_inference = False
            # The previous packed implementation widened these already-quantized
            # values once.  Keeping FP32 storage makes ``weight.float()`` a no-op,
            # exactly matching that control rather than adding a benchmark Cast.
            norm.weight.data = norm.weight.data.float()

    if w12_path == "packed":
        return

    def separate_w12(self: Any, x: torch.Tensor) -> torch.Tensor:
        left = F.linear(x, self.w12_weight[: self.intermediate_size])
        right = F.linear(x, self.w12_weight[self.intermediate_size :])
        return self.w3(F.silu(left) * right)

    for block in model.transformer["h"]:
        swiglu = block.mlp.swiglu
        swiglu.forward = MethodType(separate_w12, swiglu)  # type: ignore[method-assign]


def _clone_cache(cache: list[GDN2LayerCache | None]) -> list[GDN2LayerCache]:
    copied: list[GDN2LayerCache] = []
    for layer in cache:
        if layer is None or not isinstance(layer.conv_state, torch.Tensor):
            raise TypeError("packed CCE graph benchmark requires populated tensor caches")
        copied.append(
            GDN2LayerCache(
                recurrent_state=layer.recurrent_state.clone(),
                conv_state=layer.conv_state.clone(),
                offset=layer.offset,
            )
        )
    return copied


def _copy_cache_(
    destination: list[GDN2LayerCache], source: list[GDN2LayerCache | None]
) -> None:
    if len(destination) != len(source):
        raise ValueError(f"cache layer count mismatch: {len(destination)} != {len(source)}")
    for destination_layer, source_layer in zip(destination, source):
        if source_layer is None or not isinstance(source_layer.conv_state, torch.Tensor):
            raise TypeError("packed CCE graph benchmark requires populated tensor caches")
        assert isinstance(destination_layer.conv_state, torch.Tensor)
        destination_layer.recurrent_state.copy_(source_layer.recurrent_state)
        destination_layer.conv_state.copy_(source_layer.conv_state)


def _cache_cpu(cache: list[GDN2LayerCache | None]) -> list[dict[str, torch.Tensor]]:
    copied = []
    for layer in cache:
        if layer is None or not isinstance(layer.conv_state, torch.Tensor):
            raise TypeError("packed CCE graph benchmark requires populated tensor caches")
        copied.append(
            {
                "recurrent_state": layer.recurrent_state.cpu(),
                "conv_state": layer.conv_state.cpu(),
            }
        )
    return copied


def _relative_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    delta = got.float() - want.float()
    return float(delta.norm() / want.float().norm().clamp_min(1e-30))


def _compare(
    got_logits: torch.Tensor,
    got_cache: list[GDN2LayerCache | None],
    want_logits: torch.Tensor,
    want_cache: list[GDN2LayerCache | None],
) -> dict[str, float | bool]:
    got_logits_cpu = got_logits.cpu()
    want_logits_cpu = want_logits.cpu()
    got_cache_cpu = _cache_cpu(got_cache)
    want_cache_cpu = _cache_cpu(want_cache)
    recurrent = [
        _relative_l2(got["recurrent_state"], want["recurrent_state"])
        for got, want in zip(got_cache_cpu, want_cache_cpu)
    ]
    convolution = [
        _relative_l2(got["conv_state"], want["conv_state"])
        for got, want in zip(got_cache_cpu, want_cache_cpu)
    ]
    delta = got_logits_cpu.float() - want_logits_cpu.float()
    return {
        "logits_bitwise_equal": bool(torch.equal(got_logits_cpu, want_logits_cpu)),
        "logits_max_abs_diff": float(delta.abs().max()),
        "logits_relative_l2": _relative_l2(got_logits_cpu, want_logits_cpu),
        "recurrent_relative_l2_max": max(recurrent),
        "conv_relative_l2_max": max(convolution),
    }


def _run_eager_chain(
    model: GDN2ForCausalLM,
    tokens: list[torch.Tensor],
    cache: list[GDN2LayerCache],
) -> tuple[torch.Tensor, list[GDN2LayerCache | None]]:
    current: list[GDN2LayerCache | None] = cache
    logits: torch.Tensor | None = None
    for token in tokens:
        logits, current = model(token, cache=current, return_cache=True)
    assert logits is not None
    return logits, current


def _time_eager(
    model: GDN2ForCausalLM,
    step: torch.Tensor,
    initial_cache: list[GDN2LayerCache | None],
    mode: str,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    current: list[GDN2LayerCache | None] = _clone_cache(initial_cache)
    last: Any = None
    for _ in range(warmup):
        last = model(step, cache=current, return_cache=True)
        if mode == "stateful":
            current = last[1]
    _sync(device)
    started = time.perf_counter()
    for _ in range(iterations):
        last = model(step, cache=current, return_cache=True)
        if mode == "stateful":
            current = last[1]
    enqueued = time.perf_counter()
    _sync(device)
    finished = time.perf_counter()
    del last
    return {
        "enqueue_us_per_token": (enqueued - started) * 1e6 / iterations,
        "total_us_per_token": (finished - started) * 1e6 / iterations,
        "final_drain_us": (finished - enqueued) * 1e6,
    }


def _time_graph(
    graph: Any,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    started = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
    enqueued = time.perf_counter()
    _sync(device)
    finished = time.perf_counter()
    return {
        "enqueue_us_per_token": (enqueued - started) * 1e6 / iterations,
        "total_us_per_token": (finished - started) * 1e6 / iterations,
        "final_drain_us": (finished - enqueued) * 1e6,
    }


def _time_eager_serial(
    model: GDN2ForCausalLM,
    step: torch.Tensor,
    initial_cache: list[GDN2LayerCache | None],
    mode: str,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    current: list[GDN2LayerCache | None] = _clone_cache(initial_cache)
    last: Any = None
    for _ in range(min(warmup, 5)):
        last = model(step, cache=current, return_cache=True)
        if mode == "stateful":
            current = last[1]
        _sync(device)
    started = time.perf_counter()
    for _ in range(iterations):
        last = model(step, cache=current, return_cache=True)
        if mode == "stateful":
            current = last[1]
        _sync(device)
    finished = time.perf_counter()
    del last
    return (finished - started) * 1e6 / iterations


def _time_graph_serial(
    graph: Any,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(min(warmup, 5)):
        graph.replay()
        _sync(device)
    started = time.perf_counter()
    for _ in range(iterations):
        graph.replay()
        _sync(device)
    finished = time.perf_counter()
    return (finished - started) * 1e6 / iterations


def main() -> None:
    args = _parse_args()
    import torch_npu  # noqa: F401

    device = torch.device(args.device)
    # Claim the shared card immediately. Allocation-only torch.empty does not
    # resolve an operator before custom OPP registration.
    reservation = torch.empty((1,), dtype=torch.uint8, device=device)
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint,
        config=GDN2Config.gdn2_1_3b(),
        device=device,
        dtype=torch.bfloat16,
        strict=True,
        core_backend="cce",
        projection_layout="packed-inference",
        mlp_backend="cce" if args.mlp_boundary_path == "cce" else "torch",
    )
    _select_short_conv_path(model, args.short_conv_path)
    _select_norm_mlp_path(model, args.block_norm_path, args.mlp_w12_path)
    prompt = torch.tensor([args.prompt_tokens], dtype=torch.long).to(device)
    timing_step = torch.tensor([[args.timing_token]], dtype=torch.long).to(device)
    correctness_steps = [
        torch.tensor([[token]], dtype=torch.long).to(device)
        for token in args.correctness_tokens
    ]

    with torch.inference_mode():
        _, prompt_cache = model(prompt, return_cache=True)
        eager_before_us = _time_eager(
            model,
            timing_step,
            prompt_cache,
            args.mode,
            args.warmup,
            args.iterations,
            device,
        )
        # Remove allocator history from the eager measurement before attributing
        # incremental memory to the graph's static inputs, outputs and private pool.
        _sync(device)
        torch.npu.empty_cache()
        memory_before_capture = {
            "allocated_bytes": torch.npu.memory_allocated(device),
            "reserved_bytes": torch.npu.memory_reserved(device),
        }

        graph_input_cache = _clone_cache(prompt_cache)
        graph = torch.npu.NPUGraph()
        capture_started = time.perf_counter()
        with torch.npu.graph(graph):
            graph_logits, graph_output_cache = model(
                timing_step, cache=graph_input_cache, return_cache=True
            )
            if args.mode == "stateful":
                _copy_cache_(graph_input_cache, graph_output_cache)
        capture_seconds = time.perf_counter() - capture_started
        memory_after_capture = {
            "allocated_bytes": torch.npu.memory_allocated(device),
            "reserved_bytes": torch.npu.memory_reserved(device),
        }

        # Capture does not execute the graph.  Restore the known prompt state anyway
        # so correctness does not depend on that implementation detail.
        _copy_cache_(graph_input_cache, prompt_cache)
        _sync(device)

        eager_check_cache = _clone_cache(prompt_cache)
        eager_logits, eager_check_cache_out = _run_eager_chain(
            model, correctness_steps, eager_check_cache
        )
        if args.mode == "fixed-cache":
            # Each fixed-cache replay represents one independent step from the same
            # prompt state, so only the last token is the corresponding eager oracle.
            eager_logits, eager_check_cache_out = model(
                correctness_steps[-1], cache=eager_check_cache, return_cache=True
            )

        for token in correctness_steps:
            timing_step.copy_(token)
            _sync(device)
            graph.replay()
        _sync(device)
        graph_check_cache: list[GDN2LayerCache | None]
        if args.mode == "stateful":
            graph_check_cache = graph_input_cache
        else:
            graph_check_cache = graph_output_cache
        comparison = _compare(
            graph_logits,
            graph_check_cache,
            eager_logits,
            eager_check_cache_out,
        )

        timing_step.fill_(args.timing_token)
        _sync(device)
        # The graph cache may continue advancing between measurements; shapes and
        # addresses stay fixed, which is exactly the steady autoregressive condition.
        for _ in range(args.warmup):
            graph.replay()
        _sync(device)
        graph_first_us = _time_graph(graph, args.iterations, device)
        graph_second_us = _time_graph(graph, args.iterations, device)
        graph_serial_us = _time_graph_serial(
            graph, args.warmup, args.serial_iterations, device
        )
        eager_after_us = _time_eager(
            model,
            timing_step,
            prompt_cache,
            args.mode,
            args.warmup,
            args.iterations,
            device,
        )
        eager_serial_us = _time_eager_serial(
            model,
            timing_step,
            prompt_cache,
            args.mode,
            args.warmup,
            args.serial_iterations,
            device,
        )

    budget = 1e-2
    passed = all(
        (
            comparison["logits_relative_l2"] <= budget,
            comparison["recurrent_relative_l2_max"] <= budget,
            comparison["conv_relative_l2_max"] <= budget,
        )
    )
    eager_mean_us = (
        eager_before_us["total_us_per_token"]
        + eager_after_us["total_us_per_token"]
    ) / 2.0
    graph_mean_us = (
        graph_first_us["total_us_per_token"]
        + graph_second_us["total_us_per_token"]
    ) / 2.0
    result = {
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "dtype": "bfloat16",
        "core_backend": "cce",
        "projection_layout": "packed-inference",
        "short_conv_path": args.short_conv_path,
        "block_norm_path": args.block_norm_path,
        "mlp_w12_path": args.mlp_w12_path,
        "mlp_boundary_path": args.mlp_boundary_path,
        "mode": args.mode,
        "capture_seconds": capture_seconds,
        "memory": {
            "before_capture": memory_before_capture,
            "after_capture": memory_after_capture,
            "allocated_delta_bytes": (
                memory_after_capture["allocated_bytes"]
                - memory_before_capture["allocated_bytes"]
            ),
            "reserved_delta_bytes": (
                memory_after_capture["reserved_bytes"]
                - memory_before_capture["reserved_bytes"]
            ),
            "stateful_copyback_bytes_per_token": (
                sum(
                    layer.recurrent_state.numel()
                    * layer.recurrent_state.element_size()
                    + layer.conv_state.numel() * layer.conv_state.element_size()
                    for layer in graph_input_cache
                    if isinstance(layer.conv_state, torch.Tensor)
                )
                if args.mode == "stateful"
                else 0
            ),
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "serial_iterations": args.serial_iterations,
        "correctness_tokens": args.correctness_tokens,
        "eager_us": {
            "before": eager_before_us,
            "after": eager_after_us,
            "total_mean": eager_mean_us,
        },
        "graph_replay_us": {
            "first": graph_first_us,
            "second": graph_second_us,
            "total_mean": graph_mean_us,
        },
        "eager_tokens_per_second": 1e6 / eager_mean_us,
        "graph_tokens_per_second": 1e6 / graph_mean_us,
        "speedup": eager_mean_us / graph_mean_us,
        "serial_sync_each_token": {
            "eager_us_per_token": eager_serial_us,
            "graph_us_per_token": graph_serial_us,
            "eager_tokens_per_second": 1e6 / eager_serial_us,
            "graph_tokens_per_second": 1e6 / graph_serial_us,
            "speedup": eager_serial_us / graph_serial_us,
        },
        "comparison": comparison,
        "acceptance": {
            "relative_l2_budget": budget,
            "rule": "same-dtype relative-L2 for logits and both cache kinds",
        },
        "offset_policy": "Python caller increments offset; offset is not captured",
        "passed": passed,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("NPU graph replay exceeded the BF16 numerical budget")
    assert reservation.numel() == 1


if __name__ == "__main__":
    main()
