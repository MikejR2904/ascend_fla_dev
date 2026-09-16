#!/usr/bin/env python3
"""Profile one GDN-2 backend with real weights and summarize its decode path.

The model is warmed before profiling. Temporary ``record_function`` wrappers mark
the exact production forward path without adding ranges to normal inference. The
torch_npu profiler emits its full trace and analysed CSV files; this script also
aggregates device kernels, Torch operators and layer phases into a compact JSON
report. Run different backends in separate processes so CANN operator discovery and
profiler state cannot leak between comparisons.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
import statistics
import sys
import time
from types import MethodType
from typing import Any, Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ascend_fla.models import GDN2Config, GDN2ForCausalLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--core-backend", choices=("torch", "cce"), default="cce")
    parser.add_argument(
        "--short-conv-path",
        choices=("cce", "torch"),
        default="cce",
        help="benchmark-only A/B switch for packed BF16 CCE decode",
    )
    parser.add_argument(
        "--mlp-boundary-path",
        choices=("vendor", "cce"),
        default="vendor",
        help="profile the explicit conservative RMSNorm2/W12/SwiGLU CCE path",
    )
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
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-warmup-steps", type=int, default=1)
    parser.add_argument("--active-steps", type=int, default=3)
    args = parser.parse_args()
    if args.device.split(":", 1)[0] != "npu":
        parser.error("profiling requires an NPU device")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be nonnegative and --iterations must be positive")
    if args.profile_warmup_steps < 1 or args.active_steps < 1:
        parser.error("profile warmup and active steps must both be positive")
    if args.core_backend == "cce" and not 1 <= len(args.prompt_tokens) <= 16:
        parser.error("the first CCE recurrent scope requires 1..16 prompt tokens")
    if args.short_conv_path == "torch" and (
        args.core_backend != "cce" or args.projection_layout != "packed-inference"
    ):
        parser.error("--short-conv-path=torch requires packed-inference CCE model")
    if args.mlp_boundary_path == "cce" and (
        args.core_backend != "cce" or args.projection_layout != "packed-inference"
    ):
        parser.error("--mlp-boundary-path=cce requires packed-inference CCE model")
    if args.trace_dir.exists() and any(args.trace_dir.iterdir()):
        parser.error(f"--trace-dir must be empty or absent: {args.trace_dir}")
    return args


def _sync(device: torch.device) -> None:
    torch.npu.synchronize(device)


def _select_short_conv_path(model: GDN2ForCausalLM, path: str) -> None:
    if path == "cce":
        return
    if model.core_backend != "cce" or model.projection_layout != "packed-inference":
        raise ValueError("--short-conv-path=torch requires packed-inference CCE model")

    def disabled(_self: Any, _hidden: Any, _cache: Any, _use_cache: bool) -> bool:
        return False

    for block in model.transformer["h"]:
        block.attn._uses_fused_short_conv = MethodType(  # type: ignore[method-assign]
            disabled, block.attn
        )


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


def _wrap_range(
    obj: Any,
    attribute: str,
    label: str,
    originals: list[tuple[Any, str, Callable[..., Any]]],
) -> None:
    original = getattr(obj, attribute)

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)

    originals.append((obj, attribute, original))
    setattr(obj, attribute, wrapped)


def _install_ranges(model: GDN2ForCausalLM) -> list[tuple[Any, str, Callable[..., Any]]]:
    originals: list[tuple[Any, str, Callable[..., Any]]] = []
    _wrap_range(model.transformer["wte"], "forward", "gdn2.model.embedding", originals)
    _wrap_range(model.transformer["ln_f"], "forward", "gdn2.model.final_norm", originals)
    _wrap_range(model.lm_head, "forward", "gdn2.model.lm_head", originals)
    for index, block in enumerate(model.transformer["h"]):
        prefix = f"gdn2.layer.{index:02d}"
        mixer = block.attn
        _wrap_range(block, "forward", f"{prefix}.block", originals)
        _wrap_range(block.norm_1, "forward", f"{prefix}.norm1", originals)
        _wrap_range(mixer, "forward", f"{prefix}.mixer", originals)
        if model.projection_layout == "packed-inference":
            _wrap_range(mixer, "_project_packed", f"{prefix}.project", originals)
            _wrap_range(
                mixer.packed_projections,
                "forward",
                f"{prefix}.project.linear",
                originals,
            )
            if mixer.use_short_conv:
                _wrap_range(
                    mixer.packed_qkv_conv,
                    "forward",
                    f"{prefix}.project.conv",
                    originals,
                )
                _wrap_range(
                    mixer,
                    "_run_fused_short_conv",
                    f"{prefix}.project.conv",
                    originals,
                )
        else:
            _wrap_range(mixer, "_project_canonical", f"{prefix}.project", originals)
        _wrap_range(mixer, "_run_core", f"{prefix}.core", originals)
        _wrap_range(
            mixer,
            "_run_fused_decode",
            f"{prefix}.fused_decode",
            originals,
        )
        _wrap_range(mixer.o_norm, "forward", f"{prefix}.output_norm_gate", originals)
        _wrap_range(mixer.o_proj, "forward", f"{prefix}.output_projection", originals)
        _wrap_range(block.norm_2, "forward", f"{prefix}.norm2", originals)
        if block.mlp.swiglu.mlp_backend == "cce":
            _wrap_range(
                block.mlp,
                "forward_with_norm",
                f"{prefix}.mlp",
                originals,
            )
        else:
            _wrap_range(block.mlp, "forward", f"{prefix}.mlp", originals)
    return originals


def _restore_ranges(originals: list[tuple[Any, str, Callable[..., Any]]]) -> None:
    for obj, attribute, original in reversed(originals):
        setattr(obj, attribute, original)


def _read_csvs(root: Path, filename: str) -> tuple[list[Path], list[dict[str, str]]]:
    paths = sorted(root.rglob(filename))
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows.extend(csv.DictReader(stream))
    return paths, rows


def _number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "").strip()
    try:
        return float(value)
    except ValueError:
        return 0.0


def _top_groups(
    rows: list[dict[str, str]],
    *,
    name_key: str,
    duration_key: str,
    steps: int,
    limit: int = 30,
) -> list[dict[str, float | int | str]]:
    groups: dict[str, tuple[int, float]] = {}
    for row in rows:
        name = row.get(name_key, "") or "<empty>"
        count, duration = groups.get(name, (0, 0.0))
        groups[name] = (count + 1, duration + _number(row, duration_key))
    ranked = sorted(groups.items(), key=lambda item: item[1][1], reverse=True)
    return [
        {
            "name": name,
            "calls": count,
            "total_duration_us": duration,
            "duration_us_per_step": duration / steps,
        }
        for name, (count, duration) in ranked[:limit]
    ]


def _cast_signatures(
    rows: list[dict[str, str]], steps: int
) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str, str], tuple[int, float]] = {}
    for row in rows:
        if row.get("Type") != "Cast":
            continue
        key = (
            row.get("Input Shapes", ""),
            row.get("Input Data Types", ""),
            row.get("Output Shapes", ""),
            row.get("Output Data Types", ""),
        )
        count, duration = groups.get(key, (0, 0.0))
        groups[key] = (count + 1, duration + _number(row, "Duration(us)"))
    ranked = sorted(groups.items(), key=lambda item: item[1][1], reverse=True)
    return [
        {
            "input_shape": key[0],
            "input_dtype": key[1],
            "output_shape": key[2],
            "output_dtype": key[3],
            "calls": count,
            "calls_per_step": count / steps,
            "total_duration_us": duration,
            "duration_us_per_step": duration / steps,
        }
        for key, (count, duration) in ranked
    ]


def _custom_kernel_metrics(
    rows: list[dict[str, str]], steps: int
) -> dict[str, dict[str, Any]]:
    """Keep same-row latency/pipe evidence for repository-owned GDN-2 kernels."""
    ratio_columns = (
        "aic_mac_ratio",
        "aic_scalar_ratio",
        "aic_mte1_ratio",
        "aic_mte2_ratio",
        "aic_mte3_ratio",
        "aic_fixpipe_ratio",
        "aic_icache_miss_rate",
        "aiv_vec_ratio",
        "aiv_scalar_ratio",
        "aiv_mte2_ratio",
        "aiv_mte3_ratio",
        "aiv_icache_miss_rate",
    )
    absolute_columns = {
        "Wait Time(us)": "wait_time_us",
        "aicore_time(us)": "aicore_time_us",
        "cube_utilization(%)": "cube_utilization_percent",
    }
    by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = row.get("Name", "")
        if name.startswith("Gdn2") and name.endswith("Kernel"):
            by_name.setdefault(name, []).append(row)
    result: dict[str, dict[str, Any]] = {}
    for name, samples in by_name.items():
        durations = [_number(row, "Duration(us)") for row in samples]
        item: dict[str, Any] = {
            "calls": len(samples),
            "calls_per_step": len(samples) / steps,
            "device_ids": sorted({row.get("Device_id", "") for row in samples}),
            "duration_us": {
                "mean": statistics.fmean(durations),
                "median": statistics.median(durations),
                "min": min(durations),
                "max": max(durations),
            },
        }
        for column in ratio_columns:
            values = [_number(row, column) * 100.0 for row in samples]
            item[f"{column}_percent"] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        for column, output_name in absolute_columns.items():
            values = [_number(row, column) for row in samples]
            item[output_name] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
        result[name] = item
    return result


def _range_metrics(
    rows: list[dict[str, str]], steps: int
) -> tuple[dict[str, dict[str, float | int]], dict[str, float]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for row in rows:
        name = row.get("Name", "")
        if not name.startswith("gdn2."):
            continue
        item = metrics.setdefault(
            name,
            {
                "calls": 0,
                "host_self_us": 0.0,
                "host_total_us": 0.0,
                "device_self_us": 0.0,
                "device_total_us": 0.0,
            },
        )
        item["calls"] += 1
        item["host_self_us"] += _number(row, "Host Self Duration(us)")
        item["host_total_us"] += _number(row, "Host Total Duration(us)")
        item["device_self_us"] += _number(row, "Device Self Duration(us)")
        item["device_total_us"] += _number(row, "Device Total Duration(us)")

    for item in metrics.values():
        for key in tuple(item):
            if key != "calls":
                item[f"{key}_per_step"] = float(item[key]) / steps

    def exact_suffix(suffix: str, field: str = "device_total_us") -> float:
        return sum(
            float(item[field])
            for name, item in metrics.items()
            if name.endswith(suffix)
        ) / steps

    mixer = exact_suffix(".mixer")
    project = exact_suffix(".project")
    core = exact_suffix(".core")
    output_norm_gate = exact_suffix(".output_norm_gate")
    output_projection = exact_suffix(".output_projection")
    fused_decode = exact_suffix(".fused_decode")
    block = exact_suffix(".block")
    norm1 = exact_suffix(".norm1")
    norm2 = exact_suffix(".norm2")
    mlp = exact_suffix(".mlp")
    embedding = float(metrics.get("gdn2.model.embedding", {}).get("device_total_us", 0.0)) / steps
    final_norm = float(metrics.get("gdn2.model.final_norm", {}).get("device_total_us", 0.0)) / steps
    lm_head = float(metrics.get("gdn2.model.lm_head", {}).get("device_total_us", 0.0)) / steps
    decode = float(metrics.get("gdn2.decode_step", {}).get("device_total_us", 0.0)) / steps
    derived = {
        "decode_device_us_per_step": decode,
        "embedding_device_us_per_step": embedding,
        "blocks_device_us_per_step": block,
        "norm1_device_us_per_step": norm1,
        "mixer_device_us_per_step": mixer,
        "project_device_us_per_step": project,
        "project_linear_device_us_per_step": exact_suffix(".project.linear"),
        "project_conv_device_us_per_step": exact_suffix(".project.conv"),
        "gate_prep_residual_device_us_per_step": mixer
        - project
        - core
        - output_norm_gate
        - output_projection,
        "core_device_us_per_step": core,
        "fused_decode_torch_children_device_us_per_step": fused_decode,
        "output_norm_gate_device_us_per_step": output_norm_gate,
        "output_projection_device_us_per_step": output_projection,
        "norm2_device_us_per_step": norm2,
        "mlp_device_us_per_step": mlp,
        "block_residual_device_us_per_step": block - norm1 - mixer - norm2 - mlp,
        "final_norm_device_us_per_step": final_norm,
        "lm_head_device_us_per_step": lm_head,
        "model_unscoped_device_us_per_step": decode
        - block
        - embedding
        - final_norm
        - lm_head,
    }
    return metrics, derived


def main() -> None:
    args = _parse_args()
    import torch_npu  # noqa: F401
    import torch_npu.profiler as npu_profiler

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    reservation = torch.empty((1,), dtype=torch.uint8, device=device)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    model = GDN2ForCausalLM.from_checkpoint(
        args.checkpoint,
        config=GDN2Config.gdn2_1_3b(),
        device=device,
        dtype=dtype,
        strict=True,
        core_backend=args.core_backend,
        projection_layout=args.projection_layout,
        mlp_backend="cce" if args.mlp_boundary_path == "cce" else "torch",
    )
    _select_short_conv_path(model, args.short_conv_path)
    prompt = torch.tensor([args.prompt_tokens], dtype=torch.long).to(device)
    step = torch.tensor([[args.step_token]], dtype=torch.long).to(device)
    with torch.inference_mode():
        _, prompt_cache = model(prompt, return_cache=True)
        decode_us = _timed(
            lambda: model(step, cache=prompt_cache, return_cache=True),
            device,
            args.warmup,
            args.iterations,
        )

    originals = _install_ranges(model)
    total_profile_steps = args.profile_warmup_steps + args.active_steps
    try:
        handler = npu_profiler.tensorboard_trace_handler(
            str(args.trace_dir),
            worker_name=f"gdn2_{args.core_backend}",
            analyse_flag=True,
            async_mode=False,
        )
        experimental = npu_profiler._ExperimentalConfig(
            profiler_level=npu_profiler.ProfilerLevel.Level1,
            aic_metrics=npu_profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False,
            export_type=npu_profiler.ExportType.Text,
        )
        schedule = npu_profiler.schedule(
            wait=0,
            warmup=args.profile_warmup_steps,
            active=args.active_steps,
            repeat=1,
        )
        with npu_profiler.profile(
            activities=[
                npu_profiler.ProfilerActivity.CPU,
                npu_profiler.ProfilerActivity.NPU,
            ],
            schedule=schedule,
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            experimental_config=experimental,
        ) as profiler:
            with torch.inference_mode():
                for _ in range(total_profile_steps):
                    with torch.profiler.record_function("gdn2.decode_step"):
                        last = model(step, cache=prompt_cache, return_cache=True)
                    _sync(device)
                    profiler.step()
            del last
    finally:
        _restore_ranges(originals)

    kernel_paths, kernel_rows = _read_csvs(args.trace_dir, "kernel_details.csv")
    operator_paths, operator_rows = _read_csvs(args.trace_dir, "operator_details.csv")
    if not kernel_rows or not operator_rows:
        raise RuntimeError(
            "profiler analysis did not produce populated kernel/operator CSVs: "
            f"kernels={kernel_paths}, operators={operator_paths}"
        )
    ranges, phases = _range_metrics(operator_rows, args.active_steps)
    torch_rows = [row for row in operator_rows if not row.get("Name", "").startswith("gdn2.")]
    kernel_total = sum(_number(row, "Duration(us)") for row in kernel_rows)
    result = {
        "checkpoint": args.checkpoint.name,
        "device": str(device),
        "dtype": args.dtype,
        "core_backend": args.core_backend,
        "projection_layout": args.projection_layout,
        "short_conv_path": args.short_conv_path,
        "mlp_boundary_path": args.mlp_boundary_path,
        "prompt_length": len(args.prompt_tokens),
        "steady_decode_us": decode_us,
        "profile": {
            "warmup_steps": args.profile_warmup_steps,
            "active_steps": args.active_steps,
            "profiler_level": "Level1",
            "aic_metrics": "PipeUtilization",
            "kernel_csvs": [str(path.relative_to(args.trace_dir)) for path in kernel_paths],
            "operator_csvs": [str(path.relative_to(args.trace_dir)) for path in operator_paths],
            "kernel_rows": len(kernel_rows),
            "operator_rows": len(operator_rows),
            "kernel_duration_us_per_step": kernel_total / args.active_steps,
        },
        "derived_layer_phases": phases,
        "marked_ranges": ranges,
        "top_kernel_types": _top_groups(
            kernel_rows,
            name_key="Type",
            duration_key="Duration(us)",
            steps=args.active_steps,
        ),
        "top_kernel_names": _top_groups(
            kernel_rows,
            name_key="Name",
            duration_key="Duration(us)",
            steps=args.active_steps,
        ),
        "top_torch_ops_by_device_self": _top_groups(
            torch_rows,
            name_key="Name",
            duration_key="Device Self Duration(us)",
            steps=args.active_steps,
        ),
        "cast_signatures": _cast_signatures(kernel_rows, args.active_steps),
        "custom_kernel_metrics": _custom_kernel_metrics(
            kernel_rows, args.active_steps
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    args.output.write_text(text + "\n", encoding="utf-8")
    assert reservation.numel() == 1


if __name__ == "__main__":
    main()
