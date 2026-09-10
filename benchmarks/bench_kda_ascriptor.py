#!/usr/bin/env python3
"""KDA 自编译算子 vs torch_npu 基线 —— **同机同卡同输入**对比。

为什么必须同机同卡：两边的数只有在同一张卡、同一次 warmup 策略下才可比。
跨机对比（编译在 A 机、基线在 B 机）得出的加速比没有意义 —— 见 AGENTS.md §6。
两个实现在同一进程里比；但不同 block_dim 必须分进程，原因见下。

本脚本扫 ``SUPPORTED_BLOCK_DIM``（契约声明的全范围 1..4）。这一维不是调参而是
**并行度**：kernel 内部用 ``GetVecIdx()/GetVecNum()`` 自行切分，而
``GetVecNum() == 2 * block_dim``，所以 ``block_dim=1`` 只用到 2 个向量核。

**每个 block_dim 必须单独起进程。** 同一算子名的多份 build 在一个进程里会冲突：
CANN 按算子名在 ``ASCEND_CUSTOM_OPP_PATH`` 里查，第一个命中的胜出，后面的被静默
忽略。实测同进程扫 1..4 时四次都执行了 bd=1 的二进制，耗时完全相同；分进程后
bd=4 比 bd=1 快 3.9 倍。本脚本收到多个 ``--block-dim`` 时自动派子进程，
:func:`ascend_fla.runtime.binding._claim_op_name` 则在越界时直接报错。

用法（在有 NPU 且 opp 含 ascend950 的机器上，先 source 环境脚本）::

    python bench_kda_ascriptor.py --shape kimi_linear_layer
    python bench_kda_ascriptor.py --shape all --accuracy
    python bench_kda_ascriptor.py --shape smoke --block-dim 1 --no-baseline

计时前提：``layout_device="npu"``。若本机缺内置 copy 算子（见 AGENTS.md 的
"开机必查"），重排会绕主机往返，计时结果不能用于性能结论 —— 此时脚本直接报错，
不静默出一个假的数。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ascend_fla.ops.kda.chunk import SUPPORTED_BLOCK_DIM, chunk_kda_fwd  # noqa: E402
from ascend_fla.reference.kda import kda_chunk_vectorized  # noqa: E402
from benchmarks.bench_kda_torch_npu import SHAPES, make_inputs, rel_l2, sync  # noqa: E402


def timed(fn, kwargs, warmup, iters):
    """同步计时，返回每次迭代的毫秒数。warmup 同时吃掉首次编译开销。"""
    for _ in range(warmup):
        fn(**kwargs)
    sync("npu")
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(**kwargs)
    sync("npu")
    return (time.perf_counter() - t0) / iters * 1e3


def _sweep_in_subprocesses(args) -> int:
    """每个 block_dim 起一个子进程，父进程汇总。

    不是为了并行，而是为了**正确**：同进程多份同名 build 会互相覆盖（见模块 docstring）。
    """
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for bd in args.block_dim:
            out = pathlib.Path(td) / f"bd{bd}.json"
            cmd = [sys.executable, str(pathlib.Path(__file__).resolve()),
                   "--shape", args.shape, "--block-dim", str(bd),
                   "--warmup", str(args.warmup), "--iters", str(args.iters),
                   "--json-out", str(out)]
            if args.accuracy:
                cmd.append("--accuracy")
            if args.no_baseline:
                cmd.append("--no-baseline")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not out.is_file():
                print(f"block_dim={bd} 的子进程失败（exit={r.returncode}）：", file=sys.stderr)
                print(r.stdout[-2000:], file=sys.stderr)
                print(r.stderr[-2000:], file=sys.stderr)
                return 1
            rows += json.loads(out.read_text(encoding="utf-8"))

    print(f"dtype=bfloat16 K=V=128 chunk=64 warmup={args.warmup} iters={args.iters} "
          f"synchronized=yes fwd_only=yes layout_device=npu 每个 block_dim 一个子进程")
    print(f"{'shape':<20} {'B/H/HV/T':<16} {'bd':>3} {'ascriptor(ms)':>14} "
          f"{'torch_npu(ms)':>14} {'speedup':>9}")
    print("-" * 82)
    for r in rows:
        d = r["dims"]
        dims = f"{d['B']}/{d['H']}/{d['HV']}/{d['T']}"
        base = r.get("ms_torch_npu")
        speed = f"{base / r['ms_ascriptor']:.2f}x" if base else "—"
        print(f"{r['shape']:<20} {dims:<16} {r['block_dim']:>3} {r['ms_ascriptor']:>14.3f} "
              f"{(base if base else float('nan')):>14.3f} {speed:>9}")
        if "rel_l2_o_vs_cpu" in r:
            print(f"{'':<20} {'':<16} {'':>3} 精度: o relL2={r['rel_l2_o_vs_cpu']:.3e} "
                  f"state={r['rel_l2_state_vs_cpu']:.3e}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="all", choices=[*SHAPES, "all"])
    ap.add_argument("--block-dim", type=int, nargs="*", default=list(SUPPORTED_BLOCK_DIM),
                    help=f"要扫的 block_dim，默认全范围 {list(SUPPORTED_BLOCK_DIM)}")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--accuracy", action="store_true",
                    help="对 CPU fp32 参考比对。每个 block_dim 都比 —— 分区方式变了结果不应变")
    ap.add_argument("--no-baseline", action="store_true", help="跳过 torch_npu 基线（只看自编译侧）")
    ap.add_argument("--json-out", type=pathlib.Path)
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("本脚本只在 NPU 上有意义：torch_npu 不可用", file=sys.stderr)
        return 2
    if not torch.npu.is_available():
        print("torch.npu.is_available() 为 False", file=sys.stderr)
        return 2

    bad = [b for b in args.block_dim if b not in SUPPORTED_BLOCK_DIM]
    if bad:
        print(f"block_dim {bad} 超出契约声明的 {list(SUPPORTED_BLOCK_DIM)}", file=sys.stderr)
        return 2

    if len(args.block_dim) > 1:
        return _sweep_in_subprocesses(args)

    names = list(SHAPES) if args.shape == "all" else [args.shape]
    print(f"dtype=bfloat16 K=V=128 chunk=64 warmup={args.warmup} iters={args.iters} "
          f"synchronized=yes fwd_only=yes layout_device=npu")
    print(f"{'shape':<20} {'B/H/HV/T':<16} {'bd':>3} {'ascriptor(ms)':>14} "
          f"{'torch_npu(ms)':>14} {'speedup':>9}")
    print("-" * 82)

    records = []
    for name in names:
        shp = SHAPES[name]
        x = make_inputs(**shp, dtype=torch.bfloat16, device="npu")
        dims = f"{shp['B']}/{shp['H']}/{shp['HV']}/{shp['T']}"

        ms_base = float("nan")
        if not args.no_baseline:
            ms_base = timed(kda_chunk_vectorized, dict(**x, output_final_state=True),
                            args.warmup, args.iters)

        ref = None
        if args.accuracy:
            x_cpu = {k: v.cpu() for k, v in x.items()}
            ref = kda_chunk_vectorized(**x_cpu, output_final_state=True)

        for bd in args.block_dim:
            kw = dict(**x, output_final_state=True, block_dim=bd, layout_device="npu")
            ms = timed(chunk_kda_fwd, kw, args.warmup, args.iters)
            speed = f"{ms_base / ms:.2f}x" if ms_base == ms_base else "—"
            print(f"{name:<20} {dims:<16} {bd:>3} {ms:>14.3f} {ms_base:>14.3f} {speed:>9}")

            rec = dict(shape=name, dims=shp, block_dim=bd, dtype="bfloat16",
                       ms_ascriptor=ms, ms_torch_npu=(ms_base if ms_base == ms_base else None),
                       warmup=args.warmup, iters=args.iters, synchronized=True,
                       forward_only=True, layout_device="npu")
            if ref is not None:
                o, s = chunk_kda_fwd(**kw)
                rec["rel_l2_o_vs_cpu"] = rel_l2(o.cpu(), ref[0])
                rec["rel_l2_state_vs_cpu"] = rel_l2(s.cpu(), ref[1])
                print(f"{'':<20} {'':<16} {'':>3} 精度: o relL2={rec['rel_l2_o_vs_cpu']:.3e} "
                      f"state={rec['rel_l2_state_vs_cpu']:.3e}")
            records.append(rec)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
