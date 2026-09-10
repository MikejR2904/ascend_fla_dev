#!/usr/bin/env python3
"""拆开 chunk_kda_fwd 的 12ms —— 是 host 侧开销还是设备算力？

动机：bench_kda_ascriptor.py 实测自编译侧在 smoke（1 个工作单元）和 long_context
（512 倍数据量）上分别是 11.7ms / 20.8ms，且 ``block_dim`` 1→4 几乎无差别。这个
形状不敏感 + 并行度不敏感的组合说明耗时**不在设备上**。本脚本把它拆成四段：

1. layout 重排（token-major ↔ BHCLD 的 6 次 permute+contiguous）
2. 输出张量分配
3. 每个 kernel 的 ``aclnnXxxGetWorkspaceSize`` —— host 侧 tiling，**同步**的
4. 每个 kernel 的 ``aclnnXxx`` —— 只是往 stream 上入队，**异步**的

``_run`` 是异步的，所以它的 host 耗时只是入队成本；设备真实耗时 = 总 wall（末尾
同步）减去 host 侧各段。这个差值才是算力账。

用法::

    python profile_bridge_overhead.py --shape kimi_linear_layer --iters 10
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ascend_fla.ops.kda import chunk as kda_chunk  # noqa: E402
from ascend_fla.runtime.binding import AclnnOp  # noqa: E402
from benchmarks.bench_kda_torch_npu import SHAPES, make_inputs  # noqa: E402

TIMES: dict[str, float] = collections.defaultdict(float)
COUNTS: dict[str, int] = collections.defaultdict(int)


SYNC_EACH = False


def _sync() -> None:
    if SYNC_EACH:
        torch.npu.synchronize()


def instrument() -> None:
    """把 AclnnOp 的两段式调用各自计时。只包裹 ctypes 函数对象，不改语义。

    ``SYNC_EACH`` 为真时在每段前后同步，把**设备**耗时归因到段上。代价是破坏流水，
    所以这种模式下的总耗时必然偏大 —— 它回答的是"时间花在哪"，不是"一共多久"。
    """
    orig_init = AclnnOp.__init__

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        name, raw_ws, raw_run = self.op_name, self._get_ws, self._run

        def timed_ws(*args):
            t0 = time.perf_counter()
            rc = raw_ws(*args)
            TIMES[f"ws:{name}"] += time.perf_counter() - t0
            COUNTS[f"ws:{name}"] += 1
            return rc

        def timed_run(*args):
            _sync()
            t0 = time.perf_counter()
            rc = raw_run(*args)
            _sync()
            TIMES[f"launch:{name}"] += time.perf_counter() - t0
            COUNTS[f"launch:{name}"] += 1
            return rc

        self._get_ws, self._run = timed_ws, timed_run

    AclnnOp.__init__ = init

    for fname in ("_to_bhcld", "_from_bhcld"):
        raw = getattr(kda_chunk, fname)

        def wrap(*a, _raw=raw, _n=fname, **kw):
            _sync()
            t0 = time.perf_counter()
            out = _raw(*a, **kw)
            _sync()
            TIMES[f"layout:{_n}"] += time.perf_counter() - t0
            COUNTS[f"layout:{_n}"] += 1
            return out

        setattr(kda_chunk, fname, wrap)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="kimi_linear_layer", choices=list(SHAPES))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--sync-each", action="store_true",
                    help="每段前后同步，把设备耗时归因到段上（会破坏流水，总耗时偏大）")
    args = ap.parse_args()

    global SYNC_EACH
    SYNC_EACH = args.sync_each

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("需要 torch_npu", file=sys.stderr)
        return 2

    instrument()
    shp = SHAPES[args.shape]
    x = make_inputs(**shp, dtype=torch.bfloat16, device="npu")
    kw = dict(**x, output_final_state=True, block_dim=args.block_dim, layout_device="npu")

    for _ in range(3):  # warmup：吃掉编译与首次分配
        kda_chunk.chunk_kda_fwd(**kw)
    torch.npu.synchronize()
    TIMES.clear()
    COUNTS.clear()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        kda_chunk.chunk_kda_fwd(**kw)
    host_total = time.perf_counter() - t0      # 未同步：host 侧总耗时
    torch.npu.synchronize()
    wall_total = time.perf_counter() - t0      # 同步后：真实总耗时

    n = args.iters
    mode = "sync-each（设备耗时已归因，总耗时偏大）" if SYNC_EACH else "流水（总耗时真实，段只含 host 成本）"
    print(f"shape={args.shape} {shp} block_dim={args.block_dim} iters={n} 模式={mode}\n")
    print(f"{'段':<42} {'每次(ms)':>10} {'次数/iter':>10} {'占比':>7}")
    print("-" * 74)
    per_iter_wall = wall_total / n * 1e3
    acc = 0.0
    for key in sorted(TIMES, key=lambda k: -TIMES[k]):
        ms = TIMES[key] / n * 1e3
        acc += ms
        print(f"{key:<42} {ms:>10.3f} {COUNTS[key] / n:>10.1f} {ms / per_iter_wall * 100:>6.1f}%")
    print("-" * 74)
    ms_host = host_total / n * 1e3
    print(f"{'host 侧已计入小计':<42} {acc:>10.3f} {'':>10} {acc / per_iter_wall * 100:>6.1f}%")
    print(f"{'host 侧总计（含未计入的 python 开销）':<42} {ms_host:>10.3f} {'':>10} "
          f"{ms_host / per_iter_wall * 100:>6.1f}%")
    print(f"{'同步后总计':<42} {per_iter_wall:>10.3f} {'':>10} {100.0:>6.1f}%")
    print(f"{'⇒ 设备侧（总计 - host 侧总计）':<42} {per_iter_wall - ms_host:>10.3f} {'':>10} "
          f"{(per_iter_wall - ms_host) / per_iter_wall * 100:>6.1f}%")
    print("\n注：launch: 段是异步入队的 host 成本，不是设备执行时间。"
          "\n    设备侧那一行才是算力账 —— 如果它很小，优化方向就在 host 侧。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
