#!/usr/bin/env python3
"""反向链在宽门控跨度下还有限吗？—— 前向已用 stable 根治，反向是独立的一处。

**为什么要单独测反向。** 读 ascriptor ``kda_bwd`` 的 ``finalize_pre.py`` /
``finalize_post.py``，它把 chunk 内成对衰减分解成::

    rscale = exp((g − g_last)·ln2)   # g − g_last ≥ 0 → 最大 exp(+span)
    cscale = exp((g_last − g)·ln2)   # ≤ 0            → 最小 exp(−span)

``rscale`` 是**真上溢**（fp32 与 bf16 的上溢线都在 ``ln(MAX) ≈ 88.72``，而
``q_scaled`` / ``k_scaled`` 是 bf16 GM 输出），配对的 ``kg`` 同时下溢到 0，下游四个矩阵乘
里 ``inf × 0 = NaN``。方向与前向那边的下溢**相反**，所以前向修好不代表反向好了。

本脚本逐档扫门控跨度，报：

1. 六个梯度的**有限性**（非有限元素个数）——这是主判据；
2. ``stable`` 与 ``upstream`` 在**重叠域**内的相对 L2 —— 应当 ≈0，用来证明改锚点是
   恒等变形（锚点常数在四个成对矩阵乘里逐项抵消）。

**不经 autograd**，直接调 ``chunk_kda_fwd_with_caches`` + ``chunk_kda_bwd``，``do`` / ``dht``
在 CPU 上造好再 H2D —— 这样在**缺内置算子包**的机器上也能跑（见 AGENTS.md §5 的可用面表：
那种机器上 NPU 侧的 Cast / sum / zeros 全不可用）。

一个进程只跑一个 ``impl``：同名算子多 build 会互相覆盖（AGENTS.md §6 铁律二），而两套
实现虽然改了名，``prepare`` 的链也是按 impl 选的。多 impl 由本脚本自动派子进程。

用法::

    python benchmarks/probe_bwd_span.py                       # 两个 impl 各一个子进程
    python benchmarks/probe_bwd_span.py --impl stable --block-dim 1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

B, T, H, HV, D = 1, 64, 1, 1, 128
GRADS = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
#: 门控倍数 → 跨度约为 0.0315 × 64 × mult（g_raw ∈ [-0.03,0] 的期望）
MULTS = (1.0, 10.0, 45.0, 67.0, 80.0, 90.0, 112.0, 134.0, 156.0)


def _inputs(seed: int = 2026):
    """在 CPU 上造一份输入。不用 NPU 算子 —— 缺算子包的机器上造不了。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=g), dim=-1)
    return dict(
        q=q.bfloat16(), k=k.bfloat16(),
        v=(torch.randn(B, T, HV, D, generator=g) * 0.04).bfloat16(),
        beta=(torch.rand(B, T, HV, generator=g) * 0.45 + 0.05),
        g_unit=-torch.rand(B, T, HV, D, generator=g) * 0.03,
        h0=(torch.randn(B, HV, D, D, generator=g) * 0.01),
        do=(torch.randn(B, T, HV, D, generator=g) * 0.04).bfloat16(),
        dht=(torch.randn(B, HV, D, D, generator=g) * 0.01).bfloat16(),
    )


def _bad(t: torch.Tensor) -> tuple[int, int, int]:
    """(非有限数, nan 数, inf 数)。一律先 D2H 再判 —— NPU 上 isfinite 要算子。"""
    f = t.cpu().float()
    return int((~f.isfinite()).sum()), int(f.isnan().sum()), int(f.isinf().sum())


def run(impl: str, block_dim: int, mults=None) -> list[dict]:
    from ascend_fla.ops.kda import chunk_kda_fwd_with_caches, prepare
    from ascend_fla.ops.kda.chunk import _gate_span
    from ascend_fla.ops.kda.chunk_bwd import chunk_kda_bwd

    prepare("a5", block_dim, backward=True, impl=impl)
    x = _inputs()
    npu = lambda t: t.to("npu")  # noqa: E731

    rows = []
    for mult in (mults or MULTS):
        gf = (x["g_unit"] * mult).float()
        span = _gate_span(gf, T // 64, on_cpu=True)
        _, _, caches = chunk_kda_fwd_with_caches(
            npu(x["q"]), npu(x["k"]), npu(x["v"]), npu(gf), npu(x["beta"]),
            None, npu(x["h0"]),
            block_dim=block_dim, check_gate_range=False, impl=impl,
        )
        grads = chunk_kda_bwd(
            q=npu(x["q"]), k=npu(x["k"]), v=npu(x["v"]),
            beta=npu(x["beta"].bfloat16()), do=npu(x["do"]), dht=npu(x["dht"]),
            caches=caches, block_dim=block_dim, impl=impl,
        )
        row = dict(impl=impl, block_dim=block_dim, mult=mult, span=span)
        for name in GRADS:
            bad, nan, inf = _bad(grads[name])
            row[name] = dict(bad=bad, nan=nan, inf=inf,
                             n=grads[name].numel(),
                             vals=grads[name].cpu().float().flatten()[:4].tolist())
        # 全量存下来太大，只留一份可比的指纹：每个梯度的 L2 与前 64 个元素
        row["fingerprint"] = {n: dict(l2=grads[n].cpu().float().norm().item(),
                                      head=grads[n].cpu().float().flatten()[:64].tolist())
                              for n in GRADS}
        rows.append(row)
        # 不用嵌套 f-string：远端是 python 3.11，同引号嵌套在那儿是 SyntaxError
        flags = "  ".join(
            "{}:{}".format(n, "OK" if row[n]["bad"] == 0 else "BAD{}".format(row[n]["bad"]))
            for n in GRADS)
        print(f"{impl:>8} span={span:>7.2f}  {flags}", flush=True)
    return rows


def _rel_l2(a: list[float], b: list[float]) -> float:
    ta, tb = torch.tensor(a), torch.tensor(b)
    return ((ta - tb).norm() / tb.norm().clamp_min(1e-30)).item()


def compare(rows: list[dict]) -> None:
    """重叠域内 stable 与 upstream 的逐梯度相对 L2。应当 ≈0（恒等变形）。"""
    by = {}
    for r in rows:
        by.setdefault(r["mult"], {})[r["impl"]] = r
    print(f"\n{'span':>8} " + " ".join(f"{n:>10}" for n in GRADS) + "   ← stable vs upstream 相对 L2")
    for mult in sorted(by):
        pair = by[mult]
        if len(pair) != 2:
            continue
        st, up = pair.get("stable"), pair.get("upstream")
        if any(up["fingerprint"][n]["l2"] != up["fingerprint"][n]["l2"] for n in GRADS):
            print(f"{st['span']:>8.2f} upstream 侧含 NaN，无法比对")
            continue
        if any(up[n]["bad"] for n in GRADS):
            print(f"{st['span']:>8.2f} " + " ".join(f"{'upNaN':>10}" for _ in GRADS))
            continue
        errs = [_rel_l2(st["fingerprint"][n]["head"], up["fingerprint"][n]["head"])
                for n in GRADS]
        print(f"{st['span']:>8.2f} " + " ".join(f"{e:>10.3e}" for e in errs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=("stable", "upstream"), default=None,
                    help="不传则两个都跑（各一个子进程）")
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--mults", type=float, nargs="+", default=None,
                    help="门控倍数列表，默认 MULTS。本脚本的输入分布下跨度 ≈ 1.117 × 倍数")
    ap.add_argument("--json-out", type=pathlib.Path)
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2

    if args.impl is not None:
        rows = run(args.impl, args.block_dim, args.mults)
        if args.json_out:
            args.json_out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return 0

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        for impl in ("upstream", "stable"):
            out = pathlib.Path(td) / f"{impl}.json"
            r = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()),
                 "--impl", impl, "--block-dim", str(args.block_dim), "--json-out", str(out),
                 *(["--mults", *map(str, args.mults)] if args.mults else [])],
                capture_output=True, text=True)
            sys.stdout.write("\n".join(l for l in r.stdout.splitlines()
                                       if not l.startswith("[lint]")) + "\n")
            if r.returncode != 0:
                print(f"impl={impl} 子进程失败 exit={r.returncode}", file=sys.stderr)
                print(r.stderr[-3000:], file=sys.stderr)
                return 1
            rows += json.loads(out.read_text(encoding="utf-8"))
    compare(rows)
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"\n已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
