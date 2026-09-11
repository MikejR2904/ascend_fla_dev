#!/usr/bin/env python3
"""decode 路径的真机验收（需要 A5 NPU）。

四件事，每件对应 decode 成立的一个前提：

``acc``
    对 fp32 递推参考的精度。decode 全程 fp32、无 bf16 中间量，所以预算取
    **1e-05 量级**而不是 chunk 路径的 5e-02 —— 松预算会让真实的退化藏进去。

``chain``
    **state 串接逐位相同**：逐 token 调 T 次并串接 state，必须与一次调 T 个 token
    比特一致。decode 的正确性就建立在这条上 —— 不是"差不多"，是逐位。

``bd``
    声明的 ``block_dim`` 每个值都要实测。本 kernel 只用向量核、不碰 cube，
    ``GetVecNum() == 2*block_dim`` 而物理上有 56 个向量核，所以上限 28 是**推断**，
    必须测出来（超过物理核数会在硬件 barrier 死锁，见 AGENTS.md §5）。

``split``
    把一次调用的耗时拆成 host 布局 / 桥+设备 / 输出重排，并用 T=1 与 T=16 的差
    求设备侧的边际。**不拆就别下"谁是瓶颈"的结论** —— 我先推断 decode 是带宽瓶颈，
    实测是每次调用的固定成本，差了一个数量级（AGENTS.md §6 铁律一）。

**一个 bd 一个进程**（铁律二：一个算子名一份 build），``bd`` 那项自动派子进程。

用法::

    python benchmarks/verify_decode.py                    # 四件全做
    python benchmarks/verify_decode.py --check acc chain
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.ops.kda.chunk import HEAD_DIM, VALUE_DIM          # noqa: E402
from ascend_fla.ops.kda.fused_recurrent import (                  # noqa: E402
    SUPPORTED_BLOCK_DIM,
    T_MAX,
    fused_recurrent_kda,
)
from ascend_fla.reference.kda import kda_recurrent_ref            # noqa: E402

D = 128
#: 全 fp32 路径的预算。比 chunk 路径（5e-02）紧三个数量级 —— 见模块文档。
BUDGET = 1e-5
#: decode 形状：kimi_linear_layer 的头配置，T=1。
KIMI = dict(B=1, H=32, HV=32)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def make(B, T, H, HV, seed=2026):
    gen = torch.Generator().manual_seed(seed)
    f = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    return dict(
        q=torch.nn.functional.normalize(f(B, T, H, D), dim=-1),
        k=torch.nn.functional.normalize(f(B, T, H, D), dim=-1),
        v=f(B, T, HV, D) * 0.04,
        g=-torch.rand(B, T, HV, D, generator=gen) * 0.03,
        beta=torch.rand(B, T, HV, generator=gen) * 0.45 + 0.05,
        h0=f(B, HV, D, D) * 0.01,
    )


def npu(x: torch.Tensor) -> torch.Tensor:
    return x.to("npu")


def reference(x: dict):
    return kda_recurrent_ref(
        x["q"].float(), x["k"].float(), x["v"].float(), x["g"].float(), x["beta"].float(),
        initial_state=x["h0"].float(), output_final_state=True)


def call(x: dict, bd: int, state=None, sl: slice | None = None):
    g = (lambda n: x[n][:, sl]) if sl is not None else (lambda n: x[n])
    return fused_recurrent_kda(
        npu(g("q")), npu(g("k")), npu(g("v")), npu(g("g")), npu(g("beta")),
        initial_state=npu(x["h0"]) if state is None else state,
        output_final_state=True, block_dim=bd)


# ------------------------------------------------------------------------------ acc
def check_acc(bd: int) -> int:
    print(f"\n=== acc：对 fp32 递推参考的相对 L2（预算 {BUDGET:.0e}，bd={bd}）===")
    cases = ((1, 1, 1, 1), (1, 1, 1, 2), (1, 4, 1, 1), (1, 1, 2, 4),
             (1, T_MAX, 1, 1), (1, 1, 16, 32), (1, 1, 32, 32), (1, 8, 32, 32))
    bad = []
    for B, T, H, HV in cases:
        x = make(B, T, H, HV)
        o, ht = call(x, bd)
        o_ref, ht_ref = reference(x)
        eo, es = rel_l2(o, o_ref), rel_l2(ht, ht_ref)
        flag = "" if max(eo, es) < BUDGET else "  ← 超预算"
        if flag:
            bad.append((B, T, H, HV, eo, es))
        print(f"  B{B} T{T:<3d} H{H:<3d} HV{HV:<3d}  o={eo:.3e}  final_state={es:.3e}{flag}")
    print("  判定：" + ("全部在预算内" if not bad else f"超预算 {bad}"))
    return 1 if bad else 0


# ---------------------------------------------------------------------------- chain
def check_chain(bd: int) -> int:
    """逐 token 串接必须与整段调用**逐位相同**。decode 的正确性就是这条。"""
    print(f"\n=== chain：state 串接（{T_MAX} 次单 token vs 一次 {T_MAX} token，bd={bd}）===")
    bad = []
    for B, H, HV in ((1, 4, 8), (1, 32, 32)):
        x = make(B, T_MAX, H, HV)
        o_all, ht_all = call(x, bd)
        state, outs = npu(x["h0"]), []
        for i in range(T_MAX):
            oi, state = call(x, bd, state=state, sl=slice(i, i + 1))
            outs.append(oi)
        o_step = torch.cat(outs, dim=1)
        same_o = torch.equal(o_step.cpu(), o_all.cpu())
        same_s = torch.equal(state.cpu(), ht_all.cpu())
        if not (same_o and same_s):
            bad.append((B, H, HV, rel_l2(o_step, o_all), rel_l2(state, ht_all)))
        print(f"  B{B} H{H:<3d} HV{HV:<3d}  o 逐位相同={same_o}  final_state 逐位相同={same_s}"
              f"   （不同则 relL2 o={rel_l2(o_step, o_all):.2e} s={rel_l2(state, ht_all):.2e}）")
    print("  判定：" + ("逐位一致" if not bad else f"**不一致** {bad} —— decode 的前提不成立"))
    return 1 if bad else 0


# ------------------------------------------------------------------------------- bd
def _one_bd(bd: int, t: int) -> int:
    x = make(1, t, KIMI["H"], KIMI["HV"], seed=7)
    d = {k: npu(v) for k, v in x.items()}
    fn = lambda: fused_recurrent_kda(  # noqa: E731
        d["q"], d["k"], d["v"], d["g"], d["beta"],
        initial_state=d["h0"], output_final_state=True, block_dim=bd)
    o, ht = fn()
    o_ref, ht_ref = reference(x)
    us = timed(fn)
    print(f"  bd={bd:<3d} T={t:<3d} {us:8.1f} µs/次 ({us / t:7.1f} µs/token)   "
          f"o={rel_l2(o, o_ref):.2e} state={rel_l2(ht, ht_ref):.2e}")
    return 0 if max(rel_l2(o, o_ref), rel_l2(ht, ht_ref)) < BUDGET else 1


def timed(fn, warm: int = 5, iters: int = 50) -> float:
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def check_bd(_bd: int) -> int:
    print(f"\n=== bd：声明的 {SUPPORTED_BLOCK_DIM} 每个值都要实测（kimi decode 形状）===")
    print("  28 → 56 个向量核，是物理上限；超过会在硬件 barrier 死锁，所以这一项是安全检查")
    rc = 0
    for bd in SUPPORTED_BLOCK_DIM:
        r = subprocess.run([sys.executable, __file__, "--check", "bd", "--_one", str(bd)],
                           text=True, capture_output=True)
        sys.stdout.write("".join(ln for ln in r.stdout.splitlines(keepends=True)
                                 if "bd=" in ln) or f"  bd={bd} 无输出\n")
        if r.returncode:
            sys.stderr.write(r.stderr[-600:])
            rc = 1
    print("  判定：" + ("全部跑通且在预算内" if not rc else "有档失败，见上"))
    return rc


# ---------------------------------------------------------------------------- split
def check_split(bd: int) -> int:
    """拆开量：谁是瓶颈。不拆就别下结论。"""
    print(f"\n=== split：一次调用的耗时拆解（kimi decode 形状，bd={bd}）===")
    x = make(1, 1, KIMI["H"], KIMI["HV"], seed=7)
    d = {k: npu(v) for k, v in x.items()}
    sc = HEAD_DIM ** -0.5
    bhv = lambda t: t.to(torch.float32).permute(0, 2, 1, 3).contiguous()  # noqa: E731

    def host_only():
        bhv(d["q"] * sc), bhv(d["k"]), bhv(d["v"]), bhv(d["g"])
        d["beta"].to(torch.float32).permute(0, 2, 1).contiguous().view(1, KIMI["HV"], 1, 1)
        torch.empty(1, KIMI["HV"], 1, VALUE_DIM, dtype=torch.float32, device="npu")
        torch.empty(1, KIMI["HV"], HEAD_DIM, VALUE_DIM, dtype=torch.float32, device="npu")

    full1 = timed(lambda: fused_recurrent_kda(
        d["q"], d["k"], d["v"], d["g"], d["beta"],
        initial_state=d["h0"], output_final_state=True, block_dim=bd))
    host = timed(host_only)

    xt = make(1, T_MAX, KIMI["H"], KIMI["HV"], seed=7)
    dt = {k: npu(v) for k, v in xt.items()}
    fullt = timed(lambda: fused_recurrent_kda(
        dt["q"], dt["k"], dt["v"], dt["g"], dt["beta"],
        initial_state=dt["h0"], output_final_state=True, block_dim=bd))
    marginal = (fullt - full1) / (T_MAX - 1)

    print(f"  T=1 整次      {full1:8.1f} µs")
    print(f"  其中 host 布局 {host:8.1f} µs   （{host / full1 * 100:.0f}%）")
    print(f"  T={T_MAX} 整次     {fullt:8.1f} µs")
    print(f"  设备侧边际     {marginal:8.1f} µs/token   （(T={T_MAX} − T=1) / {T_MAX - 1}）")
    print(f"  固定成本       {full1 - marginal:8.1f} µs/次")
    print("  判定：固定成本" + ("主导 —— 先解 decode-call-overhead，不是调 kernel"
                              if (full1 - marginal) > 2 * marginal else "不再主导，可以回头看 kernel"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", nargs="+", default=["acc", "chain", "bd", "split"],
                    choices=("acc", "chain", "bd", "split"))
    ap.add_argument("--block-dim", type=int, default=4)
    ap.add_argument("--_one", type=int, help="内部：bd 子进程只测这一个值")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2

    if args._one is not None:
        return _one_bd(args._one, 1)
    rc = 0
    for name in args.check:
        rc |= {"acc": check_acc, "chain": check_chain,
               "bd": check_bd, "split": check_split}[name](args.block_dim)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
