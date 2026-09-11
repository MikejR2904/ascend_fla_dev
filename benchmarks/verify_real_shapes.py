#!/usr/bin/env python3
"""真实形状下的精度验收（需要 A5 NPU + 带 ``ascend950`` 算子包的 CANN）。

**为什么要有这个脚本。** 到目前为止本仓**所有**精度数字的形状都是玩具规模 ——
`kda_fwd` 接线 H=1/HV=1~2/C=1~2、`kda_bwd` 契约五 case 最大 HV2/C2、层级验证
B1/T128/H1/HV2、宽门控跨度 B1/T64。**最大的 C 是 2，最大的 H/HV 是 2。**
而性能一直在真实形状上测（`kimi_linear_layer` B1/H32/HV32/C16/T1024 等）。
于是"算子精度在预算内"这句话的依据里，没有一个点落在模型真会用的形状上。

这与刚修完的门控跨度是**同一类风险**：契约声明的域 ≠ 模型会用的域，窄域全绿不代表可用
（AGENTS.md §6"按量程失效的缺陷"）。那次是量程维度，这次是形状维度。

真实形状多出三件窄形状**结构上**测不到的东西，本脚本逐一对应一个 check：

``drift``
    C=16 意味着 chunk 间 state 传递串 16 层深。C≤2 上看不出 `final_state` 的误差
    是**随链长累积**还是**恒定**。做法：固定每 chunk 的输入分布，扫 C=1…16，看曲线形状。
    判的是趋势而不是单点 —— 单点在预算内但斜率为正，说明长序列会出问题。

``bitwise``
    H/HV=32 才真正喂满 `block_dim=4` 的核切分（contract 的 core_ownership 按 `B*HV` 与
    `B*HV*C` 切，kimi 形状下是 32 与 512；窄形状下是 1~4，连一个核组都喂不满）。
    **核切分只是把同样的算式分给不同核，bd=1 与 bd=4 的输出应当逐位相同** —— 这是个
    不需要参考的硬判据：不同就是切分 bug，不是精度问题。
    一个算子名一个进程一份 build（AGENTS.md §6 铁律二），所以本 check 自动派子进程。

``gqa``
    GQA 分组至今只跨过一组（HV/H=2 且 H=1）。qwen 形状是 H16/HV32，16 组。
    除了整体误差，还做**头独立性**检查：KDA 的头彼此独立，所以把第 r 组单独切出来
    按 H=1/HV=2 跑，结果应当与全量跑的对应切片一致。它同时验分组映射和核切分 ——
    映射错了会表现为"整体误差不大但某组明显更差"，那种错在 H=1 上根本不可能出现。

``bwd``
    反向在真实形状上的精度。参考是 fp32 逐 token 递推 + autograd，但 1024 步的图
    在 HV=32 下要十几 GB，所以**按 chunk 做 gradient checkpointing** ——
    checkpoint 是重算而不是近似，fp32 的精确性不变，峰值内存降到一个 chunk 的量级。

**实验设计：门控跨度固定在 46**（契约 case 所在的档），这样任何差异都只能归因于形状。
形状与跨度会不会交互，用 ``--span`` 另外跑一档看。

用法::

    python benchmarks/verify_real_shapes.py --check drift
    python benchmarks/verify_real_shapes.py --check bitwise          # 自动派 bd=1/bd=4 两个子进程
    python benchmarks/verify_real_shapes.py --check gqa
    python benchmarks/verify_real_shapes.py --check bwd --span 46
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.ops.kda import prepare                                   # noqa: E402
from ascend_fla.ops.kda.chunk import (                                   # noqa: E402
    L_PER_CHUNK,
    MAX_GATE_SPAN,
    _gate_span,
    chunk_kda_fwd,
    chunk_kda_fwd_with_caches,
)
from ascend_fla.ops.kda.chunk_bwd import chunk_kda_bwd                   # noqa: E402
from ascend_fla.reference.kda import kda_recurrent_ref                   # noqa: E402

D = 128
#: 来自 docs/matrix/models.json 的 test_case_shapes —— 不在这里另造形状
SHAPES = {
    "kimi_linear_layer": dict(B=1, H=32, HV=32, C=16),
    "qwen3_next_layer": dict(B=1, H=16, HV=32, C=16),
    "long_context": dict(B=1, H=16, HV=32, C=64),
}
BUDGET = {"o": 0.05, "final_state": 0.05,
          "dq": 0.05, "dk": 0.15, "dv": 0.05, "dbeta": 0.05, "dg": 0.25, "dh0": 0.05}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def make_inputs(B, H, HV, C, span, seed=2026, *, want_grads=False):
    """在 CPU 上造一份 chunk 内门控跨度约为 ``span`` 的真实形状输入。

    每 chunk 的 g 分布固定（先造单位 g 再整体缩放），所以扫 C 时"每 chunk 有多深"不变 ——
    这正是 ``drift`` 要的控制变量：只有链长在变。
    """
    T = C * L_PER_CHUNK
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, generator=gen), dim=-1)
    g_unit = -torch.rand(B, T, HV, D, generator=gen) * 0.03
    scale = span / _gate_span(g_unit.float(), C, on_cpu=True)
    x = dict(
        q=q.bfloat16().contiguous(), k=k.bfloat16().contiguous(),
        v=(torch.randn(B, T, HV, D, generator=gen) * 0.04).bfloat16().contiguous(),
        g=(g_unit * scale).float().contiguous(),
        beta=(torch.rand(B, T, HV, generator=gen) * 0.45 + 0.05).contiguous(),
        h0=(torch.randn(B, HV, D, D, generator=gen) * 0.01).contiguous(),
    )
    if want_grads:
        x["do"] = (torch.randn(B, T, HV, D, generator=gen) * 0.04).bfloat16().contiguous()
        x["dht"] = (torch.randn(B, HV, D, D, generator=gen) * 0.01).bfloat16().contiguous()
    return x


def to_npu(x: dict) -> dict:
    return {k: v.to("npu") for k, v in x.items()}


def ref_fwd(x: dict):
    """fp32 递推参考。**必须把 v 升到 fp32 再传** —— `kda_recurrent_ref` 的返回 dtype
    跟 ``v`` 走（`reference/kda.py:76`），喂 bf16 的 v 会让"fp32 参考"自己先舍到 bf16，
    那就不是语义权威而是第二个被测对象了。
    """
    return kda_recurrent_ref(
        x["q"].float(), x["k"].float(), x["v"].float(), x["g"].float(), x["beta"].float(),
        initial_state=x["h0"].float(), output_final_state=True)


# --------------------------------------------------------------------------- drift
def check_drift(args) -> int:
    """`final_state` / `o` 的误差是随 chunk 链长累积，还是恒定？"""
    shape = SHAPES["kimi_linear_layer"]
    print(f"\n=== drift：kimi 形状 H{shape['H']}/HV{shape['HV']}，跨度 {args.span}，"
          f"bd={args.block_dim} ===")
    print("  每 chunk 的门控深度固定，只有 C 在变 —— 看的是斜率，不是单点")
    rows = []
    for C in (1, 2, 4, 8, 16):
        x = make_inputs(**shape | dict(C=C), span=args.span)
        assert x["q"].shape[1] == C * L_PER_CHUNK
        dev = to_npu(x)
        o, ht = chunk_kda_fwd(dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"],
                              initial_state=dev["h0"], output_final_state=True,
                              block_dim=args.block_dim)
        o_ref, ht_ref = ref_fwd(x)
        rows.append((C, rel_l2(o, o_ref), rel_l2(ht, ht_ref)))
        print(f"  C={C:<3d} T={C * L_PER_CHUNK:<5d} o={rows[-1][1]:.3e}  "
              f"final_state={rows[-1][2]:.3e}")
    first, last = rows[0], rows[-1]
    print(f"  C=1 → C=16：o ×{last[1] / first[1]:.2f}，final_state ×{last[2] / first[2]:.2f}")
    over = [(n, C, e) for C, eo, eh in rows
            for n, e in (("o", eo), ("final_state", eh)) if not e < BUDGET[n]]
    print("  判定：" + ("全部在预算内" if not over else f"超预算 {over}"))
    return 1 if over else 0


# ------------------------------------------------------------------------- bitwise
def _dump_for_bitwise(args) -> int:
    """子进程：按给定 block_dim 跑前向+反向，把每个输出的 sha256 与张量落盘。"""
    out = pathlib.Path(args._dump)
    out.mkdir(parents=True, exist_ok=True)
    digests = {}
    for name in args.shapes:
        shape = SHAPES[name]
        x = make_inputs(**shape, span=args.span, want_grads=True)
        dev = to_npu(x)
        o, ht, caches = chunk_kda_fwd_with_caches(
            dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"], None, dev["h0"],
            block_dim=args.block_dim)
        got = dict(o=o, final_state=ht)
        got.update(chunk_kda_bwd(
            q=dev["q"], k=dev["k"], v=dev["v"], beta=dev["beta"].bfloat16(),
            do=dev["do"], dht=dev["dht"], caches=caches, block_dim=args.block_dim))
        for tname, t in got.items():
            cpu = t.cpu().contiguous()
            key = f"{name}.{tname}"
            digests[key] = hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()
            torch.save(cpu, out / f"{key}.bd{args.block_dim}.pt")
        print(f"  [bd={args.block_dim}] {name} 落盘 {len(got)} 个张量")
    (out / f"digests.bd{args.block_dim}.json").write_text(
        json.dumps(digests, indent=2), encoding="utf-8")
    return 0


def check_bitwise(args) -> int:
    """bd=1 与 bd=4 的输出应当逐位相同 —— 核切分不改算式。"""
    print(f"\n=== bitwise：bd=1 vs bd={args.max_bd}，形状 {args.shapes}，跨度 {args.span} ===")
    print("  核切分只是把同样的算式分给不同核 → 逐位相同。不同即切分 bug，不是精度问题")
    print("  一个算子名一份 build，所以两个 bd 各派一个子进程（AGENTS.md §6 铁律二）")
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="afla_bitwise_"))
    try:
        for bd in (1, args.max_bd):
            cmd = [sys.executable, __file__, "--check", "bitwise", "--_dump", str(tmp),
                   "--block-dim", str(bd), "--span", str(args.span),
                   "--shapes", *args.shapes]
            r = subprocess.run(cmd, text=True, capture_output=True)
            sys.stdout.write(r.stdout)
            if r.returncode:
                sys.stderr.write(r.stderr[-3000:])
                return r.returncode
        d1 = json.loads((tmp / "digests.bd1.json").read_text())
        d4 = json.loads((tmp / f"digests.bd{args.max_bd}.json").read_text())
        assert d1.keys() == d4.keys()
        bad = []
        for key in d1:
            same = d1[key] == d4[key]
            note = "逐位相同" if same else "**不同**"
            if not same:
                a = torch.load(tmp / f"{key}.bd1.pt").float()
                b = torch.load(tmp / f"{key}.bd{args.max_bd}.pt").float()
                n_diff = int((a != b).sum())
                note += (f" 差 {n_diff}/{a.numel()} 个元素，max_abs_diff="
                         f"{(a - b).abs().max().item():.3e}，相对 L2={rel_l2(b, a):.3e}")
                bad.append(key)
            print(f"  {key:<34} {note}")
        print("  判定：" + ("全部逐位相同" if not bad else f"这些量受 block_dim 影响：{bad}"))
        return 1 if bad else 0
    finally:
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()


# ----------------------------------------------------------------------------- gqa
def check_gqa(args) -> int:
    """qwen 形状 H16/HV32：逐组误差 + 头独立性。"""
    shape = SHAPES["qwen3_next_layer"]
    H, HV, C = shape["H"], shape["HV"], shape["C"]
    groups = HV // H
    print(f"\n=== gqa：qwen 形状 H{H}/HV{HV}（{H} 个 q/k 头 × {groups} 个 v 头），"
          f"C={C}，跨度 {args.span}，bd={args.block_dim} ===")
    x = make_inputs(**shape, span=args.span)
    dev = to_npu(x)
    o, ht = chunk_kda_fwd(dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"],
                          initial_state=dev["h0"], output_final_state=True,
                          block_dim=args.block_dim)
    o_ref, ht_ref = ref_fwd(x)
    print(f"  整体：o={rel_l2(o, o_ref):.3e}  final_state={rel_l2(ht, ht_ref):.3e}")

    # 逐 v 头的误差 —— 分组映射错会表现为某些头明显更差，而不是整体变差
    o_c, oref_c = o.cpu().float(), o_ref.float()
    per_head = [rel_l2(o_c[:, :, i], oref_c[:, :, i]) for i in range(HV)]
    print(f"  逐 v 头 o 误差：min={min(per_head):.3e} max={max(per_head):.3e} "
          f"max/min={max(per_head) / max(min(per_head), 1e-30):.2f}")
    worst = sorted(range(HV), key=lambda i: -per_head[i])[:3]
    print(f"  最差三个头：{[(i, f'{per_head[i]:.3e}') for i in worst]}")

    # 头独立性：第 r 组单独按 H=1/HV=2 跑，应与全量跑的对应切片一致
    print("  头独立性（第 r 组单独跑 vs 全量跑的切片）：")
    ok = True
    for r in (0, H // 2, H - 1):
        vsl = slice(r * groups, (r + 1) * groups)
        sub = chunk_kda_fwd(
            dev["q"][:, :, r:r + 1].contiguous(), dev["k"][:, :, r:r + 1].contiguous(),
            dev["v"][:, :, vsl].contiguous(), dev["g"][:, :, vsl].contiguous(),
            dev["beta"][:, :, vsl].contiguous(),
            initial_state=dev["h0"][:, vsl].contiguous(), output_final_state=True,
            block_dim=args.block_dim)
        e_o = rel_l2(sub[0], o[:, :, vsl])
        e_h = rel_l2(sub[1], ht[:, vsl])
        # 这是"同一算式、不同并行规模"，本应逐位相同；放一个极小容差给核切分留余地
        flag = "" if max(e_o, e_h) < 1e-6 else "  ← 超出 1e-6，分组映射或核切分可疑"
        ok &= max(e_o, e_h) < 1e-6
        print(f"    组 {r:<3d} v 头 {vsl.start}..{vsl.stop - 1}：o={e_o:.3e} "
              f"final_state={e_h:.3e}{flag}")
    over = [n for n, e in (("o", rel_l2(o, o_ref)), ("final_state", rel_l2(ht, ht_ref)))
            if not e < BUDGET[n]]
    print("  判定：" + ("整体在预算内，头独立" if not over and ok
                      else f"超预算 {over}；头独立={ok}"))
    return 0 if (not over and ok) else 1


# ----------------------------------------------------------------------------- bwd
def _ref_grads_checkpointed(x: dict, C: int) -> dict:
    """fp32 递推参考的梯度，按 chunk 做 gradient checkpointing。

    1024 步的完整图在 HV=32 下要十几 GB。checkpoint 是**重算**不是近似 —— fp32 的
    精确性不变，峰值内存降到一个 chunk（64 步）的量级。
    """
    from torch.utils.checkpoint import checkpoint

    leaves = {n: x[n].clone().float().requires_grad_(True)
              for n in ("q", "k", "v", "beta", "g", "h0")}

    def seg(q, k, v, g, beta, state):
        return kda_recurrent_ref(q, k, v, g, beta, initial_state=state,
                                 output_final_state=True)

    state, outs = leaves["h0"], []
    for c in range(C):
        s = slice(c * L_PER_CHUNK, (c + 1) * L_PER_CHUNK)
        o_c, state = checkpoint(
            seg, leaves["q"][:, s], leaves["k"][:, s], leaves["v"][:, s],
            leaves["g"][:, s], leaves["beta"][:, s], state, use_reentrant=False)
        outs.append(o_c)
    o = torch.cat(outs, dim=1)
    loss = (o * x["do"].float()).sum() + (state * x["dht"].float()).sum()
    names = ("q", "k", "v", "beta", "g", "h0")
    grads = torch.autograd.grad(loss, [leaves[n] for n in names])
    return dict(zip(("dq", "dk", "dv", "dbeta", "dg", "dh0"), grads))


def check_bwd(args) -> int:
    shape = SHAPES[args.shapes[0]]
    C = shape["C"]
    print(f"\n=== bwd：{args.shapes[0]} H{shape['H']}/HV{shape['HV']}/C{C}，"
          f"跨度 {args.span}，bd={args.block_dim} ===")
    x = make_inputs(**shape, span=args.span, want_grads=True)
    got_span = _gate_span(x["g"], C, on_cpu=True)
    print(f"  实际跨度 {got_span:.2f}（闸 {MAX_GATE_SPAN['stable']['backward']}）")
    dev = to_npu(x)
    _, _, caches = chunk_kda_fwd_with_caches(
        dev["q"], dev["k"], dev["v"], dev["g"], dev["beta"], None, dev["h0"],
        block_dim=args.block_dim)
    got = chunk_kda_bwd(q=dev["q"], k=dev["k"], v=dev["v"], beta=dev["beta"].bfloat16(),
                        do=dev["do"], dht=dev["dht"], caches=caches,
                        block_dim=args.block_dim)
    broken = [n for n in got if not got[n].cpu().float().isfinite().all()]
    if broken:
        print(f"  ✗ 这些梯度不是有限值：{broken}")
        return 1
    want = _ref_grads_checkpointed(x, C)
    errs = {n: rel_l2(got[n], want[n]) for n in want}
    print("  " + "  ".join(f"{n}={errs[n]:.3e}/{BUDGET[n]}" for n in errs))
    over = {n: e for n, e in errs.items() if not e < BUDGET[n]}
    print("  判定：" + ("全部在预算内" if not over else f"超预算 {over}"))
    return 1 if over else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", required=True,
                    choices=("drift", "bitwise", "gqa", "bwd"))
    ap.add_argument("--block-dim", type=int, default=4)
    ap.add_argument("--max-bd", type=int, default=4, help="bitwise 对比的上限 bd")
    ap.add_argument("--span", type=float, default=46.0,
                    help="门控跨度。默认 46 = 契约 case 所在的档，固定它才能把差异归因到形状")
    ap.add_argument("--shapes", nargs="+", default=["kimi_linear_layer"],
                    choices=list(SHAPES))
    ap.add_argument("--_dump", help="内部：bitwise 子进程的落盘目录")
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401  —— 注册 "npu" 设备类型，不导入 .to("npu") 就报错
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2

    if args._dump:
        prepare("a5", args.block_dim, backward=True)
        return _dump_for_bitwise(args)
    if args.check == "bitwise":
        return check_bitwise(args)

    prepare("a5", args.block_dim, backward=(args.check == "bwd"))
    return {"drift": check_drift, "gqa": check_gqa, "bwd": check_bwd}[args.check](args)


if __name__ == "__main__":
    raise SystemExit(main())
