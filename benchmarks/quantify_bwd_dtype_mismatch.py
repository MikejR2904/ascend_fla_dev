#!/usr/bin/env python3
"""量化 kda 的 fwd/bwd dtype 不一致对梯度的影响 —— 第二期的前置。

问题（`docs/matrix/gaps.json` 的 ``kda-fwd-bwd-dtype-mismatch``）：

* ``kda_fwd`` 声明 ``g`` / ``beta`` / ``initial_state`` 为 **float32**；
* ``kda_bwd`` 对同名张量声明 **bfloat16**，且 ``dg`` / ``dbeta`` / ``dh0`` 也是 bfloat16。

所以组装 ``autograd.Function`` 时，forward 保存的 FP32 张量必须降到 BF16 才能喂给
backward，而返回的梯度又是 BF16。这两步降精度**不在任何一侧的契约预算内** —— 在往
里插 ``.to(bfloat16)`` 之前，得先知道它值多少。

本脚本把三个效应拆开测（全部在 CPU fp32 参考上做，不需要 NPU）：

A. **保存值降精度**：g/beta/initial_state 走一遍 bf16 往返再算梯度。
   这是 bwd kernel 实际会看到的输入。
B. **梯度输出精度**：把 fp32 基准梯度舍到 bf16。这是 bwd kernel 输出端的损失。
C. **两者叠加**：A 的输入 + B 的输出舍入。这才是端到端会发生的事。

基准是"全程 fp32"。注意 q/k/v 在 fwd 与 bwd 两侧都是 bf16，不在本次考察范围内 ——
它们的精度损失属于算子契约已声明的部分。

用法::

    python quantify_bwd_dtype_mismatch.py
    python quantify_bwd_dtype_mismatch.py --shape kimi_linear_layer --json-out out.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from ascend_fla.reference.kda import kda_chunk_vectorized  # noqa: E402

# 这些形状比 models.json 的 test_case_shapes 小：本脚本要在 CPU 上跑反向，
# T=1024/HV=32 的 fp32 反向图在 Mac 上要几分钟而结论不变。B/H/HV 的比例保持一致。
SHAPES = {
    "smoke": dict(B=1, H=1, HV=1, T=64),
    "multi_chunk": dict(B=1, H=1, HV=1, T=256),
    "gva": dict(B=1, H=2, HV=4, T=256),
    "kimi_shaped": dict(B=1, H=4, HV=4, T=256),
}
K_DIM = V_DIM = 128
# 被 bwd ABI 降精度的张量。q/k/v 两侧都是 bf16，不在考察范围。
DOWNCAST = ("g", "beta", "initial_state")


def make_inputs(B, H, HV, T, seed=2026):
    """与 kda_fwd contract 的 input_generation 同分布（见 bench_kda_torch_npu.py）。"""
    gen = torch.Generator().manual_seed(seed)
    f = dict(generator=gen, dtype=torch.float32)
    return dict(
        q=torch.randn(B, T, H, K_DIM, **f) * 0.04,
        k=torch.randn(B, T, H, K_DIM, **f) * 0.04,
        v=torch.randn(B, T, HV, V_DIM, **f) * 0.04,
        g=torch.empty(B, T, HV, K_DIM).uniform_(-0.03, 0.0, generator=gen),
        beta=torch.empty(B, T, HV).uniform_(0.05, 0.5, generator=gen),
        initial_state=torch.randn(B, HV, K_DIM, V_DIM, **f) * 0.01,
    )


def grads(x: dict, *, downcast_saved: bool, seed=7) -> dict[str, torch.Tensor]:
    """对同一个 loss 求梯度。``downcast_saved`` 时把被 bwd ABI 降精度的张量走 bf16 往返。

    走往返（``.bfloat16().float()``）而不是直接用 bf16 算图：bwd kernel 内部以什么精度
    计算由 kernel 决定，我们要隔离的是"**输入信息被截断**"这一项。
    """
    leaves = {}
    for name, t in x.items():
        t = t.clone()
        if downcast_saved and name in DOWNCAST:
            t = t.bfloat16().float()
        leaves[name] = t.requires_grad_(True)

    o, state = kda_chunk_vectorized(**leaves, output_final_state=True)
    # 固定的随机上游梯度：等价于给 o 和 final_state 各接一个线性头。
    # 只对 o 求和会让 dht 这条路径完全测不到。
    gen = torch.Generator().manual_seed(seed)
    do = torch.randn(o.shape, generator=gen, dtype=torch.float32)
    dht = torch.randn(state.shape, generator=gen, dtype=torch.float32)
    (o.float() * do).sum().add_((state.float() * dht).sum()).backward()
    return {name: t.grad.detach().clone() for name, t in leaves.items()}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", default="all", choices=[*SHAPES, "all"])
    ap.add_argument("--json-out", type=pathlib.Path)
    args = ap.parse_args()

    names = list(SHAPES) if args.shape == "all" else [args.shape]
    print("基准 = 全程 fp32。相对 L2 残差，越小越好。")
    print(f"被 bwd ABI 降到 bf16 的输入：{', '.join(DOWNCAST)}")
    print(f"bf16 的机器 epsilon = {torch.finfo(torch.bfloat16).eps:.3e}"
          f"（≈ 单次舍入的相对误差上界）\n")

    records = []
    for name in names:
        shp = SHAPES[name]
        x = make_inputs(**shp)
        ref = grads(x, downcast_saved=False)
        down = grads(x, downcast_saved=True)
        rounded = {k: v.bfloat16().float() for k, v in ref.items()}
        both = {k: v.bfloat16().float() for k, v in down.items()}

        print(f"=== {name}  B{shp['B']} H{shp['H']} HV{shp['HV']} T{shp['T']} ===")
        print(f"{'梯度':<16} {'A 保存值降精度':>15} {'B 输出舍到bf16':>15} {'C 两者叠加':>13}")
        print("-" * 64)
        rec = dict(shape=name, dims=shp, per_grad={})
        for g_name in ref:
            a = rel_l2(down[g_name], ref[g_name])
            b = rel_l2(rounded[g_name], ref[g_name])
            c = rel_l2(both[g_name], ref[g_name])
            print(f"d{g_name:<15} {a:>15.3e} {b:>15.3e} {c:>13.3e}")
            rec["per_grad"][f"d{g_name}"] = dict(saved_downcast=a, output_rounded=b, both=c)
        worst = max(rec["per_grad"].items(), key=lambda kv: kv[1]["both"])
        rec["worst"] = dict(grad=worst[0], **worst[1])
        print(f"最差：{worst[0]}  叠加后 relL2={worst[1]['both']:.3e}\n")
        records.append(rec)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"已写入 {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
