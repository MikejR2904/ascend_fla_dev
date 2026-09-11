#!/usr/bin/env python3
"""整层 decode 一步的耗时拆解（需要 A5 NPU + ascend950 算子包）。

算子侧已经量过：设备时间只有 2.7~4.8 µs/token，固定成本约 48~58 µs
（`gaps.json` 的 `decode-call-overhead`）。但**层**里还有一堆小算子：
7 个投影、3 个短卷积、softplus、sigmoid、l2norm、FusedRMSNormGated。
T=1 时它们每一个都是"几乎没有计算量但要走一次 launch"的调用。

所以这里要回答的是：**一步 decode 的时间花在哪** —— KDA 算子、投影、还是其它。
拆法是逐段关掉/替换，而不是靠推断（AGENTS.md §6 铁律一）。

用法::

    python benchmarks/bench_kda_decode_layer.py
    python benchmarks/bench_kda_decode_layer.py --hidden 2048 --num-heads 32 --num-v-heads 32
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def timed(fn, warm: int = 10, iters: int = 50) -> float:
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--num-v-heads", type=int, default=32)
    ap.add_argument("--block-dim", type=int, default=1)
    ap.add_argument("--prefill", type=int, default=128)
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
    except ImportError:
        print("本脚本只在 NPU 上有意义", file=sys.stderr)
        return 2

    from ascend_fla.layers.kda import KimiDeltaAttention
    from ascend_fla.ops.kda import prepare
    from ascend_fla.ops.kda.chunk import HEAD_DIM

    # 所有 kernel 必须在第一次执行之前编完（opp-path-read-once）
    prepare(block_dim=args.block_dim, backward=False, decode=True)

    torch.manual_seed(0)
    layer = KimiDeltaAttention(
        hidden_size=args.hidden, num_heads=args.num_heads, num_v_heads=args.num_v_heads,
        head_dim=HEAD_DIM, layer_idx=0, block_dim=args.block_dim,
    ).to("npu").eval()
    x = torch.randn(1, args.prefill + 1, args.hidden).to("npu")

    print(f"\n=== 整层 decode 一步：hidden={args.hidden} H={args.num_heads} "
          f"HV={args.num_v_heads} bd={args.block_dim} ===")
    with torch.no_grad():
        # prefill 一次，拿到 cache
        cache: dict = {}
        layer(x[:, :args.prefill], cache=cache, mode="chunk")
        step = x[:, args.prefill:args.prefill + 1]

        # 每次都从同一份 cache 出发：decode 会原地更新 cache，反复跑会让 state 漂移，
        # 但耗时与 state 的值无关 —— 拷一份出来用，免得 state 越跑越大影响不了计时却混淆语义
        frozen = {"recurrent_state": cache["recurrent_state"].clone(),
                  "conv_state": tuple(c.clone() for c in cache["conv_state"])}

        def full():
            c = {"recurrent_state": frozen["recurrent_state"],
                 "conv_state": frozen["conv_state"]}
            layer(step, cache=c, mode="fused_recurrent")

        t_full = timed(full)

        # 只做层里除 KDA 算子以外的部分：投影 + 卷积 + 门控 + o_norm + o_proj
        import torch.nn.functional as F

        def without_op():
            b, t = 1, 1
            q = layer.q_proj(step); k = layer.k_proj(step); v = layer.v_proj(step)
            if layer.use_short_conv:
                q, _ = layer.q_conv1d(q, cache=frozen["conv_state"][0], output_final_state=True)
                k, _ = layer.k_conv1d(k, cache=frozen["conv_state"][1], output_final_state=True)
                v, _ = layer.v_conv1d(v, cache=frozen["conv_state"][2], output_final_state=True)
            q = F.normalize(q.view(b, t, layer.num_heads, layer.head_k_dim).float(),
                            dim=-1, eps=1e-6).bfloat16()
            k = F.normalize(k.view(b, t, layer.num_heads, layer.head_k_dim).float(),
                            dim=-1, eps=1e-6).bfloat16()
            v = v.view(b, t, layer.num_v_heads, layer.head_v_dim).bfloat16()
            layer._gate(step, b, t)
            torch.sigmoid(layer.b_proj(step).float())
            gate = layer.g_proj(step).view(b, t, layer.num_v_heads, layer.head_v_dim)
            o = layer.o_norm(v, gate).reshape(b, t, layer.value_dim)
            layer.o_proj(o.to(layer.o_proj.weight.dtype))

        t_rest = timed(without_op)

        # 只做 KDA 算子
        from ascend_fla.ops.kda import fused_recurrent_kda
        hv, kd = layer.num_v_heads, layer.head_k_dim
        qb = torch.randn(1, 1, layer.num_heads, kd).bfloat16().to("npu")
        kb = torch.randn(1, 1, layer.num_heads, kd).bfloat16().to("npu")
        vb = torch.randn(1, 1, hv, kd).bfloat16().to("npu")
        gb = (-torch.rand(1, 1, hv, kd) * 0.03).to("npu")
        bb = torch.rand(1, 1, hv).to("npu")
        t_op = timed(lambda: fused_recurrent_kda(
            qb, kb, vb, gb, bb, initial_state=frozen["recurrent_state"],
            output_final_state=True, block_dim=args.block_dim))

    print(f"  整层一步            {t_full:8.1f} µs")
    print(f"  其中 KDA 算子       {t_op:8.1f} µs   （{t_op / t_full * 100:.0f}%）")
    print(f"  其中层里其余部分    {t_rest:8.1f} µs   （{t_rest / t_full * 100:.0f}%）")
    print(f"  48 层外推           {t_full * 48 / 1000:8.2f} ms/token")
    print("  读法：占比大的那一侧才是该优化的地方。算子侧的设备时间只有几 µs，")
    print("        所以两边的大头都是每次调用的固定成本 —— 见 gaps.json 的")
    print("        decode-call-overhead 与 decode-layer-overhead。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
