"""门控 cumsum —— 同时输出 log 空间的 ``g_cumsum`` 与 ``eg = exp(g_cumsum)``。

改自 ascriptor 的 ``kda_fwd/kernels/gate.py``（只读引用，AGENTS.md §3）。算术完全不变，
唯一的差别是**把 cumsum 本身也写出去**。

为什么要这么做：原版只写 ``eg``，于是下游的 scores 与 wy 必须用 ``eg_i / eg_j`` 的形式
取比值。当 chunk 内衰减深到 ``exp(cumsum)`` 下溢（fp32 的非正规数被 flush 到 0，阈值
``-ln(FLT_MIN_NORMAL) ≈ 87.3``）时，那就是 ``0/0`` 和 ``0×inf`` —— 实测输出全 NaN。
信息在 ``exp()`` 落盘那一刻就没了，下游无论怎么写都救不回来。

``g_cumsum`` 同时也是 ``kda_bwd`` 需要的九个前向检查点之一（见 gaps.json 的
``fwd-caches-not-emitted``），所以这一个输出解决两个问题。
"""

from ascriptor.a5 import *

L = 64

K_DIM = 128

K_BLOCK = 64

K_TILES = K_DIM // K_BLOCK


@vf()
def gate_cumsum_stable_vf(src_ub: Tensor, cum_ub: Tensor, dst_ub: Tensor):
    prev = Reg(DT.float)
    curr = Reg(DT.float)
    prev_exp = Reg(DT.float)

    for tile in range(K_TILES):
        k_begin = tile * K_BLOCK
        k_end = k_begin + K_BLOCK
        prev <<= src_ub[0:1, k_begin:k_end]
        cum_ub[0:1, k_begin:k_end] <<= prev
        prev_exp <<= prev.exp()
        dst_ub[0:1, k_begin:k_end] <<= prev_exp

        for r in range(1, L):
            curr <<= src_ub[r:r + 1, k_begin:k_end]
            prev <<= prev + curr
            cum_ub[r:r + 1, k_begin:k_end] <<= prev
            prev_exp <<= prev.exp()
            dst_ub[r:r + 1, k_begin:k_end] <<= prev_exp


@kernel()
def kda_sub1_gate_stable_kernel(
    g_raw: GM[f32, ('B', 'HV', 'C', 64, 128)],
    g_cumsum: GM[f32, ('B', 'HV', 'C', 64, 128)],
    eg: GM[f32, ('B', 'HV', 'C', 64, 128)],
    B: i32,
    HV: i32,
    C: i32,
    length_per_chunk: i32,
    head_dim: i32,
):
    src_ub = DBuff(DT.float, [L, K_DIM], Position.UB)
    cum_ub = DBuff(DT.float, [L, K_DIM], Position.UB)
    dst_ub = DBuff(DT.float, [L, K_DIM], Position.UB)

    work_count = B * HV * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            tmp = Var(work // C)
            hv_idx = Var(tmp % HV)
            b_idx = Var(tmp // HV)

            src_ub[work][0:L, 0:K_DIM] <<= g_raw[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
            gate_cumsum_stable_vf(src_ub[work], cum_ub[work], dst_ub[work])
            g_cumsum[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= cum_ub[work][0:L, 0:K_DIM]
            eg[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= dst_ub[work][0:L, 0:K_DIM]

    return g_cumsum, eg
