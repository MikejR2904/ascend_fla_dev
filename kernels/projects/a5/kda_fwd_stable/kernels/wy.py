"""WY 表示 —— 把 ``eg_last / eg`` 换成 ``exp(gc_last − gc)``，去掉除法。

改自 ascriptor 的 ``kda_fwd/kernels/wy.py``（只读引用，AGENTS.md §3）。矩阵乘、流水、
``abeta`` 的处理一概不变；差别只在 ``wy_preprocess_vf`` 里门控那几行。

原版收 ``eg = exp(gc)`` 与 ``eg_last``，算

    qg   = q · eg                     绝对量
    kexp = k · eg                     绝对量
    kg   = k · (eg_last / eg)         ← 除法

前两项是绝对量，``eg`` 下溢到 0 时结果为 0，这是**正确的**（该 token 完全不见旧状态）。
第三项是比值：``eg_last`` 与 ``eg`` 同时下溢到 0 就是 ``0/0 = NaN``。实测跨度 88.67 时
``kg`` 出 11 个 NaN。

这里收 log 空间的 ``g_cumsum``，自己做指数：

    eg   = exp(gc)                    下溢到 0 —— 正确，不是病态
    kg   = k · exp(gc_last − gc)      先减后指数，恒 ≤1，任何跨度都安全

所以这个 kernel **不需要中点技巧** —— 它要么是绝对量（下溢即正确），要么是可以先做减法
的差值。只有 ``intra.py`` 的成对衰减因为要走 matmul 才必须分解。
"""

from ascriptor.a5 import *

L = 64

K_DIM = 128

V_DIM = 128

K_BLOCK = 64

K_TILES = K_DIM // K_BLOCK

HALF_L = L // 2

SPLIT_N = 64


@vf()
def wy_preprocess_stable_vf(
    q_ub: Tensor,
    k_ub: Tensor,
    beta_full_ub: Tensor,
    gc_ub: Tensor,
    glast_ub: Tensor,
    Akk_ub: Tensor,
    kexp_ub: Tensor,
    abeta_ub: Tensor,
    qg_ub: Tensor,
    kg_ub: Tensor,
    rows: Var,
):
    q_row = Reg(DT.float)
    k_row = Reg(DT.float)
    gc_row = Reg(DT.float)
    gc_last = Reg(DT.float)
    eg_row = Reg(DT.float)
    ratio_row = Reg(DT.float)
    a_row = Reg(DT.float)
    beta_row = Reg(DT.float)
    tmp_k = Reg(DT.float)
    tmp_a = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(K_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        for tile in range(K_TILES):
            k_begin = tile * K_BLOCK
            k_end = k_begin + K_BLOCK
            q_row <<= q_ub[r:r + 1, k_begin:k_end]
            k_row <<= k_ub[r:r + 1, k_begin:k_end]
            gc_row <<= gc_ub[r:r + 1, k_begin:k_end]
            gc_last <<= glast_ub[0:1, k_begin:k_end]

            eg_row <<= gc_row.exp()                     # exp(gc)，下溢到 0 即正确

            tmp_k <<= q_row * eg_row
            row_bf16 <<= tmp_k.astype(DT.bfloat16)
            reg_to_ub_downsample(qg_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

            tmp_k <<= k_row * eg_row
            row_bf16 <<= tmp_k.astype(DT.bfloat16)
            reg_to_ub_downsample(kexp_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

            # 原版这里是 eg_last / eg_row；改成先减后指数，恒 ≤1
            ratio_row <<= gc_last + gc_row.neg()
            ratio_row <<= ratio_row.exp()
            tmp_k <<= k_row * ratio_row
            row_bf16 <<= tmp_k.astype(DT.bfloat16)
            reg_to_ub_downsample(kg_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

        a_row <<= Akk_ub[r:r + 1, 0:L]
        beta_row <<= beta_full_ub[0:1, 0:L]
        tmp_a <<= a_row * beta_row
        row_bf16 <<= tmp_a.astype(DT.bfloat16)
        reg_to_ub_downsample(abeta_ub[r:r + 1, 0:L], row_bf16, mask=row_mask_bf16)


@kernel()
def kda_sub3_wy_stable_kernel(
    q: GM[bf16, ('B', 'H', 'C', 64, 128)],
    k: GM[bf16, ('B', 'H', 'C', 64, 128)],
    v: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    beta: GM[f32, ('B', 'HV', 'C', 64)],
    Akk: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    g_cumsum: GM[f32, ('B', 'HV', 'C', 64, 128)],
    w: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    u: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    qg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    kg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    B: i32,
    H: i32,
    HV: i32,
    C: i32,
    length_per_chunk: i32,
    head_dim: i32,
    value_dim: i32,
):
    vcmutex = VcMutex(
        0,
        depth=2,  # Preserve the former default for the two rotating L1 slots.
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    l1_abeta = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_kexp = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_v = DBuff(DT.bfloat16, [L, V_DIM], Position.L1)
    l0c_w = DBuff(DT.float, [L, K_DIM], Position.L0C)
    l0c_u = DBuff(DT.float, [L, V_DIM], Position.L0C)

    q_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    k_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    beta_full_ub = DBuff(DT.float, [1, L], Position.UB)
    gc_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    glast_ub = DBuff(DT.float, [1, K_DIM], Position.UB)
    Akk_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    kexp_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    abeta_ub = DBuff(DT.bfloat16, [HALF_L, L], Position.UB)
    qg_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    kg_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    group = Var(HV // H)
    vcnt = Var(0)
    ccnt = Var(0)

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            tmp = Var(work // C)
            hv_idx = Var(tmp % HV)
            b_idx = Var(tmp // HV)
            h_idx = Var(hv_idx // group)

            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Min(row_begin + HALF_L, L)
            rows_this = Var(row_end - row_begin)
            q_ub[vcnt][0:rows_this, 0:K_DIM] <<= q[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
            k_ub[vcnt][0:rows_this, 0:K_DIM] <<= k[b_idx, h_idx, c_idx, row_begin:row_end, 0:K_DIM]
            beta_full_ub[vcnt][0:1, 0:L] <<= beta[b_idx, hv_idx, c_idx, 0:L]
            gc_ub[vcnt][0:rows_this, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM]
            glast_ub[vcnt][0:1, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, L - 1:L, 0:K_DIM]
            Akk_ub[vcnt][0:rows_this, 0:L] <<= Akk[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L]
            wy_preprocess_stable_vf(
                q_ub[vcnt],
                k_ub[vcnt],
                beta_full_ub[vcnt],
                gc_ub[vcnt],
                glast_ub[vcnt],
                Akk_ub[vcnt],
                kexp_ub[vcnt],
                abeta_ub[vcnt],
                qg_ub[vcnt],
                kg_ub[vcnt],
                rows_this,
            )

            qg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM] <<= qg_ub[vcnt][0:rows_this, 0:K_DIM]
            kg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM] <<= kg_ub[vcnt][0:rows_this, 0:K_DIM]

            vcmutex.lock()
            l1_abeta[ccnt][row_begin:row_end, 0:L] <<= abeta_ub[vcnt][0:rows_this, 0:L]
            l1_kexp[ccnt][row_begin:row_end, 0:K_DIM] <<= kexp_ub[vcnt][0:rows_this, 0:K_DIM]
            vcmutex.ready()

            l1_v[ccnt][0:L, 0:V_DIM] <<= v[b_idx, hv_idx, c_idx, 0:L, 0:V_DIM]

            vcnt += 1

            vcmutex.wait()
            matmul(l0c_w[ccnt], l1_abeta[ccnt], l1_kexp[ccnt].T, m=L, n=K_DIM, k=L, splitn=SPLIT_N)
            matmul(l0c_u[ccnt], l1_abeta[ccnt], l1_v[ccnt].T, m=L, n=V_DIM, k=L, splitn=SPLIT_N)
            vcmutex.free()
            w[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= l0c_w[ccnt][0:L, 0:K_DIM]
            u[b_idx, hv_idx, c_idx, 0:L, 0:V_DIM] <<= l0c_u[ccnt][0:L, 0:V_DIM]
            ccnt += 1

    return w, u, qg, kg
