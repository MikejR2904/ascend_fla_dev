"""A2 chunk-local scores: ``Aqk`` (BF16, lower triangle incl. diagonal) and ``strict`` (FP32, strictly lower).

Port of this repository's ``kernels/projects/a5/kda_fwd_stable/kernels/intra.py`` (the midpoint-anchored gate
decomposition) to the A2 (c220) facade. Arithmetic, per chunk and value head, with ``m[d] = gc[63, d] / 2``::

    pos[i,d]   = exp(gc[i,d] - m[d])      neg[j,d] = exp(m[d] - gc[j,d])
    qg[i,d]    =  q[i,d] * pos[i,d] * scale
    kbg[i,d]   = -k[i,d] * pos[i,d] * beta[i]
    kgneg[j,d] =  k[j,d] * neg[j,d]
    Aqk        = lower_incl(qg @ kgneg^T)          strict = lower_strict(kbg @ kgneg^T)

The vector-side operation order is the A5 one (so the pre-exp values are bitwise the same); ``exp`` is the A2
instruction. What changes is the form, because A2 has neither ``@vf`` nor the direct UB->L1 / L0C->UB moves
(``dma.ub_to_l1`` and ``dma.l0c_to_ub`` are "not available on device b3" in the pinned library):

* pre (vector, 32 rows per sub-block): tile-vector ops on UB; ``qg``/``kbg``/``kgneg`` go to a two-slot GM
  workspace, handed to the cube with ``VcMutex`` (MTE3 -> MTE2). Every GM ring is indexed by the one pipeline beat
  ``pipe_work`` (producer at the beat, readers at beat - 1 / beat - 2), as the library's GMBuff pass requires.
* cube: GM -> L1, two FP32 ``matmul(..., splitk=64)`` (the pinned library inserts the A2-family PIPE_M settle for
  FP32 split-K; A2-01), L0C -> GM workspace, handed back with ``CvMutex`` (FIX -> MTE2).
* post (vector): per-row ``compare_scalar`` against an in-kernel column index and ``select`` against zero
  (``select`` replaces, so an overflowed upper-triangle value never becomes ``inf * 0 = NaN``), then BF16 cast.
* ``q``/``k`` (BF16 ``[B*T, H*128]``) and ``beta`` (FP32 ``[B*T, HV]``) are read in their public token-major
  layout; ``g_cumsum`` and the two outputs are chain-internal ``[B, HV, C, 64, *]``.
"""

from ascriptor.a2 import *

L = 64

K_DIM = 128

K_BLOCK = 64

HALF_L = L // 2


@kernel()
def kda_sub2_score_a2_kernel(
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    g_cumsum: GM[f32, ('B', 'HV', 'C', 64, 128)],
    beta: GM[f32, ('BT', 'HVB')],
    Aqk: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    strict: GM[f32, ('B', 'HV', 'C', 64, 64)],
    B: i32,
    H: i32,
    HV: i32,
    C: i32,
    scale: f32,
):
    qg_ws = GMBuff(DT.float, [L, K_DIM], slots=2, name="qg_ws")
    kbg_ws = GMBuff(DT.float, [L, K_DIM], slots=2, name="kbg_ws")
    kgneg_ws = GMBuff(DT.float, [L, K_DIM], slots=2, name="kgneg_ws")
    aqk_ws = GMBuff(DT.float, [L, L], slots=2, name="aqk_ws")
    strict_ws = GMBuff(DT.float, [L, L], slots=2, name="strict_ws")
    vcmutex = VcMutex(0, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    cvmutex = CvMutex(1, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1_qg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l1_kbg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l1_kgneg = DBuff(DT.float, [L, K_DIM], Position.L1)
    l0c_aqk = DBuff(DT.float, [L, L], Position.L0C)
    l0c_strict = DBuff(DT.float, [L, L], Position.L0C)

    qb_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    kb_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    qf_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    kf_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    sh_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    pos_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    neg_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    glast_ub = Tensor(DT.float, [1, K_DIM], Position.UB)
    mid_ub = Tensor(DT.float, [1, K_DIM], Position.UB)
    aqk_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    strict_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    aqk_m_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    strict_m_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    aqk_b_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    col_ub = Tensor(DT.float, [1, L], Position.UB)
    zero_ub = Tensor(DT.float, [1, L], Position.UB)
    pred_ub = Tensor(DT.uint8, [1, 32], Position.UB)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    group = Var(HV // H)
    cube_cnt = Var(0)
    beta_val = Var(0.0, dtype=DT.float)
    col_val = Var(0.0, dtype=DT.float)
    row_val = Var(0.0, dtype=DT.float)

    with auto_sync():
        # column index 0..63 and a zero row, built once in the kernel (no host-made mask tensors)
        with vec_scope():
            col_val.set(0.0)
            for j in range(0, L):
                col_val.SetValueTo(col_ub[0:1, j:j + 1])
                col_val.set(col_val + 1.0)
            dup(zero_ub[0:1, 0:L], 0.0, count=L)

        for pipe_work in range(work_begin, work_end + 2):
            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Var(row_begin + HALF_L)

            if pipe_work < work_end:
                c_idx = Var(pipe_work % C)
                tmp = Var(pipe_work // C)
                hv_idx = Var(tmp % HV)
                b_idx = Var(tmp // HV)
                h_idx = Var(hv_idx // group)
                tok0 = Var(b_idx * C * L + c_idx * L + row_begin)
                qk_col = Var(h_idx * K_DIM)

                qb_ub[0:HALF_L, 0:K_DIM] <<= q[tok0:tok0 + HALF_L, qk_col:qk_col + K_DIM]
                kb_ub[0:HALF_L, 0:K_DIM] <<= k[tok0:tok0 + HALF_L, qk_col:qk_col + K_DIM]
                sh_ub[0:HALF_L, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM]
                glast_ub[0:1, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, L - 1:L, 0:K_DIM]

                cast(qf_ub[0:HALF_L, 0:K_DIM], qb_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.NONE, count=HALF_L * K_DIM)
                cast(kf_ub[0:HALF_L, 0:K_DIM], kb_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.NONE, count=HALF_L * K_DIM)
                muls(mid_ub[0:1, 0:K_DIM], glast_ub[0:1, 0:K_DIM], 0.5, count=K_DIM)
                for r in range(0, HALF_L):
                    sub(sh_ub[r:r + 1, 0:K_DIM], sh_ub[r:r + 1, 0:K_DIM], mid_ub[0:1, 0:K_DIM], count=K_DIM)
                exp(pos_ub[0:HALF_L, 0:K_DIM], sh_ub[0:HALF_L, 0:K_DIM], count=HALF_L * K_DIM)
                muls(neg_ub[0:HALF_L, 0:K_DIM], sh_ub[0:HALF_L, 0:K_DIM], -1.0, count=HALF_L * K_DIM)
                exp(neg_ub[0:HALF_L, 0:K_DIM], neg_ub[0:HALF_L, 0:K_DIM], count=HALF_L * K_DIM)
                # qg = q * pos * scale
                mul(qf_ub[0:HALF_L, 0:K_DIM], qf_ub[0:HALF_L, 0:K_DIM], pos_ub[0:HALF_L, 0:K_DIM], count=HALF_L * K_DIM)
                muls(qf_ub[0:HALF_L, 0:K_DIM], qf_ub[0:HALF_L, 0:K_DIM], scale, count=HALF_L * K_DIM)
                # kbg = -(k * pos * beta_row)
                mul(pos_ub[0:HALF_L, 0:K_DIM], kf_ub[0:HALF_L, 0:K_DIM], pos_ub[0:HALF_L, 0:K_DIM], count=HALF_L * K_DIM)
                for r in range(0, HALF_L):
                    beta_val.GetValueFrom(beta[tok0 + r:tok0 + r + 1, hv_idx:hv_idx + 1])
                    beta_val.set(beta_val * -1.0)
                    muls(pos_ub[r:r + 1, 0:K_DIM], pos_ub[r:r + 1, 0:K_DIM], beta_val, count=K_DIM)
                # kgneg = k * neg
                mul(neg_ub[0:HALF_L, 0:K_DIM], kf_ub[0:HALF_L, 0:K_DIM], neg_ub[0:HALF_L, 0:K_DIM], count=HALF_L * K_DIM)

                vcmutex.lock()
                qg_ws[pipe_work][row_begin:row_end, 0:K_DIM] <<= qf_ub[0:HALF_L, 0:K_DIM]
                kbg_ws[pipe_work][row_begin:row_end, 0:K_DIM] <<= pos_ub[0:HALF_L, 0:K_DIM]
                kgneg_ws[pipe_work][row_begin:row_end, 0:K_DIM] <<= neg_ub[0:HALF_L, 0:K_DIM]
                vcmutex.ready()

            if (pipe_work > work_begin) and (pipe_work < work_end + 1):
                vcmutex.wait()
                l1_qg[cube_cnt] <<= qg_ws[pipe_work - 1][0:L, 0:K_DIM]
                l1_kbg[cube_cnt] <<= kbg_ws[pipe_work - 1][0:L, 0:K_DIM]
                l1_kgneg[cube_cnt] <<= kgneg_ws[pipe_work - 1][0:L, 0:K_DIM]
                vcmutex.free()
                matmul(l0c_aqk[cube_cnt], l1_qg[cube_cnt], l1_kgneg[cube_cnt], splitk=K_BLOCK, m=L, n=L, k=K_DIM)
                matmul(l0c_strict[cube_cnt], l1_kbg[cube_cnt], l1_kgneg[cube_cnt], splitk=K_BLOCK, m=L, n=L, k=K_DIM)
                cvmutex.lock()
                aqk_ws[pipe_work - 1][0:L, 0:L] <<= l0c_aqk[cube_cnt]
                strict_ws[pipe_work - 1][0:L, 0:L] <<= l0c_strict[cube_cnt]
                cvmutex.ready()
                cube_cnt += 1

            if pipe_work > work_begin + 1:
                work = Var(pipe_work - 2)
                c_idx = Var(work % C)
                tmp = Var(work // C)
                hv_idx = Var(tmp % HV)
                b_idx = Var(tmp // HV)

                cvmutex.wait()
                aqk_ub[0:HALF_L, 0:L] <<= aqk_ws[pipe_work - 2][row_begin:row_end, 0:L]
                strict_ub[0:HALF_L, 0:L] <<= strict_ws[pipe_work - 2][row_begin:row_end, 0:L]
                cvmutex.free()
                row_val.set(row_begin)
                for r in range(0, HALF_L):
                    compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LE)
                    select(aqk_m_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], aqk_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                           SelectMode.TENSOR_SCALAR)
                    compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LT)
                    select(strict_m_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], strict_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                           SelectMode.TENSOR_SCALAR)
                    row_val.set(row_val + 1.0)
                cast(aqk_b_ub[0:HALF_L, 0:L], aqk_m_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=HALF_L * L)
                Aqk[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L] <<= aqk_b_ub[0:HALF_L, 0:L]
                strict[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L] <<= strict_m_ub[0:HALF_L, 0:L]

    return Aqk, strict
