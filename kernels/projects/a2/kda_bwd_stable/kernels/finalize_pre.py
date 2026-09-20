"""A2 backward finalize_pre: gate-anchored q/k scalings and the three masked dA matrices.

Port of this repository's ``a5/kda_bwd_stable/kernels/finalize_pre.py`` (midpoint anchor, which doubles the
usable gate span) to the A2 (c220) facade. Arithmetic and order unchanged, per chunk and value head::

    gmid   = g_last / 2
    rscale = exp((g - gmid) * ln2)      cscale = exp((gmid - g) * ln2)
    q_scaled = bf16(q * rscale)  k_scaled = bf16(k * rscale)  kg = bf16(k * cscale)
    M_qk   = tril_incl(dAqk)     M_base = tril_strict(dAkk)   M_beta = M_base * beta[row]

A2 form: no ``@vf``; whole-tile ops with one op per row where a row-broadcast or a per-row scalar is needed;
masking by an in-kernel column index with ``compare_scalar`` + ``select`` (never multiply by zero, so an
overflowed value cannot become NaN). Public ``q``/``k``/``beta`` and the chain caches are read token-major
through 2-D views; the GQA head map is in-kernel.
"""

import math

from ascriptor.a2 import *

L = 64

D = 128

LN2 = math.log(2.0)

MID = 0.5


@kernel(mode="vec")
def finalize_pre_a2_kernel(
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    g_cumsum: GM[bf16, ('BT', 'HVK')],
    beta: GM[bf16, ('BT', 'HVB')],
    dAqk: GM[bf16, ('BT', 'HVL')],
    dAkk: GM[bf16, ('BT', 'HVL')],
    q_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    k_scaled: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    kg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    m_qk: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    m_base: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    m_beta: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    B: i32,
    H: i32,
    HV: i32,
    C: i32,
):
    qb_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    kb_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    gb_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    q_f_ub = Tensor(DT.float, [L, D], Position.UB)
    k_f_ub = Tensor(DT.float, [L, D], Position.UB)
    g_f_ub = Tensor(DT.float, [L, D], Position.UB)
    gmid_ub = Tensor(DT.float, [1, D], Position.UB)
    sh_ub = Tensor(DT.float, [L, D], Position.UB)
    rs_ub = Tensor(DT.float, [L, D], Position.UB)
    cs_ub = Tensor(DT.float, [L, D], Position.UB)
    prod_ub = Tensor(DT.float, [L, D], Position.UB)
    out_b_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    daqk_b_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    dakk_b_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    daqk_f_ub = Tensor(DT.float, [L, L], Position.UB)
    dakk_f_ub = Tensor(DT.float, [L, L], Position.UB)
    mask_f_ub = Tensor(DT.float, [L, L], Position.UB)
    mbeta_f_ub = Tensor(DT.float, [L, L], Position.UB)
    mask_b_ub = Tensor(DT.bfloat16, [L, L], Position.UB)
    col_ub = Tensor(DT.float, [1, L], Position.UB)
    zero_ub = Tensor(DT.float, [1, L], Position.UB)
    pred_ub = Tensor(DT.uint8, [1, 32], Position.UB)

    group = Var(HV // H)
    work_count = B * HV * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)
    col_val = Var(0.0, dtype=DT.float)
    row_val = Var(0.0, dtype=DT.float)
    beta_val = Var(0.0, dtype=DT.float)
    n_d = L * D
    n_l = L * L

    with auto_sync():
        col_val.set(0.0)
        for j in range(0, L):
            col_val.SetValueTo(col_ub[0:1, j:j + 1])
            col_val.set(col_val + 1.0)
        dup(zero_ub[0:1, 0:L], 0.0, count=L)

        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bhv = Var(work // C)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv // HV)
            h_idx = Var(hv_idx // group)
            row0 = Var(b_idx * C * L + c_idx * L)
            qk_col = Var(h_idx * D)
            g_col = Var(hv_idx * D)
            l_col = Var(hv_idx * L)

            qb_ub[0:L, 0:D] <<= q[row0:row0 + L, qk_col:qk_col + D]
            kb_ub[0:L, 0:D] <<= k[row0:row0 + L, qk_col:qk_col + D]
            gb_ub[0:L, 0:D] <<= g_cumsum[row0:row0 + L, g_col:g_col + D]
            daqk_b_ub[0:L, 0:L] <<= dAqk[row0:row0 + L, l_col:l_col + L]
            dakk_b_ub[0:L, 0:L] <<= dAkk[row0:row0 + L, l_col:l_col + L]

            cast(q_f_ub[0:L, 0:D], qb_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            cast(k_f_ub[0:L, 0:D], kb_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            cast(g_f_ub[0:L, 0:D], gb_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            cast(daqk_f_ub[0:L, 0:L], daqk_b_ub[0:L, 0:L], round_mode=RoundMode.NONE, count=n_l)
            cast(dakk_f_ub[0:L, 0:L], dakk_b_ub[0:L, 0:L], round_mode=RoundMode.NONE, count=n_l)

            # rscale = exp((g - g_last/2) * ln2), cscale = exp((g_last/2 - g) * ln2)
            muls(gmid_ub[0:1, 0:D], g_f_ub[L - 1:L, 0:D], MID, count=D)
            for r in range(0, L):
                sub(sh_ub[r:r + 1, 0:D], g_f_ub[r:r + 1, 0:D], gmid_ub[0:1, 0:D], count=D)
            muls(sh_ub[0:L, 0:D], sh_ub[0:L, 0:D], LN2, count=n_d)
            exp(rs_ub[0:L, 0:D], sh_ub[0:L, 0:D], count=n_d)
            muls(cs_ub[0:L, 0:D], sh_ub[0:L, 0:D], -1.0, count=n_d)
            exp(cs_ub[0:L, 0:D], cs_ub[0:L, 0:D], count=n_d)

            mul(prod_ub[0:L, 0:D], q_f_ub[0:L, 0:D], rs_ub[0:L, 0:D], count=n_d)
            cast(out_b_ub[0:L, 0:D], prod_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            q_scaled[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= out_b_ub[0:L, 0:D]

            mul(prod_ub[0:L, 0:D], k_f_ub[0:L, 0:D], rs_ub[0:L, 0:D], count=n_d)
            cast(out_b_ub[0:L, 0:D], prod_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            k_scaled[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= out_b_ub[0:L, 0:D]

            mul(prod_ub[0:L, 0:D], k_f_ub[0:L, 0:D], cs_ub[0:L, 0:D], count=n_d)
            cast(out_b_ub[0:L, 0:D], prod_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            kg[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= out_b_ub[0:L, 0:D]

            # M_qk = tril incl. diagonal, M_base = strictly lower, M_beta = M_base * beta[row]
            row_val.set(0.0)
            for r in range(0, L):
                compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LE)
                select(mask_f_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], daqk_f_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                       SelectMode.TENSOR_SCALAR)
                row_val.set(row_val + 1.0)
            cast(mask_b_ub[0:L, 0:L], mask_f_ub[0:L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_l)
            m_qk[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mask_b_ub[0:L, 0:L]

            row_val.set(0.0)
            for r in range(0, L):
                compare_scalar(pred_ub[0:1, 0:32], col_ub[0:1, 0:L], row_val, CompareMode.LT)
                select(mask_f_ub[r:r + 1, 0:L], pred_ub[0:1, 0:32], dakk_f_ub[r:r + 1, 0:L], zero_ub[0:1, 0:L],
                       SelectMode.TENSOR_SCALAR)
                beta_val.GetValueFrom(beta[row0 + r:row0 + r + 1, hv_idx:hv_idx + 1])
                muls(mbeta_f_ub[r:r + 1, 0:L], mask_f_ub[r:r + 1, 0:L], beta_val, count=L)
                row_val.set(row_val + 1.0)
            cast(mask_b_ub[0:L, 0:L], mask_f_ub[0:L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_l)
            m_base[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mask_b_ub[0:L, 0:L]
            cast(mask_b_ub[0:L, 0:L], mbeta_f_ub[0:L, 0:L], round_mode=RoundMode.TO_EVEN, count=n_l)
            m_beta[b_idx, hv_idx, c_idx, 0:L, 0:L] <<= mask_b_ub[0:L, 0:L]

    return q_scaled, k_scaled, kg, m_qk, m_base, m_beta
