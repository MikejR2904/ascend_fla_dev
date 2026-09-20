"""A2 backward inverse_epilogue: un-gate the inverse_mm gradients into dq/dk/dv/dbeta/dg_core and k_exp.

Port of ascriptor ``a5/kda_bwd/kernels/inverse_epilogue.py`` to the A2 (c220) facade. Arithmetic and order
unchanged, per chunk and value head (gates are log2, hence the ln2 scaling before ``exp``)::

    exp_g   = exp(g * ln2)                 exp_lmg = exp((g_last - g) * ln2)
    dq      = d_qg * exp_g * D^-0.5        k_exp   = k * exp_g
    dk_kg   = d_kg * exp_lmg               dv      = d_v_beta * beta
    dbeta   = sum_D(d_v_beta * v) + sum_D(d_k_beta_g * k_exp)
    dk      = dk_kg + d_k_beta_g * beta * exp_g
    dg      = q * dq - k * dk_kg + d_k_beta_g * k_exp * beta
    dg[63] += sum_D(h * dh) * exp_g_last + sum_tokens(k * dk_kg)

A2 form: no ``@vf``; one tile op per row per term, the three reductions via a two-stage ``cadd`` plus a scalar
read, and the per-token ``beta``/``dbeta`` scalars through ``Var``. Public ``q``/``k``/``v``/``beta`` and the
public gradients are token-major 2-D views; the chain caches stay chunk-major.
"""

import math

from ascriptor.a2 import *

L = 64

D = 128

LN2 = math.log(2.0)

SCALE = 1.0 / (D ** 0.5)


@kernel(mode="vec")
def inverse_epilogue_a2_kernel(
    d_qg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_kg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_v_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    d_k_beta_g: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    v: GM[bf16, ('BT', 'HVK')],
    g_cumsum: GM[bf16, ('BT', 'HVK')],
    beta: GM[bf16, ('BT', 'HVB')],
    h: GM[bf16, ('B', 'C', 'HV', 128, 128)],
    dh: GM[bf16, ('B', 'C', 'HV', 128, 128)],
    dq_hv: GM[bf16, ('BT', 'HVK')],
    dk_hv: GM[bf16, ('BT', 'HVK')],
    dv: GM[bf16, ('BT', 'HVK')],
    dbeta: GM[f32, ('BT', 'HVB')],
    dg_core: GM[bf16, ('BT', 'HVK')],
    k_exp: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    B: i32,
    Hq: i32,  # named Hq, not H: the ACLNN API lowercases it and it would collide with the h tensor
    HV: i32,
    C: i32,
):
    src_b_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    g_ub = Tensor(DT.float, [L, D], Position.UB)
    q_ub = Tensor(DT.float, [L, D], Position.UB)
    k_ub = Tensor(DT.float, [L, D], Position.UB)
    v_ub = Tensor(DT.float, [L, D], Position.UB)
    dqg_ub = Tensor(DT.float, [L, D], Position.UB)
    dkg_ub = Tensor(DT.float, [L, D], Position.UB)
    dvb_ub = Tensor(DT.float, [L, D], Position.UB)
    dkbg_ub = Tensor(DT.float, [L, D], Position.UB)
    dq_f_ub = Tensor(DT.float, [L, D], Position.UB)
    dk_f_ub = Tensor(DT.float, [L, D], Position.UB)
    dv_f_ub = Tensor(DT.float, [L, D], Position.UB)
    dg_f_ub = Tensor(DT.float, [L, D], Position.UB)
    kexp_f_ub = Tensor(DT.float, [L, D], Position.UB)
    out_b_ub = Tensor(DT.bfloat16, [L, D], Position.UB)
    dbeta_f_ub = Tensor(DT.float, [L, 8], Position.UB)
    hrow_b_ub = Tensor(DT.bfloat16, [1, D], Position.UB)
    dhrow_b_ub = Tensor(DT.bfloat16, [1, D], Position.UB)
    hrow_ub = Tensor(DT.float, [1, D], Position.UB)
    dhrow_ub = Tensor(DT.float, [1, D], Position.UB)
    expg_ub = Tensor(DT.float, [1, D], Position.UB)
    explmg_ub = Tensor(DT.float, [1, D], Position.UB)
    expgl_ub = Tensor(DT.float, [1, D], Position.UB)
    sh_ub = Tensor(DT.float, [1, D], Position.UB)
    dkkg_ub = Tensor(DT.float, [1, D], Position.UB)
    acc_ub = Tensor(DT.float, [1, D], Position.UB)
    term2_ub = Tensor(DT.float, [1, D], Position.UB)
    dglast_ub = Tensor(DT.float, [1, D], Position.UB)
    tmp_ub = Tensor(DT.float, [1, D], Position.UB)
    dot_ub = Tensor(DT.float, [1, 64], Position.UB)
    dot2_ub = Tensor(DT.float, [1, 64], Position.UB)

    group = Var(HV // Hq)
    work_count = B * HV * C
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)
    beta_val = Var(0.0, dtype=DT.float)
    dot_val = Var(0.0, dtype=DT.float)
    sum_val = Var(0.0, dtype=DT.float)
    n_d = L * D

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            bhv = Var(work // C)
            hv_idx = Var(bhv % HV)
            b_idx = Var(bhv // HV)
            h_idx = Var(hv_idx // group)
            row0 = Var(b_idx * C * L + c_idx * L)
            qk_col = Var(h_idx * D)
            hv_col = Var(hv_idx * D)

            src_b_ub[0:L, 0:D] <<= g_cumsum[row0:row0 + L, hv_col:hv_col + D]
            cast(g_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            src_b_ub[0:L, 0:D] <<= q[row0:row0 + L, qk_col:qk_col + D]
            cast(q_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            src_b_ub[0:L, 0:D] <<= k[row0:row0 + L, qk_col:qk_col + D]
            cast(k_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            src_b_ub[0:L, 0:D] <<= v[row0:row0 + L, hv_col:hv_col + D]
            cast(v_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            src_b_ub[0:L, 0:D] <<= d_qg[b_idx, hv_idx, c_idx, 0:L, 0:D]
            cast(dqg_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            src_b_ub[0:L, 0:D] <<= d_kg[b_idx, hv_idx, c_idx, 0:L, 0:D]
            cast(dkg_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            src_b_ub[0:L, 0:D] <<= d_v_beta[b_idx, hv_idx, c_idx, 0:L, 0:D]
            cast(dvb_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)
            src_b_ub[0:L, 0:D] <<= d_k_beta_g[b_idx, hv_idx, c_idx, 0:L, 0:D]
            cast(dkbg_ub[0:L, 0:D], src_b_ub[0:L, 0:D], round_mode=RoundMode.NONE, count=n_d)

            # exp_g_last, and term1 of d_g_last = sum_D(h[row] * dh[row]) * exp_g_last[row]
            muls(sh_ub[0:1, 0:D], g_ub[L - 1:L, 0:D], LN2, count=D)
            exp(expgl_ub[0:1, 0:D], sh_ub[0:1, 0:D], count=D)
            for gk in range(0, D):
                hrow_b_ub[0:1, 0:D] <<= h[b_idx, c_idx, hv_idx, gk:gk + 1, 0:D]
                dhrow_b_ub[0:1, 0:D] <<= dh[b_idx, c_idx, hv_idx, gk:gk + 1, 0:D]
                cast(hrow_ub[0:1, 0:D], hrow_b_ub[0:1, 0:D], round_mode=RoundMode.NONE, count=D)
                cast(dhrow_ub[0:1, 0:D], dhrow_b_ub[0:1, 0:D], round_mode=RoundMode.NONE, count=D)
                mul(tmp_ub[0:1, 0:D], hrow_ub[0:1, 0:D], dhrow_ub[0:1, 0:D], count=D)
                cadd(dot_ub[0:1, 0:2], tmp_ub[0:1, 0:D], repeat=2, count_per_rep=64)
                cadd(dot2_ub[0:1, 0:1], dot_ub[0:1, 0:2], repeat=1, count_per_rep=2)
                dot_val.GetValueFrom(dot2_ub[0:1, 0:1])
                sum_val.GetValueFrom(expgl_ub[0:1, gk:gk + 1])
                sum_val.set(sum_val * dot_val)
                sum_val.SetValueTo(dglast_ub[0:1, gk:gk + 1])

            dup(term2_ub[0:1, 0:D], 0.0, count=D)
            for r in range(0, L):
                muls(sh_ub[0:1, 0:D], g_ub[r:r + 1, 0:D], LN2, count=D)
                exp(expg_ub[0:1, 0:D], sh_ub[0:1, 0:D], count=D)
                sub(sh_ub[0:1, 0:D], g_ub[L - 1:L, 0:D], g_ub[r:r + 1, 0:D], count=D)
                muls(sh_ub[0:1, 0:D], sh_ub[0:1, 0:D], LN2, count=D)
                exp(explmg_ub[0:1, 0:D], sh_ub[0:1, 0:D], count=D)
                beta_val.GetValueFrom(beta[row0 + r:row0 + r + 1, hv_idx:hv_idx + 1])

                # dq = d_qg * exp_g * scale
                mul(dq_f_ub[r:r + 1, 0:D], dqg_ub[r:r + 1, 0:D], expg_ub[0:1, 0:D], count=D)
                muls(dq_f_ub[r:r + 1, 0:D], dq_f_ub[r:r + 1, 0:D], SCALE, count=D)
                # dk_from_kg = d_kg * exp(g_last - g) ; k_exp = k * exp_g
                mul(dkkg_ub[0:1, 0:D], dkg_ub[r:r + 1, 0:D], explmg_ub[0:1, 0:D], count=D)
                mul(kexp_f_ub[r:r + 1, 0:D], k_ub[r:r + 1, 0:D], expg_ub[0:1, 0:D], count=D)
                # dv = d_v_beta * beta
                muls(dv_f_ub[r:r + 1, 0:D], dvb_ub[r:r + 1, 0:D], beta_val, count=D)

                # dbeta = sum_D(d_v_beta * v) + sum_D(d_k_beta_g * k_exp)
                mul(tmp_ub[0:1, 0:D], dvb_ub[r:r + 1, 0:D], v_ub[r:r + 1, 0:D], count=D)
                cadd(dot_ub[0:1, 0:2], tmp_ub[0:1, 0:D], repeat=2, count_per_rep=64)
                cadd(dot2_ub[0:1, 0:1], dot_ub[0:1, 0:2], repeat=1, count_per_rep=2)
                sum_val.GetValueFrom(dot2_ub[0:1, 0:1])
                mul(tmp_ub[0:1, 0:D], dkbg_ub[r:r + 1, 0:D], kexp_f_ub[r:r + 1, 0:D], count=D)
                cadd(dot_ub[0:1, 0:2], tmp_ub[0:1, 0:D], repeat=2, count_per_rep=64)
                cadd(dot2_ub[0:1, 0:1], dot_ub[0:1, 0:2], repeat=1, count_per_rep=2)
                dot_val.GetValueFrom(dot2_ub[0:1, 0:1])
                sum_val.set(sum_val + dot_val)
                sum_val.SetValueTo(dbeta_f_ub[r:r + 1, 0:1])

                # dk = dk_from_kg + d_k_beta_g * beta * exp_g
                muls(tmp_ub[0:1, 0:D], dkbg_ub[r:r + 1, 0:D], beta_val, count=D)
                mul(tmp_ub[0:1, 0:D], tmp_ub[0:1, 0:D], expg_ub[0:1, 0:D], count=D)
                add(dk_f_ub[r:r + 1, 0:D], dkkg_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)

                # dg = q * dq - k * dk_from_kg + d_k_beta_g * k_exp * beta
                mul(acc_ub[0:1, 0:D], q_ub[r:r + 1, 0:D], dq_f_ub[r:r + 1, 0:D], count=D)
                mul(tmp_ub[0:1, 0:D], k_ub[r:r + 1, 0:D], dkkg_ub[0:1, 0:D], count=D)
                sub(acc_ub[0:1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                add(term2_ub[0:1, 0:D], term2_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)
                mul(tmp_ub[0:1, 0:D], dkbg_ub[r:r + 1, 0:D], kexp_f_ub[r:r + 1, 0:D], count=D)
                muls(tmp_ub[0:1, 0:D], tmp_ub[0:1, 0:D], beta_val, count=D)
                add(dg_f_ub[r:r + 1, 0:D], acc_ub[0:1, 0:D], tmp_ub[0:1, 0:D], count=D)

            # dg[L-1] += term1 + term2
            add(tmp_ub[0:1, 0:D], dglast_ub[0:1, 0:D], term2_ub[0:1, 0:D], count=D)
            add(dg_f_ub[L - 1:L, 0:D], dg_f_ub[L - 1:L, 0:D], tmp_ub[0:1, 0:D], count=D)

            cast(out_b_ub[0:L, 0:D], dq_f_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            dq_hv[row0:row0 + L, hv_col:hv_col + D] <<= out_b_ub[0:L, 0:D]
            cast(out_b_ub[0:L, 0:D], dk_f_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            dk_hv[row0:row0 + L, hv_col:hv_col + D] <<= out_b_ub[0:L, 0:D]
            cast(out_b_ub[0:L, 0:D], dv_f_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            dv[row0:row0 + L, hv_col:hv_col + D] <<= out_b_ub[0:L, 0:D]
            cast(out_b_ub[0:L, 0:D], dg_f_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            dg_core[row0:row0 + L, hv_col:hv_col + D] <<= out_b_ub[0:L, 0:D]
            cast(out_b_ub[0:L, 0:D], kexp_f_ub[0:L, 0:D], round_mode=RoundMode.TO_EVEN, count=n_d)
            k_exp[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= out_b_ub[0:L, 0:D]
            ub_to_gm_pad(dbeta[row0:row0 + L, hv_idx:hv_idx + 1], dbeta_f_ub[0:L, 0:1], L, 1, HV - 1, 0)

    return dq_hv, dk_hv, dv, dbeta, dg_core, k_exp
