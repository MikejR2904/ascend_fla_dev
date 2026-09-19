"""A2 WY representation: ``w = (Akk diag(beta)) (k * exp(gc))``, ``u = (Akk diag(beta)) v`` plus the gated
``qg = q * exp(gc)`` and ``kg = k * exp(gc_last - gc)`` (all BF16 out).

Port of this repository's ``kernels/projects/a5/kda_fwd_stable/kernels/wy.py`` to the A2 (c220) facade. The
vector-side arithmetic and its order are the A5 ones (``exp`` after the subtraction for ``kg``, so every factor
is <= 1 and underflow to 0 is correct); what changes is the form:

* No ``@vf``: tile-vector ops on UB. The per-row ``gc_last - gc`` and ``Akk[r, :] * beta[:]`` are one op per
  row (a broadcast row operand), then whole-tile ``exp`` / ``mul`` / BF16 ``cast`` (TO_EVEN).
* No UB -> L1 on A2: ``abeta`` and ``kexp`` go to the cube through a two-slot GM workspace (``VcMutex``,
  MTE3 -> MTE2); the cube loads ``v`` straight from GM.
* ``q``/``k`` (BF16 ``[B*T, H*128]``), ``v`` (BF16 ``[B*T, HV*128]``) and ``beta`` (FP32 ``[B*T, HV]``) are read
  in their public token-major layout; the 64 ``beta`` values of a chunk are strided by ``HV`` and are gathered
  into one UB row with scalar reads. ``Akk``/``g_cumsum`` in and the four outputs are chain-internal
  ``[B, HV, C, 64, *]``.
* Both matmuls are ``splitn`` with ``is_init=True`` (disjoint L0C tiles), so there is no accumulate chain.
"""

from ascriptor.a2 import *

L = 64

K_DIM = 128

V_DIM = 128

HALF_L = L // 2

SPLIT_N = 64


@kernel()
def kda_sub3_wy_a2_kernel(
    q: GM[bf16, ('BT', 'HK')],
    k: GM[bf16, ('BT', 'HK')],
    v: GM[bf16, ('BT', 'HVV')],
    beta: GM[f32, ('BT', 'HVB')],
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
):
    abeta_ws = GMBuff(DT.bfloat16, [L, L], slots=2, name="abeta_ws")
    kexp_ws = GMBuff(DT.bfloat16, [L, K_DIM], slots=2, name="kexp_ws")
    vcmutex = VcMutex(0, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)

    l1_abeta = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_kexp = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_v = DBuff(DT.bfloat16, [L, V_DIM], Position.L1)
    l0c_w = DBuff(DT.float, [L, K_DIM], Position.L0C)
    l0c_u = DBuff(DT.float, [L, V_DIM], Position.L0C)

    qb_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    kb_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    qf_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    kf_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    gc_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    eg_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    ratio_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    tmp_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    glast_ub = Tensor(DT.float, [1, K_DIM], Position.UB)
    beta_ub = Tensor(DT.float, [1, L], Position.UB)
    akk_b_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    akk_f_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    abeta_f_ub = Tensor(DT.float, [HALF_L, L], Position.UB)
    abeta_b_ub = Tensor(DT.bfloat16, [HALF_L, L], Position.UB)
    qg_b_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    kexp_b_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    kg_b_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)

    work_count = B * HV * C
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    group = Var(HV // H)
    ccnt = Var(0)
    beta_val = Var(0.0, dtype=DT.float)
    n_elem = HALF_L * K_DIM

    with auto_sync():
        for work in range(work_begin, work_end):
            c_idx = Var(work % C)
            tmp = Var(work // C)
            hv_idx = Var(tmp % HV)
            b_idx = Var(tmp // HV)
            h_idx = Var(hv_idx // group)
            chunk_tok = Var(b_idx * C * L + c_idx * L)

            row_begin = Var(GetSubBlockIdx() * HALF_L)
            row_end = Var(row_begin + HALF_L)
            tok0 = Var(chunk_tok + row_begin)
            qk_col = Var(h_idx * K_DIM)
            v_col = Var(hv_idx * V_DIM)

            qb_ub[0:HALF_L, 0:K_DIM] <<= q[tok0:tok0 + HALF_L, qk_col:qk_col + K_DIM]
            kb_ub[0:HALF_L, 0:K_DIM] <<= k[tok0:tok0 + HALF_L, qk_col:qk_col + K_DIM]
            gc_ub[0:HALF_L, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM]
            glast_ub[0:1, 0:K_DIM] <<= g_cumsum[b_idx, hv_idx, c_idx, L - 1:L, 0:K_DIM]
            akk_b_ub[0:HALF_L, 0:L] <<= Akk[b_idx, hv_idx, c_idx, row_begin:row_end, 0:L]
            with vec_scope():
                for j in range(0, L):
                    beta_val.GetValueFrom(beta[chunk_tok + j:chunk_tok + j + 1, hv_idx:hv_idx + 1])
                    beta_val.SetValueTo(beta_ub[0:1, j:j + 1])

            cast(qf_ub[0:HALF_L, 0:K_DIM], qb_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.NONE, count=n_elem)
            cast(kf_ub[0:HALF_L, 0:K_DIM], kb_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.NONE, count=n_elem)
            exp(eg_ub[0:HALF_L, 0:K_DIM], gc_ub[0:HALF_L, 0:K_DIM], count=n_elem)
            mul(tmp_ub[0:HALF_L, 0:K_DIM], qf_ub[0:HALF_L, 0:K_DIM], eg_ub[0:HALF_L, 0:K_DIM], count=n_elem)
            cast(qg_b_ub[0:HALF_L, 0:K_DIM], tmp_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.TO_EVEN, count=n_elem)
            mul(tmp_ub[0:HALF_L, 0:K_DIM], kf_ub[0:HALF_L, 0:K_DIM], eg_ub[0:HALF_L, 0:K_DIM], count=n_elem)
            cast(kexp_b_ub[0:HALF_L, 0:K_DIM], tmp_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.TO_EVEN, count=n_elem)
            for r in range(0, HALF_L):
                sub(ratio_ub[r:r + 1, 0:K_DIM], glast_ub[0:1, 0:K_DIM], gc_ub[r:r + 1, 0:K_DIM], count=K_DIM)
            exp(ratio_ub[0:HALF_L, 0:K_DIM], ratio_ub[0:HALF_L, 0:K_DIM], count=n_elem)
            mul(tmp_ub[0:HALF_L, 0:K_DIM], kf_ub[0:HALF_L, 0:K_DIM], ratio_ub[0:HALF_L, 0:K_DIM], count=n_elem)
            cast(kg_b_ub[0:HALF_L, 0:K_DIM], tmp_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.TO_EVEN, count=n_elem)
            cast(akk_f_ub[0:HALF_L, 0:L], akk_b_ub[0:HALF_L, 0:L], round_mode=RoundMode.NONE, count=HALF_L * L)
            for r in range(0, HALF_L):
                mul(abeta_f_ub[r:r + 1, 0:L], akk_f_ub[r:r + 1, 0:L], beta_ub[0:1, 0:L], count=L)
            cast(abeta_b_ub[0:HALF_L, 0:L], abeta_f_ub[0:HALF_L, 0:L], round_mode=RoundMode.TO_EVEN, count=HALF_L * L)

            qg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM] <<= qg_b_ub[0:HALF_L, 0:K_DIM]
            kg[b_idx, hv_idx, c_idx, row_begin:row_end, 0:K_DIM] <<= kg_b_ub[0:HALF_L, 0:K_DIM]

            vcmutex.lock()
            abeta_ws[work][row_begin:row_end, 0:L] <<= abeta_b_ub[0:HALF_L, 0:L]
            kexp_ws[work][row_begin:row_end, 0:K_DIM] <<= kexp_b_ub[0:HALF_L, 0:K_DIM]
            vcmutex.ready()

            l1_v[ccnt][0:L, 0:V_DIM] <<= v[chunk_tok:chunk_tok + L, v_col:v_col + V_DIM]
            vcmutex.wait()
            l1_abeta[ccnt] <<= abeta_ws[work][0:L, 0:L]
            l1_kexp[ccnt] <<= kexp_ws[work][0:L, 0:K_DIM]
            vcmutex.free()
            matmul(l0c_w[ccnt], l1_abeta[ccnt], l1_kexp[ccnt].T, m=L, n=K_DIM, k=L, splitn=SPLIT_N)
            matmul(l0c_u[ccnt], l1_abeta[ccnt], l1_v[ccnt].T, m=L, n=V_DIM, k=L, splitn=SPLIT_N)
            w[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM] <<= l0c_w[ccnt][0:L, 0:K_DIM]
            u[b_idx, hv_idx, c_idx, 0:L, 0:V_DIM] <<= l0c_u[ccnt][0:L, 0:V_DIM]
            ccnt += 1

    return w, u, qg, kg
