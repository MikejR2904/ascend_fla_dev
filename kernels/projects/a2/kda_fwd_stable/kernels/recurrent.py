"""A2 chunk recurrence (KDA sub4 + sub5): state propagation across chunks and the chunk output.

Port of this repository's ``kernels/projects/a5/kda_fwd_stable/kernels/recurrent.py`` (the A5K-01 repaired
Aqk handoff) to the A2 (c220) facade. Per value head and 64-wide value tile, for each chunk ``c``::

    h      = bf16(S)                                  # [K=128, 64]
    qg'    = bf16(q * eg * scale)                     # [64, 128]
    prod   = w @ h                                    # cube, FP32
    v_new  = bf16(u - prod)
    S_dec  = S * eg[63, :]^T                           # per-K-row decay
    delta  = kg^T @ v_new                             # cube, FP32
    o      = qg' @ h + Aqk @ v_new                    # cube, BF16 out
    S      = S_dec + delta

The arithmetic and the precision boundaries are the A5 ones. What changes on A2:

* No ``@vf``: tile-vector ops on UB; the per-row decay is one ``muls`` per state row with a scalar read.
* No UB -> L1 / L0C -> UB on A2: the five cross-side handoffs (``h``, ``qg'``, ``v_new`` to the cube; ``prod``,
  ``delta`` back) are two-slot GM rings, each indexed by one chunk beat and guarded by its own mutex
  (``VcMutex`` MTE3 -> MTE2, ``CvMutex`` FIX -> MTE2). The A5 kernel's 31 hand-placed events are replaced by
  ``auto_sync``; pipesim checks the result.
* A2-01: the BF16 output chain ``qg' @ h`` then ``Aqk @ v_new`` (``is_init=False``) gets an explicit
  ``barrier(Pipe.M)``; the settle rule is not dtype-dependent on this path.
* ``q`` is read and ``o`` is written in the public token-major layout (``[B*T, H*128]`` / ``[B*T, HV*128]``).
  The two A5 value tiles of a head are separate work items here (``w``/``kg``/``Aqk`` are loaded per tile).
* The A5K-01 Aqk slot rotation concerned an L1 ring shared across heads; here ``Aqk`` is loaded per chunk into a
  cube-local buffer that the auto-synchronised program orders, so there is no rotating credit to get wrong.
"""

from ascriptor.a2 import *

L = 64

K_DIM = 128

V_DIM = 128

V_BLOCK = 64

V_TILES = V_DIM // V_BLOCK

HALF_L = L // 2

HALF_K = K_DIM // 2


@kernel()
def kda_sub45_a2_kernel(
    q: GM[bf16, ('BT', 'HK')],
    Aqk: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    kg: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    w: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    u: GM[bf16, ('B', 'HV', 'C', 64, 128)],
    eg: GM[f32, ('B', 'HV', 'C', 64, 128)],
    initial_state: GM[f32, ('B', 'HV', 128, 128)],
    o: GM[bf16, ('BT', 'HVV')],
    final_state: GM[f32, ('B', 'HV', 128, 128)],
    B: i32,
    H: i32,
    HV: i32,
    C: i32,
    scale: f32,
):
    h_ws = GMBuff(DT.bfloat16, [K_DIM, V_BLOCK], slots=2, name="h_ws")
    qg_ws = GMBuff(DT.bfloat16, [L, K_DIM], slots=2, name="qg_ws")
    vnew_ws = GMBuff(DT.bfloat16, [L, V_BLOCK], slots=2, name="vnew_ws")
    prod_ws = GMBuff(DT.float, [L, V_BLOCK], slots=2, name="prod_ws")
    delta_ws = GMBuff(DT.float, [K_DIM, V_BLOCK], slots=2, name="delta_ws")
    h_mx = VcMutex(0, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    qg_mx = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    vnew_mx = VcMutex(2, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    prod_mx = CvMutex(3, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
    delta_mx = CvMutex(4, depth=2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)

    l1_w = Tensor(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_kg = Tensor(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_aqk = Tensor(DT.bfloat16, [L, L], Position.L1)
    l1_h = Tensor(DT.bfloat16, [K_DIM, V_BLOCK], Position.L1)
    l1_qg = Tensor(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_vnew = Tensor(DT.bfloat16, [L, V_BLOCK], Position.L1)
    l0c_prod = Tensor(DT.float, [L, V_BLOCK], Position.L0C)
    l0c_delta = Tensor(DT.float, [K_DIM, V_BLOCK], Position.L0C)
    l0c_out = Tensor(DT.float, [L, V_BLOCK], Position.L0C)

    s_ub = Tensor(DT.float, [HALF_K, V_BLOCK], Position.UB)
    sdec_ub = Tensor(DT.float, [HALF_K, V_BLOCK], Position.UB)
    h_b_ub = Tensor(DT.bfloat16, [HALF_K, V_BLOCK], Position.UB)
    delta_ub = Tensor(DT.float, [HALF_K, V_BLOCK], Position.UB)
    glast_ub = Tensor(DT.float, [1, HALF_K], Position.UB)
    q_b_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    q_f_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    egq_ub = Tensor(DT.float, [HALF_L, K_DIM], Position.UB)
    qg_b_ub = Tensor(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    u_b_ub = Tensor(DT.bfloat16, [HALF_L, V_BLOCK], Position.UB)
    u_f_ub = Tensor(DT.float, [HALF_L, V_BLOCK], Position.UB)
    prod_ub = Tensor(DT.float, [HALF_L, V_BLOCK], Position.UB)
    vn_f_ub = Tensor(DT.float, [HALF_L, V_BLOCK], Position.UB)
    vn_b_ub = Tensor(DT.bfloat16, [HALF_L, V_BLOCK], Position.UB)

    work_count = B * HV * V_TILES
    work_per_cube = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_cube * GetCubeIdx())
    work_end = Min(work_begin + work_per_cube, work_count)
    group = Var(HV // H)
    step = Var(0)
    decay = Var(0.0, dtype=DT.float)

    with auto_sync():
        for work in range(work_begin, work_end):
            v_tile = Var(work % V_TILES)
            head = Var(work // V_TILES)
            hv_idx = Var(head % HV)
            b_idx = Var(head // HV)
            h_idx = Var(hv_idx // group)
            v0 = Var(v_tile * V_BLOCK)
            rk0 = Var(GetSubBlockIdx() * HALF_K)
            rl0 = Var(GetSubBlockIdx() * HALF_L)
            q_col = Var(h_idx * K_DIM)
            o_col = Var(hv_idx * V_DIM + v0)

            s_ub[0:HALF_K, 0:V_BLOCK] <<= initial_state[b_idx, hv_idx, rk0:rk0 + HALF_K, v0:v0 + V_BLOCK]

            for c_idx in range(0, C):
                tok0 = Var(b_idx * C * L + c_idx * L)

                # vector: h = bf16(S) for this sub-block's K rows
                cast(h_b_ub[0:HALF_K, 0:V_BLOCK], s_ub[0:HALF_K, 0:V_BLOCK], round_mode=RoundMode.TO_EVEN,
                     count=HALF_K * V_BLOCK)
                h_mx.lock()
                h_ws[step][rk0:rk0 + HALF_K, 0:V_BLOCK] <<= h_b_ub[0:HALF_K, 0:V_BLOCK]
                h_mx.ready()

                # vector: qg' = bf16(q * eg * scale) for this sub-block's L rows
                q_b_ub[0:HALF_L, 0:K_DIM] <<= q[tok0 + rl0:tok0 + rl0 + HALF_L, q_col:q_col + K_DIM]
                egq_ub[0:HALF_L, 0:K_DIM] <<= eg[b_idx, hv_idx, c_idx, rl0:rl0 + HALF_L, 0:K_DIM]
                cast(q_f_ub[0:HALF_L, 0:K_DIM], q_b_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.NONE, count=HALF_L * K_DIM)
                mul(q_f_ub[0:HALF_L, 0:K_DIM], q_f_ub[0:HALF_L, 0:K_DIM], egq_ub[0:HALF_L, 0:K_DIM], count=HALF_L * K_DIM)
                muls(q_f_ub[0:HALF_L, 0:K_DIM], q_f_ub[0:HALF_L, 0:K_DIM], scale, count=HALF_L * K_DIM)
                cast(qg_b_ub[0:HALF_L, 0:K_DIM], q_f_ub[0:HALF_L, 0:K_DIM], round_mode=RoundMode.TO_EVEN, count=HALF_L * K_DIM)
                qg_mx.lock()
                qg_ws[step][rl0:rl0 + HALF_L, 0:K_DIM] <<= qg_b_ub[0:HALF_L, 0:K_DIM]
                qg_mx.ready()

                # vector: S_dec = S * eg[63, k] per K row
                glast_ub[0:1, 0:HALF_K] <<= eg[b_idx, hv_idx, c_idx, L - 1:L, rk0:rk0 + HALF_K]
                with vec_scope():
                    for r in range(0, HALF_K):
                        decay.GetValueFrom(glast_ub[0:1, r:r + 1])
                        muls(sdec_ub[r:r + 1, 0:V_BLOCK], s_ub[r:r + 1, 0:V_BLOCK], decay, count=V_BLOCK)

                # cube: prod = w @ h
                l1_w <<= w[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
                l1_kg <<= kg[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
                l1_aqk <<= Aqk[b_idx, hv_idx, c_idx, 0:L, 0:L]
                h_mx.wait()
                l1_h <<= h_ws[step][0:K_DIM, 0:V_BLOCK]
                h_mx.free()
                matmul(l0c_prod, l1_w, l1_h.T, m=L, n=V_BLOCK, k=K_DIM)
                prod_mx.lock()
                prod_ws[step][0:L, 0:V_BLOCK] <<= l0c_prod
                prod_mx.ready()

                # vector: v_new = bf16(u - prod) for this sub-block's L rows
                u_b_ub[0:HALF_L, 0:V_BLOCK] <<= u[b_idx, hv_idx, c_idx, rl0:rl0 + HALF_L, v0:v0 + V_BLOCK]
                cast(u_f_ub[0:HALF_L, 0:V_BLOCK], u_b_ub[0:HALF_L, 0:V_BLOCK], round_mode=RoundMode.NONE, count=HALF_L * V_BLOCK)
                prod_mx.wait()
                prod_ub[0:HALF_L, 0:V_BLOCK] <<= prod_ws[step][rl0:rl0 + HALF_L, 0:V_BLOCK]
                prod_mx.free()
                sub(vn_f_ub[0:HALF_L, 0:V_BLOCK], u_f_ub[0:HALF_L, 0:V_BLOCK], prod_ub[0:HALF_L, 0:V_BLOCK], count=HALF_L * V_BLOCK)
                cast(vn_b_ub[0:HALF_L, 0:V_BLOCK], vn_f_ub[0:HALF_L, 0:V_BLOCK], round_mode=RoundMode.TO_EVEN, count=HALF_L * V_BLOCK)
                vnew_mx.lock()
                vnew_ws[step][rl0:rl0 + HALF_L, 0:V_BLOCK] <<= vn_b_ub[0:HALF_L, 0:V_BLOCK]
                vnew_mx.ready()

                # cube: delta = kg^T @ v_new ; o = qg' @ h + Aqk @ v_new
                vnew_mx.wait()
                l1_vnew <<= vnew_ws[step][0:L, 0:V_BLOCK]
                vnew_mx.free()
                matmul(l0c_delta, l1_kg.T, l1_vnew.T, m=K_DIM, n=V_BLOCK, k=L)
                delta_mx.lock()
                delta_ws[step][0:K_DIM, 0:V_BLOCK] <<= l0c_delta
                delta_mx.ready()
                qg_mx.wait()
                l1_qg <<= qg_ws[step][0:L, 0:K_DIM]
                qg_mx.free()
                matmul(l0c_out, l1_qg, l1_h.T, m=L, n=V_BLOCK, k=K_DIM)
                barrier(Pipe.M)
                matmul(l0c_out, l1_aqk, l1_vnew.T, m=L, n=V_BLOCK, k=L, is_init=False)
                o[tok0:tok0 + L, o_col:o_col + V_BLOCK] <<= l0c_out

                # vector: S = S_dec + delta
                delta_mx.wait()
                delta_ub[0:HALF_K, 0:V_BLOCK] <<= delta_ws[step][rk0:rk0 + HALF_K, 0:V_BLOCK]
                delta_mx.free()
                add(s_ub[0:HALF_K, 0:V_BLOCK], sdec_ub[0:HALF_K, 0:V_BLOCK], delta_ub[0:HALF_K, 0:V_BLOCK], count=HALF_K * V_BLOCK)
                step += 1

            final_state[b_idx, hv_idx, rk0:rk0 + HALF_K, v0:v0 + V_BLOCK] <<= s_ub[0:HALF_K, 0:V_BLOCK]

    return o, final_state
