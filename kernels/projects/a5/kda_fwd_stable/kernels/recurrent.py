"""A5K-01 Aqk handoff repair, derived from accepted kernels b3b3f9c.

Both Aqk accesses rotate by local publication count, including head boundaries.
Two outstanding event credits now correspond to two distinct physical slots.
All arithmetic, other slots, event calls and precision boundaries are unchanged.
The original kernel remains a read-only baseline in the accepted dependency.
"""
from ascriptor.a5 import *

L = 64

K_DIM = 128

V_DIM = 128

V_BLOCK = 64

V_TILES = V_DIM // V_BLOCK

HALF_L = L // 2

HALF_K = K_DIM // 2

K_BLOCK = 64

K_TILES = K_DIM // K_BLOCK

@vf()
def cast_state_to_h_vf(state_ub: Tensor, h_ub: Tensor, rows: Var):
    state_row = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(V_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        state_row <<= state_ub[r:r + 1, 0:V_BLOCK]
        row_bf16 <<= state_row.astype(DT.bfloat16)
        reg_to_ub_downsample(h_ub[r:r + 1, 0:V_BLOCK], row_bf16, mask=row_mask_bf16)

@vf()
def make_vnew_vf(prod_ub: Tensor, u_ub: Tensor, vnew_ub: Tensor, rows: Var):
    prod_row = Reg(DT.float)
    u_row = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(V_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        prod_row <<= prod_ub[r:r + 1, 0:V_BLOCK]
        u_row <<= u_ub[r:r + 1, 0:V_BLOCK]
        prod_row <<= u_row - prod_row
        row_bf16 <<= prod_row.astype(DT.bfloat16)
        reg_to_ub_downsample(vnew_ub[r:r + 1, 0:V_BLOCK], row_bf16, mask=row_mask_bf16)

@vf()
def decay_state_vf(state_ub: Tensor, g_last_ub: Tensor, state_decayed_ub: Tensor, rows: Var):
    state_row = Reg(DT.float)
    decay = Reg(DT.float)

    for r in range(rows):
        state_row <<= state_ub[r:r + 1, 0:V_BLOCK]
        decay <<= g_last_ub[0:1, r:r + 1].single()
        state_row <<= state_row * decay
        state_decayed_ub[r:r + 1, 0:V_BLOCK] <<= state_row

@vf()
def add_delta_to_state_vf(state_decayed_ub: Tensor, delta_ub: Tensor, state_ub: Tensor, rows: Var):
    state_row = Reg(DT.float)
    delta_row = Reg(DT.float)

    for r in range(rows):
        state_row <<= state_decayed_ub[r:r + 1, 0:V_BLOCK]
        delta_row <<= delta_ub[r:r + 1, 0:V_BLOCK]
        state_row <<= state_row + delta_row
        state_ub[r:r + 1, 0:V_BLOCK] <<= state_row

@vf()
def qg_scale_vf(q_ub: Tensor, eg_ub: Tensor, qg_ub: Tensor, rows: Var, scale: Var):
    q_row = Reg(DT.float)
    eg_row = Reg(DT.float)
    tmp = Reg(DT.float)
    row_bf16 = Reg(DT.bfloat16)
    row_mask_bf16 = MaskReg(DT.bfloat16, init_mode=MaskType.NONE)

    row_mask_bf16 <<= Var(K_BLOCK * 2, dtype=DT.uint32)

    for r in range(rows):
        for tile in range(K_TILES):
            k_begin = tile * K_BLOCK
            k_end = k_begin + K_BLOCK
            q_row <<= q_ub[r:r + 1, k_begin:k_end]
            eg_row <<= eg_ub[r:r + 1, k_begin:k_end]
            tmp <<= q_row * eg_row
            tmp <<= tmp * scale
            row_bf16 <<= tmp.astype(DT.bfloat16)
            reg_to_ub_downsample(qg_ub[r:r + 1, k_begin:k_end], row_bf16, mask=row_mask_bf16)

@kernel()
def kda_sub45_aqk_repaired_kernel(q: GM[bf16, ('B', 'H', 'C', 64, 128)], Aqk: GM[bf16, ('B', 'HV', 'C', 64, 64)], kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], w: GM[bf16, ('B', 'HV', 'C', 64, 128)], u: GM[bf16, ('B', 'HV', 'C', 64, 128)], eg: GM[f32, ('B', 'HV', 'C', 64, 128)], initial_state: GM[f32, ('B', 'HV', 128, 128)], o: GM[bf16, ('B', 'HV', 'C', 64, 128)], final_state: GM[f32, ('B', 'HV', 128, 128)], B: i32, H: i32, HV: i32, C: i32, length_per_chunk: i32, head_dim: i32, value_dim: i32, scale: f32):
    prod_mutex = CvMutex(0, depth=2, src_start_pipe=Pipe.FIX, src_end_pipe=Pipe.FIX, dst_start_pipe=Pipe.V, dst_end_pipe=Pipe.V)
    vnew_mutex = VcMutex(1, depth=2, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)
    state_mutex = VcMutex(2, depth=2, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)
    delta_mutex = CvMutex(3, depth=2, src_start_pipe=Pipe.FIX, src_end_pipe=Pipe.FIX, dst_start_pipe=Pipe.V, dst_end_pipe=Pipe.V)
    qg_mutex = VcMutex(4, depth=2, src_start_pipe=Pipe.MTE3, src_end_pipe=Pipe.MTE3, dst_start_pipe=Pipe.MTE1, dst_end_pipe=Pipe.MTE1)

    state_ubout_valid = DEvent(Pipe.MTE3, Pipe.MTE2, preset=True)
    state_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    state_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)

    h_ubout_valid = DEvent(Pipe.MTE3, Pipe.V, preset=True)
    h_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)
    vnew_ubout_valid = DEvent(Pipe.MTE3, Pipe.V, preset=True)
    vnew_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)

    u_ubin_valid = DEvent(Pipe.V, Pipe.MTE2, preset=True)
    u_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    g_ubin_valid = DEvent(Pipe.V, Pipe.MTE2, preset=True)
    g_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    q_ubin_valid = DEvent(Pipe.V, Pipe.MTE2, preset=True)
    q_ubin_ready = DEvent(Pipe.MTE2, Pipe.V)
    qg_ubout_valid = DEvent(Pipe.MTE3, Pipe.V, preset=True)
    qg_ubout_ready = DEvent(Pipe.V, Pipe.MTE3)

    w_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)
    w_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)
    kg_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)
    kg_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)
    aqk_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)
    aqk_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)

    prod_l0_valid = DEvent(Pipe.M, Pipe.MTE1, preset=True)
    prod_l0_ready = DEvent(Pipe.MTE1, Pipe.M)
    prod_l0c_valid = DEvent(Pipe.FIX, Pipe.M, preset=True)
    prod_l0c_ready = DEvent(Pipe.M, Pipe.FIX)
    delta_l0_valid = DEvent(Pipe.M, Pipe.MTE1, preset=True)
    delta_l0_ready = DEvent(Pipe.MTE1, Pipe.M)
    delta_l0c_valid = DEvent(Pipe.FIX, Pipe.M, preset=True)
    delta_l0c_ready = DEvent(Pipe.M, Pipe.FIX)
    out_l0c_valid = DEvent(Pipe.FIX, Pipe.M, preset=True)
    out_l0c_ready = DEvent(Pipe.M, Pipe.FIX)

    l1_state = DBuff(DT.bfloat16, [K_DIM, V_BLOCK], Position.L1)
    l1_qg = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_Aqk = DBuff(DT.bfloat16, [L, L], Position.L1)
    l1_w = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_kg = DBuff(DT.bfloat16, [L, K_DIM], Position.L1)
    l1_vnew = DBuff(DT.bfloat16, [L, V_BLOCK], Position.L1)
    l0a_prod = DBuff(DT.bfloat16, [L, K_DIM], Position.L0A)
    l0b_prod = DBuff(DT.bfloat16, [V_BLOCK, K_DIM], Position.L0B)
    l0a_delta = DBuff(DT.bfloat16, [K_DIM, L], Position.L0A)
    l0b_delta = DBuff(DT.bfloat16, [V_BLOCK, L], Position.L0B)
    l0c_prod = DBuff(DT.float, [L, V_BLOCK], Position.L0C)
    l0c_delta = DBuff(DT.float, [K_DIM, V_BLOCK], Position.L0C)
    l0c_out = DBuff(DT.float, [L, V_BLOCK], Position.L0C)

    state_ub = DBuff(DT.float, [HALF_K, V_BLOCK], Position.UB)
    h_ub = DBuff(DT.bfloat16, [HALF_K, V_BLOCK], Position.UB)
    prod_ub = DBuff(DT.float, [HALF_L, V_BLOCK], Position.UB)
    u_ub = DBuff(DT.bfloat16, [HALF_L, V_BLOCK], Position.UB)
    vnew_ub = DBuff(DT.bfloat16, [HALF_L, V_BLOCK], Position.UB)
    delta_ub = DBuff(DT.float, [HALF_K, V_BLOCK], Position.UB)
    state_decayed_ub = DBuff(DT.float, [HALF_K, V_BLOCK], Position.UB)
    g_last_ub = DBuff(DT.float, [1, HALF_K], Position.UB)
    q_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)
    eg_q_ub = DBuff(DT.float, [HALF_L, K_DIM], Position.UB)
    qg_ub = DBuff(DT.bfloat16, [HALF_L, K_DIM], Position.UB)

    pair_count = B * HV
    work_count = pair_count * V_TILES
    pair_begin = Var((pair_count * GetCubeIdx()) // GetCubeNum())
    pair_end = Var((pair_count * (GetCubeIdx() + 1)) // GetCubeNum())
    group = Var(HV // H)

    # auto sync is not used here because the nested for loops interfere with it,
    # causing auto sync to not output the correct event pairs
    for pair_idx in range(pair_begin, pair_end):
        pair = Var(pair_idx * V_TILES)
        row_begin_l = Var(GetSubBlockIdx() * HALF_L)
        row_end_l = Var(row_begin_l + HALF_L)
        row_begin_k = Var(GetSubBlockIdx() * HALF_K)
        row_end_k = Var(row_begin_k + HALF_K)

        for slot in range(2):
            work = Var(pair + slot)
            v_tile = Var(work % V_TILES)
            head_work = Var(work // V_TILES)
            hv_idx = Var(head_work % HV)
            b_idx = Var(head_work // HV)
            v_begin = Var(v_tile * V_BLOCK)
            v_end = Var(v_begin + V_BLOCK)
            state_ubout_valid.wait()
            state_ub[slot][0:HALF_K, 0:V_BLOCK] <<= initial_state[
                b_idx, hv_idx, row_begin_k:row_end_k, v_begin:v_end
            ]
            state_ubin_ready.set()
            state_ubin_ready.wait()

        for c_idx in range(C):
            # Vec stage A: publish current state as h and as bf16 L1.
            for slot in range(2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                h_ubout_valid.wait()
                cast_state_to_h_vf(state_ub[slot], h_ub[slot], HALF_K)
                h_ubout_ready.set()
                h_ubout_ready.wait()
                state_mutex.lock()
                l1_state[slot][row_begin_k:row_end_k, 0:V_BLOCK] <<= h_ub[slot][
                    0:HALF_K, 0:V_BLOCK
                ]
                state_mutex.ready()
                h_ubout_valid.set()

            # Vec stage A1: build q * eg * scale once per head/chunk for both value tiles.
            work_q = Var(pair)
            head_work_q = Var(work_q // V_TILES)
            hv_q = Var(head_work_q % HV)
            b_q = Var(head_work_q // HV)
            h_q = Var(hv_q // group)
            q_slot = Var(c_idx % 2)
            q_ubin_valid.wait()
            q_ub[q_slot][0:HALF_L, 0:K_DIM] <<= q[b_q, h_q, c_idx, row_begin_l:row_end_l, 0:K_DIM]
            eg_q_ub[q_slot][0:HALF_L, 0:K_DIM] <<= eg[b_q, hv_q, c_idx, row_begin_l:row_end_l, 0:K_DIM]
            q_ubin_ready.set()
            q_ubin_ready.wait()
            qg_ubout_valid.wait()
            qg_scale_vf(q_ub[q_slot], eg_q_ub[q_slot], qg_ub[q_slot], HALF_L, scale)
            q_ubin_valid.set()
            qg_ubout_ready.set()
            qg_ubout_ready.wait()
            qg_mutex.lock()
            l1_qg[q_slot][row_begin_l:row_end_l, 0:K_DIM] <<= qg_ub[q_slot][0:HALF_L, 0:K_DIM]
            qg_mutex.ready()
            qg_ubout_valid.set()

            # Cube-side input for the local output product, shared by both value tiles.
            work_aqk = Var(pair)
            head_work_aqk = Var(work_aqk // V_TILES)
            hv_aqk = Var(head_work_aqk % HV)
            b_aqk = Var(head_work_aqk // HV)
            aqk_slot = Var(((pair_idx - pair_begin) * C + c_idx) % 2)
            aqk_l1_valid.wait()
            l1_Aqk[aqk_slot][0:L, 0:L] <<= Aqk[b_aqk, hv_aqk, c_idx, 0:L, 0:L]
            aqk_l1_ready.set()

            # Vec stage A2: use the first w @ state wait gap for slot 0 decay.
            for slot in range(1):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                g_ubin_valid.wait()
                g_last_ub[slot][0:1, 0:HALF_K] <<= eg[
                    b_idx, hv_idx, c_idx, L - 1:L, row_begin_k:row_end_k
                ]
                g_ubin_ready.set()
                g_ubin_ready.wait()
                decay_state_vf(state_ub[slot], g_last_ub[slot], state_decayed_ub[slot], HALF_K)
                g_ubin_valid.set()

            # Cube stage A: w @ state.
            for slot in range(2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                w_l1_valid.wait()
                l1_w[slot][0:L, 0:K_DIM] <<= w[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
                w_l1_ready.set()
                state_mutex.wait()
                w_l1_ready.wait()
                prod_l0_valid.wait()
                l0a_prod[slot][0:L, 0:K_DIM] <<= l1_w[slot][0:L, 0:K_DIM]
                l0b_prod[slot][0:V_BLOCK, 0:K_DIM] <<= l1_state[slot].T
                w_l1_valid.set()
                prod_l0_ready.set()
                prod_l0_ready.wait()
                prod_l0c_valid.wait()
                # Keep the explicit cube pipe sequence visible because sync is manual here.
                mmad(l0c_prod[slot], l0a_prod[slot], l0b_prod[slot], M=L, N=V_BLOCK, K=K_DIM, is_init=True)
                prod_l0_valid.set()
                prod_l0c_ready.set()
                prod_mutex.lock()
                prod_l0c_ready.wait()
                prod_ub[slot] <<= l0c_prod[slot]
                prod_l0c_valid.set()
                prod_mutex.ready()

            # Vec stage B: u - w @ state, then publish v_new to L1.
            for slot in range(2):
                work = Var(pair + slot)
                v_tile = Var(work % V_TILES)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                v_begin = Var(v_tile * V_BLOCK)
                v_end = Var(v_begin + V_BLOCK)
                u_ubin_valid.wait()
                u_ub[slot][0:HALF_L, 0:V_BLOCK] <<= u[
                    b_idx, hv_idx, c_idx, row_begin_l:row_end_l, v_begin:v_end
                ]
                u_ubin_ready.set()
                prod_mutex.wait()
                u_ubin_ready.wait()
                vnew_ubout_valid.wait()
                make_vnew_vf(prod_ub[slot], u_ub[slot], vnew_ub[slot], HALF_L)
                u_ubin_valid.set()
                prod_mutex.free()
                vnew_ubout_ready.set()
                vnew_ubout_ready.wait()
                vnew_mutex.lock()
                l1_vnew[slot][row_begin_l:row_end_l, 0:V_BLOCK] <<= vnew_ub[slot][
                    0:HALF_L, 0:V_BLOCK
                ]
                vnew_mutex.ready()
                vnew_ubout_valid.set()

            # Vec stage B2: use the delta wait gap for slot 1 decay.
            for slot in range(1, 2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                g_ubin_valid.wait()
                g_last_ub[slot][0:1, 0:HALF_K] <<= eg[
                    b_idx, hv_idx, c_idx, L - 1:L, row_begin_k:row_end_k
                ]
                g_ubin_ready.set()
                g_ubin_ready.wait()
                decay_state_vf(state_ub[slot], g_last_ub[slot], state_decayed_ub[slot], HALF_K)
                g_ubin_valid.set()

            # Cube stage B: kg.T @ v_new.
            for slot in range(2):
                work = Var(pair + slot)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                kg_l1_valid.wait()
                l1_kg[slot][0:L, 0:K_DIM] <<= kg[b_idx, hv_idx, c_idx, 0:L, 0:K_DIM]
                kg_l1_ready.set()
                vnew_mutex.wait()
                kg_l1_ready.wait()
                delta_l0_valid.wait()
                l0a_delta[slot][0:K_DIM, 0:L] <<= l1_kg[slot].T
                l0b_delta[slot][0:V_BLOCK, 0:L] <<= l1_vnew[slot].T
                kg_l1_valid.set()
                vnew_mutex.free()
                delta_l0_ready.set()
                delta_l0_ready.wait()
                delta_l0c_valid.wait()
                mmad(
                    l0c_delta[slot],
                    l0a_delta[slot],
                    l0b_delta[slot],
                    M=K_DIM,
                    N=V_BLOCK,
                    K=L,
                    is_init=True,
                )
                delta_l0_valid.set()
                delta_l0c_ready.set()
                delta_mutex.lock()
                delta_l0c_ready.wait()
                delta_ub[slot] <<= l0c_delta[slot]
                delta_l0c_valid.set()
                delta_mutex.ready()

            # Cube stage C: assemble the sub5 output while h and v_new are still on chip.
            work_out = Var(pair)
            aqk_slot = Var(((pair_idx - pair_begin) * C + c_idx) % 2)
            qg_slot = Var(c_idx % 2)
            aqk_l1_ready.wait()
            qg_mutex.wait()

            for slot in range(2):
                work = Var(pair + slot)
                v_tile = Var(work % V_TILES)
                head_work = Var(work // V_TILES)
                hv_idx = Var(head_work % HV)
                b_idx = Var(head_work // HV)
                v_begin = Var(v_tile * V_BLOCK)
                v_end = Var(v_begin + V_BLOCK)

                prod_l0_valid.wait()
                l0a_prod[slot][0:L, 0:K_DIM] <<= l1_qg[qg_slot][0:L, 0:K_DIM]
                l0b_prod[slot][0:V_BLOCK, 0:K_DIM] <<= l1_state[slot].T
                state_mutex.free()
                prod_l0_ready.set()
                prod_l0_ready.wait()
                out_l0c_valid.wait()
                # Reuse the prod L0A/L0B buffers for qg @ h.
                mmad(l0c_out[slot], l0a_prod[slot], l0b_prod[slot], M=L, N=V_BLOCK, K=K_DIM, is_init=True)
                prod_l0_valid.set()
                delta_l0_valid.wait()
                l0a_delta[slot][0:L, 0:L] <<= l1_Aqk[aqk_slot][0:L, 0:L]
                delta_l0_ready.set()
                delta_l0_ready.wait()
                mmad(l0c_out[slot], l0a_delta[slot][0:L, 0:L], l0b_delta[slot], M=L, N=V_BLOCK, K=L, is_init=False)
                delta_l0_valid.set()
                out_l0c_ready.set()
                out_l0c_ready.wait()
                o[b_idx, hv_idx, c_idx, 0:L, v_begin:v_end] <<= l0c_out[slot][0:L, 0:V_BLOCK]
                out_l0c_valid.set()

            qg_mutex.free()
            aqk_l1_valid.set()

            # Vec stage C: apply recurrent state update.
            for slot in range(2):
                work = Var(pair + slot)
                delta_mutex.wait()
                add_delta_to_state_vf(state_decayed_ub[slot], delta_ub[slot], state_ub[slot], HALF_K)
                delta_mutex.free()

        for slot in range(2):
            work = Var(pair + slot)
            v_tile = Var(work % V_TILES)
            head_work = Var(work // V_TILES)
            hv_idx = Var(head_work % HV)
            b_idx = Var(head_work // HV)
            v_begin = Var(v_tile * V_BLOCK)
            v_end = Var(v_begin + V_BLOCK)
            state_ubout_ready.set()
            state_ubout_ready.wait()
            final_state[b_idx, hv_idx, row_begin_k:row_end_k, v_begin:v_end] <<= state_ub[
                slot
            ][0:HALF_K, 0:V_BLOCK]
            state_ubout_valid.set()

    return o, final_state
