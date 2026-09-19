"""Resource-bounded KDA inverse_mm derivative, A5K-02 / D-PM-24.

Source: a5.kda_bwd/kernels/inverse_mm.py at kernels b3b3f9c16df.
Only l0c_dvh and l0c_dvbeta become single-slot allocations. Their M writes
and last FIX reads retain mode-zero local ownership, including loop reuse.
All arithmetic, event credits, cross-side buffers and two-work lookahead stay
unchanged. The public cached-forward prerequisite is specified in the adjacent
kda_fwd_stable/contract.json; upstream backward ABI and precision budgets apply.

The arithmetic and schedule come from the reviewed source; only imports and
typed kernel signatures are migrated here.
M10 hardware repair: use integer division for the two static VF trip counts.
Float bounds made the vendor compiler assert before producing a vector object.
"""

from ascriptor.a5 import *

L = 64

D = 128

HALF_L = L // 2

@vf()
def negate_cast_dw_vf(dvh_f_ub: Tensor, dw_h_ub: Tensor):
    reg_f32 = Reg(DT.float)
    n_loops = Var(HALF_L * D // 64)
    for i in range(n_loops):
        reg_f32 <<= dvh_f_ub[i * 64]
        reg_f32 <<= reg_f32 * (-1.0)
        dw_h_ub[i * 64] <<= reg_f32
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@vf()
def cast_out_vf(out_f_ub: Tensor, out_h_ub: Tensor):
    reg_f32 = Reg(DT.float)
    n_loops = Var(HALF_L * D // 64)
    for i in range(n_loops):
        reg_f32 <<= out_f_ub[i * 64]
        out_h_ub[i * 64] <<= reg_f32
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

@kernel()
def inverse_mm_bounded_kernel(do_bf16: GM[bf16, ('B', 'T', 'HV', 128)], vnew_bf16: GM[bf16, ('B', 'T', 'HV', 128)], dv_bf16: GM[bf16, ('B', 'T', 'HV', 128)], h_bf16: GM[bf16, ('B', 'C', 'HV', 128, 128)], dh_bf16: GM[bf16, ('B', 'C', 'HV', 128, 128)], Akk_bf16: GM[bf16, ('B', 'T', 'HV', 64)], d_qg: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_kg: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_vh: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_v_beta: GM[bf16, ('B', 'HV', 'C', 64, 128)], d_k_beta_g: GM[bf16, ('B', 'HV', 'C', 64, 128)], B: i32, HV: i32, C: i32):
    dvh_mutex = CvMutex(0, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    dw_bridge = VcMutex(
        1,
        depth=3,
        src_start_pipe=Pipe.MTE3,
        src_end_pipe=Pipe.MTE3,
        dst_start_pipe=Pipe.MTE1,
        dst_end_pipe=Pipe.MTE1,
    )
    out_mutex = CvMutex(2, depth=3, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE3)

    l1_do = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_vnew = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_dv = DBuff(DT.bfloat16, [L, D], Position.L1)
    l1_h = DBuff(DT.bfloat16, [D, D], Position.L1)
    l1_dh = DBuff(DT.bfloat16, [D, D], Position.L1)
    l1_Akk = TBuff(DT.bfloat16, [L, L], Position.L1)
    l1_dw = TBuff(DT.bfloat16, [L, D], Position.L1)

    l0c_dqg = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dkg = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dvh = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dvbeta = Tensor(DT.float, [L, D], Position.L0C)
    l0c_dkbetag = DBuff(DT.float, [L, D], Position.L0C)

    dvh_f_ub = TBuff(DT.float, [HALF_L, D], Position.UB)
    dw_h_ub = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)
    out_f_ub = TBuff(DT.float, [HALF_L, D], Position.UB)
    out_h_ub = TBuff(DT.bfloat16, [HALF_L, D], Position.UB)

    work_count = Var(B * HV * C)
    work_per_core = CeilDiv(work_count, GetCubeNum())
    work_begin = Var(work_per_core * GetCubeIdx())
    work_end = Min(work_begin + work_per_core, work_count)
    row_begin_l = Var(GetSubBlockIdx() * HALF_L)
    row_end_l = Var(row_begin_l + HALF_L)

    with auto_sync():
        for pipe_work in range(work_begin, work_end + 2):
            if pipe_work < work_end:
                c_idx = Var(pipe_work % C)
                bhv_idx = Var(pipe_work / C)
                hv_idx = Var(bhv_idx % HV)
                b_idx = Var(bhv_idx / HV)
                row0 = Var(c_idx * L)
                row1 = Var(row0 + L)

                # Strided BTHVD/BTHVL reads from the frozen public layout: explicit
                # N_src = HV*D / HV*L pitch (the <<= auto-infer would use D/L and
                # silently read compacted adjacent tokens -- wrong data, no crash).
                # h/dh stay contiguous [D, D] blocks (frozen [B, C, HV, D, D]); only
                # their index order changes, so the <<= auto-infer (N_src=D) is right.
                gm_to_l1_nd2nz(l1_dv[pipe_work][0:L, 0:D], dv_bf16[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
                l1_h[pipe_work][0:D, 0:D] <<= h_bf16[b_idx, c_idx, hv_idx, 0:D, 0:D]
                gm_to_l1_nd2nz(l1_do[pipe_work][0:L, 0:D], do_bf16[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
                gm_to_l1_nd2nz(l1_vnew[pipe_work][0:L, 0:D], vnew_bf16[b_idx, row0:row1, hv_idx, 0:D], L, D, HV * D, L)
                l1_dh[pipe_work][0:D, 0:D] <<= dh_bf16[b_idx, c_idx, hv_idx, 0:D, 0:D]
                gm_to_l1_nd2nz(l1_Akk[pipe_work][0:L, 0:L], Akk_bf16[b_idx, row0:row1, hv_idx, 0:L], L, L, HV * L, L)

                # The dvh matmul leads: it gates the FIX -> UB -> L1 d_w chain.
                matmul(l0c_dvh, l1_dv[pipe_work], l1_h[pipe_work], m=L, n=D, k=D, splitn=D)
                matmul(l0c_dqg, l1_do[pipe_work], l1_h[pipe_work], m=L, n=D, k=D, splitn=D)
                matmul(l0c_dvbeta, l1_Akk[pipe_work].T, l1_dv[pipe_work].T, m=L, n=D, k=L, splitn=D)
                matmul(l0c_dkg, l1_vnew[pipe_work], l1_dh[pipe_work], m=L, n=D, k=D, splitn=D)

                d_vh[b_idx, hv_idx, c_idx, 0:L, 0:D] <<= l0c_dvh

                dvh_mutex.lock()
                l0c_to_ub(dvh_f_ub[pipe_work], l0c_dvh, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                dvh_mutex.ready()
                dvh_mutex.wait()
                negate_cast_dw_vf(dvh_f_ub[pipe_work], dw_h_ub[pipe_work])
                dvh_mutex.free()

                dw_bridge.lock()
                l1_dw[pipe_work][row_begin_l:row_end_l, 0:D] <<= dw_h_ub[pipe_work][0:HALF_L, 0:D]
                dw_bridge.ready()

                # dqg / dvbeta / dkg leave through the vector side: FIX is L0C
                # read bound, so l0c_to_ub (138cy) + MTE3 store beats five
                # l0c_to_gm stores (1324cy each).
                out_mutex.lock()
                l0c_to_ub(out_f_ub[3 * pipe_work], l0c_dqg, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                out_mutex.ready()
                out_mutex.wait()
                cast_out_vf(out_f_ub[3 * pipe_work], out_h_ub[3 * pipe_work])
                d_qg[b_idx, hv_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_h_ub[3 * pipe_work][0:HALF_L, 0:D]
                out_mutex.free()

                out_mutex.lock()
                l0c_to_ub(out_f_ub[3 * pipe_work + 1], l0c_dvbeta, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                out_mutex.ready()
                out_mutex.wait()
                cast_out_vf(out_f_ub[3 * pipe_work + 1], out_h_ub[3 * pipe_work + 1])
                d_v_beta[b_idx, hv_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_h_ub[3 * pipe_work + 1][0:HALF_L, 0:D]
                out_mutex.free()

                out_mutex.lock()
                l0c_to_ub(out_f_ub[3 * pipe_work + 2], l0c_dkg, M=L, N=D, N_dst=D, M_src=L, dual_mode=DualMode.SPLITM, sub_block_id=0)
                out_mutex.ready()
                out_mutex.wait()
                cast_out_vf(out_f_ub[3 * pipe_work + 2], out_h_ub[3 * pipe_work + 2])
                d_kg[b_idx, hv_idx, c_idx, row_begin_l:row_end_l, 0:D] <<= out_h_ub[3 * pipe_work + 2][0:HALF_L, 0:D]
                out_mutex.free()

            if pipe_work >= work_begin + 2:
                prev_work = Var(pipe_work - 2)
                prev_c = Var(prev_work % C)
                prev_bhv = Var(prev_work / C)
                prev_hv = Var(prev_bhv % HV)
                prev_b = Var(prev_bhv / HV)

                dw_bridge.wait()
                matmul(l0c_dkbetag[prev_work], l1_Akk[prev_work].T, l1_dw[prev_work].T, m=L, n=D, k=L, splitn=D)
                dw_bridge.free()

                d_k_beta_g[prev_b, prev_hv, prev_c, 0:L, 0:D] <<= l0c_dkbetag[prev_work]

    return d_qg, d_kg, d_vh, d_v_beta, d_k_beta_g
