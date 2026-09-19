"""FP32 domain/default guard for D-PM-37; original math kernels stay untouched."""
from ascriptor.a5 import *
from .guard import fill_row, fill_small, check_initial, check_center, check_row, check_span


@vf()
def prefix_row(g: Tensor, prefix: Tensor, compensation: Tensor):
    value = RegList(DT.float, 2)
    total = RegList(DT.float, 2)
    error = RegList(DT.float, 2)
    result = RegList(DT.float, 2)
    value <<= g[0:1, 0:128]
    total <<= prefix[0:1, 0:128]
    error <<= compensation[0:1, 0:128]
    value <<= value - error
    result <<= total + value
    error <<= result - total
    error <<= error - value
    prefix[0:1, 0:128] <<= result
    compensation[0:1, 0:128] <<= error
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pkda_fp32_guard(
    q: GM[f32, ("B", "T", "H", 128)], k: GM[f32, ("B", "T", "H", 128)],
    v: GM[f32, ("B", "T", "H", 128)], g: GM[f32, ("B", "T", "H", 128)],
    g_atk: GM[f32, ("B", "T", "H")], beta_atk: GM[f32, ("B", "T", "H")],
    beta: GM[f32, ("B", "T", "H")],
    initial_state: GM[f32, ("B", "H", 128, 128)],
    initial_A_state: GM[f32, ("B", "H", 128)],
    log_atk_scale: GM[f32, (1, "H")],
    state: GM[f32, ("B", "H", 128, 128)],
    astate: GM[f32, ("B", "H", 128)],
    center: GM[f32, (1, "H")],
    status: GM[f32, ("B", "H", 64)],
    B: i32, T: i32, H: i32, N: i32, has_state: i32, has_A: i32, has_center: i32,
):
    qu = Tensor(DT.float, [1, 128], Position.UB)
    ku = Tensor(DT.float, [1, 128], Position.UB)
    vu = Tensor(DT.float, [1, 128], Position.UB)
    gu = Tensor(DT.float, [1, 128], Position.UB)
    ga = Tensor(DT.float, [1, 8], Position.UB)
    ba = Tensor(DT.float, [1, 8], Position.UB)
    be = Tensor(DT.float, [1, 8], Position.UB)
    prefix = Tensor(DT.float, [1, 128], Position.UB)
    compensation = Tensor(DT.float, [1, 128], Position.UB)
    row = Tensor(DT.float, [1, 128], Position.UB)
    flags = Tensor(DT.float, [1, 64], Position.UB)
    cu = Tensor(DT.float, [1, 64], Position.UB)
    per = CeilDiv(B*H, GetVecNum())
    begin = Var(per*GetVecIdx())
    end = Min(begin+per, B*H)
    with auto_sync():
        for item in range(begin, end):
            bb = Var(item//H)
            hh = Var(item%H)
            fill_small(flags, 0.0)
            for rr in range(128):
                if has_state != 0:
                    row[:, :] <<= initial_state[bb, hh, rr:rr+1, :]
                    check_initial(row, flags, 0)
                else:
                    fill_row(row, 0.0)
                state[bb, hh, rr:rr+1, :] <<= row[:, :]
            if has_A != 0:
                row[:, :] <<= initial_A_state[bb, hh:hh+1, :]
                check_initial(row, flags, 1)
            else:
                fill_row(row, 0.0)
            astate[bb, hh:hh+1, :] <<= row[:, :]
            fill_small(cu, -0.2)
            if has_center != 0:
                cu[:, 0:1] <<= log_atk_scale[0:1, hh:hh+1]
                check_center(cu, flags)
            if bb == 0:
                center[0:1, hh:hh+1] <<= cu[:, 0:1]
            for cc in range(N):
                fill_row(prefix, 0.0)
                fill_row(compensation, 0.0)
                for ii in range(64):
                    tt = Var(cc*64+ii)
                    if tt < T:
                        qu[:, :] <<= q[bb, tt:tt+1, hh, :]
                        ku[:, :] <<= k[bb, tt:tt+1, hh, :]
                        vu[:, :] <<= v[bb, tt:tt+1, hh, :]
                        gu[:, :] <<= g[bb, tt:tt+1, hh, :]
                        ga[:, 0:1] <<= g_atk[bb, tt:tt+1, hh:hh+1]
                        ba[:, 0:1] <<= beta_atk[bb, tt:tt+1, hh:hh+1]
                        be[:, 0:1] <<= beta[bb, tt:tt+1, hh:hh+1]
                        check_row(qu, ku, vu, gu, ga, ba, be, flags)
                        prefix_row(gu, prefix, compensation)
                check_span(prefix, flags)
            status[bb, hh:hh+1, :] <<= flags[:, :]
    return state, astate, center, status

