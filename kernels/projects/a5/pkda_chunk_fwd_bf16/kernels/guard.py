"""Device-only defaults and numeric-domain checks; no host tensor arithmetic."""
from ascriptor.a5 import *


@vf()
def fill_row(out: Tensor, value: Var):
    r = RegList(DT.float, 2)
    r <<= value
    out[0:1, 0:128] <<= r


@vf()
def fill_small(out: Tensor, value: Var):
    r = Reg(DT.float)
    r <<= value
    out[0:1, 0:64] <<= r


@vf()
def widen_row(source: Tensor, out: Tensor):
    r = Reg(DT.float)
    for half in unroll(2):
        # Unpack consumes exactly64 BF16 elements, including the last half-row.
        r <<= source[0:1, half*64:half*64+64].unpack()
        out[0:1, half*64:half*64+64] <<= r


@vf()
def check_initial(source: Tensor, flags: Tensor, nonnegative: Var):
    r = Reg(DT.float)
    errors = Reg(DT.float)
    code = Reg(DT.float)
    zero = Reg(DT.float)
    tmp = Reg(DT.float)
    good = MaskReg(DT.float)
    lower = MaskReg(DT.float)
    zero <<= 0.0
    errors <<= flags[0:1, 0:64]
    for half in unroll(2):
        r <<= source[0:1, half*64:half*64+64]
        good <<= r <= 3.4028234663852886e38
        lower <<= r >= -3.4028234663852886e38
        good <<= good & lower
        code <<= 1.0
        tmp <<= good.select(zero, code)
        errors <<= errors.vmax(tmp)
        if nonnegative != 0:
            good <<= r >= 0.0
            code <<= 6.0
            tmp <<= good.select(zero, code)
            errors <<= errors.vmax(tmp)
    flags[0:1, 0:64] <<= errors
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def check_center(source: Tensor, flags: Tensor):
    r = Reg(DT.float)
    errors = Reg(DT.float)
    one = Reg(DT.float)
    zero = Reg(DT.float)
    tmp = Reg(DT.float)
    good = MaskReg(DT.float)
    lower = MaskReg(DT.float)
    r <<= source[0:1, 0:1].single()
    errors <<= flags[0:1, 0:64]
    zero <<= 0.0
    one <<= 1.0
    good <<= r <= 3.4028234663852886e38
    lower <<= r >= -3.4028234663852886e38
    good <<= good & lower
    tmp <<= good.select(zero, one)
    errors <<= errors.vmax(tmp)
    flags[0:1, 0:64] <<= errors
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pkda_bf16_init(
    initial_state: GM[f32, ("B", "H", 128, 128)],
    initial_A_state: GM[f32, ("B", "H", 128)],
    log_atk_scale: GM[f32, (1, "H")],
    state: GM[f32, ("B", "H", 128, 128)],
    astate: GM[f32, ("B", "H", 128)],
    center: GM[f32, ("B", "H", 8)],
    status: GM[f32, ("B", "H", 64)],
    B: i32, H: i32, has_state: i32, has_A: i32, has_center: i32,
):
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
            center[bb, hh:hh+1, :] <<= cu[:, 0:8]
            status[bb, hh:hh+1, :] <<= flags[:, :]
    return state, astate, center, status


@vf()
def check_span(prefix: Tensor, flags: Tensor):
    r = Reg(DT.float)
    errors = Reg(DT.float)
    code = Reg(DT.float)
    zero = Reg(DT.float)
    tmp = Reg(DT.float)
    good = MaskReg(DT.float)
    zero <<= 0.0
    code <<= 8.0
    errors <<= flags[0:1, 0:64]
    for half in unroll(2):
        r <<= prefix[0:1, half*64:half*64+64]
        good <<= r >= -155.0
        tmp <<= good.select(zero, code)
        errors <<= errors.vmax(tmp)
    flags[0:1, 0:64] <<= errors
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def check_row(q: Tensor, k: Tensor, v: Tensor, g: Tensor, ga: Tensor,
              ba: Tensor, beta: Tensor, flags: Tensor):
    r = Reg(DT.float)
    errors = Reg(DT.float)
    code = Reg(DT.float)
    zero = Reg(DT.float)
    tmp = Reg(DT.float)
    norm = Reg(DT.float)
    dot = Reg(DT.float)
    good = MaskReg(DT.float)
    lower = MaskReg(DT.float)
    zero <<= 0.0
    norm <<= 0.0
    errors <<= flags[0:1, 0:64]
    for half in unroll(2):
        r <<= q[0:1, half*64:half*64+64]
        good <<= r <= 3.4028234663852886e+38
        lower <<= r >= -3.4028234663852886e+38
        good <<= good & lower
        code <<= 1.0
        tmp <<= good.select(zero, code)
        errors <<= errors.vmax(tmp)
    for half in unroll(2):
        r <<= k[0:1, half*64:half*64+64]
        good <<= r <= 3.4028234663852886e+38
        lower <<= r >= -3.4028234663852886e+38
        good <<= good & lower
        code <<= 1.0
        tmp <<= good.select(zero, code)
        errors <<= errors.vmax(tmp)
        tmp <<= r * r
        norm <<= norm + tmp
    for half in unroll(2):
        r <<= v[0:1, half*64:half*64+64]
        good <<= r <= 3.4028234663852886e+38
        lower <<= r >= -3.4028234663852886e+38
        good <<= good & lower
        code <<= 1.0
        tmp <<= good.select(zero, code)
        errors <<= errors.vmax(tmp)
    for half in unroll(2):
        r <<= g[0:1, half*64:half*64+64]
        good <<= r <= 0.0
        lower <<= r >= -3.4028234663852886e+38
        good <<= good & lower
        code <<= 2.0
        tmp <<= good.select(zero, code)
        errors <<= errors.vmax(tmp)
    r <<= ga[0:1, 0:1].single()
    good <<= r <= 0.0
    lower <<= r >= -3.4028234663852886e+38
    good <<= good & lower
    code <<= 3.0
    tmp <<= good.select(zero, code)
    errors <<= errors.vmax(tmp)
    r <<= beta[0:1, 0:1].single()
    good <<= r <= 1.0
    lower <<= r >= 0.0
    good <<= good & lower
    code <<= 4.0
    tmp <<= good.select(zero, code)
    errors <<= errors.vmax(tmp)
    r <<= ba[0:1, 0:1].single()
    good <<= r <= 1.0
    lower <<= r >= 0.0
    good <<= good & lower
    code <<= 5.0
    tmp <<= good.select(zero, code)
    errors <<= errors.vmax(tmp)
    cadd(dot, norm)
    # Broadcast leading reduction result through UB before the norm guard.
    flags[0:1, 0:1] <<= dot.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    dot <<= flags[0:1, 0:1].single()
    good <<= dot <= 1.0000200001
    code <<= 7.0
    tmp <<= good.select(zero, code)
    errors <<= errors.vmax(tmp)
    flags[0:1, 0:64] <<= errors
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
