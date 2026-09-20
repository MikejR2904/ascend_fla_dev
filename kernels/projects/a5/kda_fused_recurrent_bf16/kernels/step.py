"""Token-major KDA decode; whole FP32 state belongs to one vector participant."""
from ascriptor.a5 import *

T_MAX = 16
D = 128


@vf()
def initialize_state(state: Tensor):
    zero = RegList(DT.float, 2)
    zero <<= 0.0
    for row in range(D):
        state[row:row+1, 0:D] <<= zero
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def prepare_bf16(qb: Tensor, kb: Tensor, vb: Tensor,
                 q: Tensor, k: Tensor, v: Tensor, rows: Var, scale: Var):
    r = Reg(DT.float)
    for row in range(rows):
        for half in unroll(2):
            r <<= qb[row:row+1, half*64:half*64+64].unpack()
            r <<= r * scale
            q[row:row+1, half*64:half*64+64] <<= r
            r <<= kb[row:row+1, half*64:half*64+64].unpack()
            k[row:row+1, half*64:half*64+64] <<= r
            r <<= vb[row:row+1, half*64:half*64+64].unpack()
            v[row:row+1, half*64:half*64+64] <<= r
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def scale_fp32(q: Tensor, rows: Var, scale: Var):
    r = RegList(DT.float, 2)
    for row in range(rows):
        r <<= q[row:row+1, 0:D]
        r <<= r * scale
        q[row:row+1, 0:D] <<= r
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def recurrent_steps(state: Tensor, q: Tensor, k: Tensor, v: Tensor,
                    g: Tensor, beta: Tensor, out: Tensor, rows: Var):
    row = RegList(DT.float, 2)
    prediction = RegList(DT.float, 2)
    output = RegList(DT.float, 2)
    delta = RegList(DT.float, 2)
    product = RegList(DT.float, 2)
    decay = Reg(DT.float)
    key = Reg(DT.float)
    query = Reg(DT.float)
    weight = Reg(DT.float)
    for ti in range(rows):
        prediction <<= 0.0
        output <<= 0.0
        for ki in range(D):
            decay <<= g[ti:ti+1, ki:ki+1].single()
            decay <<= decay.exp()
            key <<= k[ti:ti+1, ki:ki+1].single()
            row <<= state[ki:ki+1, 0:D]
            row <<= row * decay
            state[ki:ki+1, 0:D] <<= row
            product <<= row * key
            prediction <<= prediction + product
        weight <<= beta[ti:ti+1, 0:1].single()
        delta <<= v[ti:ti+1, 0:D]
        delta <<= delta - prediction
        for ki in range(D):
            key <<= k[ti:ti+1, ki:ki+1].single()
            key <<= key * weight
            query <<= q[ti:ti+1, ki:ki+1].single()
            row <<= state[ki:ki+1, 0:D]
            product <<= delta * key
            row <<= row + product
            state[ki:ki+1, 0:D] <<= row
            product <<= row * query
            output <<= output + product
        out[ti:ti+1, 0:D] <<= output
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def store_bf16(source: Tensor, out: Tensor, rows: Var):
    value = Reg(DT.float)
    packed = Reg(DT.bfloat16)
    full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
    config = CastConfig(round_mode=RoundMode.TO_EVEN)
    for row in range(rows):
        for half in unroll(2):
            value <<= source[row:row+1, half*64:half*64+64]
            packed <<= value.astype(DT.bfloat16, config)
            reg_to_ub_downsample(out[row:row+1, half*64:half*64+64], packed, mask=full)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel(mode="vec")
def kda_decode_bf16_kernel(
    q: GM[bf16, ("B", "T", "H", 128)],
    k: GM[bf16, ("B", "T", "H", 128)],
    v: GM[bf16, ("B", "T", "HV", 128)],
    g: GM[f32, ("B", "T", "HV", 128)],
    beta: GM[f32, ("B", "T", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    o: GM[bf16, ("B", "T", "HV", 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32, T: i32, H: i32, HV: i32, has_initial: i32, scale: f32,
):
    state = Tensor(DT.float, [D, D], Position.UB)
    qb = Tensor(DT.bfloat16, [T_MAX, D], Position.UB)
    kb = Tensor(DT.bfloat16, [T_MAX, D], Position.UB)
    vb = Tensor(DT.bfloat16, [T_MAX, D], Position.UB)
    qu = Tensor(DT.float, [T_MAX, D], Position.UB)
    ku = Tensor(DT.float, [T_MAX, D], Position.UB)
    vu = Tensor(DT.float, [T_MAX, D], Position.UB)
    gu = Tensor(DT.float, [T_MAX, D], Position.UB)
    bu = Tensor(DT.float, [T_MAX, 8], Position.UB)
    ou = Tensor(DT.float, [T_MAX, D], Position.UB)
    ob = Tensor(DT.bfloat16, [T_MAX, D], Position.UB)
    count = B * HV
    per_vec = CeilDiv(count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, count)
    with auto_sync():
        for work in range(begin, end):
            bi = Var(work // HV)
            hi = Var(work % HV)
            qi = Var(hi // (HV // H))
            if has_initial != 0:
                state[0:D, 0:D] <<= initial_state[bi, hi, 0:D, 0:D]
            else:
                initialize_state(state)
            # Explicit row strides retain the intervening physical head axis.
            gm_to_ub_pad(qb[0:T, 0:D], q[bi, 0:T, qi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(H - 1) * D, dst_stride=0)
            gm_to_ub_pad(kb[0:T, 0:D], k[bi, 0:T, qi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(H - 1) * D, dst_stride=0)
            gm_to_ub_pad(vb[0:T, 0:D], v[bi, 0:T, hi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(HV - 1) * D, dst_stride=0)
            gm_to_ub_pad(gu[0:T, 0:D], g[bi, 0:T, hi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(HV - 1) * D, dst_stride=0)
            bu[0:T, 0:1] <<= beta[bi, 0:T, hi:hi+1]
            prepare_bf16(qb, kb, vb, qu, ku, vu, T, scale)
            recurrent_steps(state, qu, ku, vu, gu, bu, ou, T)
            store_bf16(ou, ob, T)
            ub_to_gm_pad(o[bi, 0:T, hi, 0:D], ob[0:T, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride=0, dst_stride_element=(HV - 1) * D)
            final_state[bi, hi, 0:D, 0:D] <<= state[0:D, 0:D]
    return o, final_state


@kernel(mode="vec")
def kda_decode_fp32_kernel(
    q: GM[f32, ("B", "T", "H", 128)],
    k: GM[f32, ("B", "T", "H", 128)],
    v: GM[f32, ("B", "T", "HV", 128)],
    g: GM[f32, ("B", "T", "HV", 128)],
    beta: GM[f32, ("B", "T", "HV")],
    initial_state: GM[f32, ("B", "HV", 128, 128)],
    o: GM[f32, ("B", "T", "HV", 128)],
    final_state: GM[f32, ("B", "HV", 128, 128)],
    B: i32, T: i32, H: i32, HV: i32, has_initial: i32, scale: f32,
):
    state = Tensor(DT.float, [D, D], Position.UB)
    qu = Tensor(DT.float, [T_MAX, D], Position.UB)
    ku = Tensor(DT.float, [T_MAX, D], Position.UB)
    vu = Tensor(DT.float, [T_MAX, D], Position.UB)
    gu = Tensor(DT.float, [T_MAX, D], Position.UB)
    bu = Tensor(DT.float, [T_MAX, 8], Position.UB)
    ou = Tensor(DT.float, [T_MAX, D], Position.UB)
    count = B * HV
    per_vec = CeilDiv(count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, count)
    with auto_sync():
        for work in range(begin, end):
            bi = Var(work // HV)
            hi = Var(work % HV)
            qi = Var(hi // (HV // H))
            if has_initial != 0:
                state[0:D, 0:D] <<= initial_state[bi, hi, 0:D, 0:D]
            else:
                initialize_state(state)
            gm_to_ub_pad(qu[0:T, 0:D], q[bi, 0:T, qi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(H - 1) * D, dst_stride=0)
            gm_to_ub_pad(ku[0:T, 0:D], k[bi, 0:T, qi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(H - 1) * D, dst_stride=0)
            gm_to_ub_pad(vu[0:T, 0:D], v[bi, 0:T, hi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(HV - 1) * D, dst_stride=0)
            gm_to_ub_pad(gu[0:T, 0:D], g[bi, 0:T, hi, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride_element=(HV - 1) * D, dst_stride=0)
            bu[0:T, 0:1] <<= beta[bi, 0:T, hi:hi+1]
            scale_fp32(qu, T, scale)
            recurrent_steps(state, qu, ku, vu, gu, bu, ou, T)
            ub_to_gm_pad(o[bi, 0:T, hi, 0:D], ou[0:T, 0:D],
                         n_burst=T, burst_len_element=D,
                         src_stride=0, dst_stride_element=(HV - 1) * D)
            final_state[bi, hi, 0:D, 0:D] <<= state[0:D, 0:D]
    return o, final_state
