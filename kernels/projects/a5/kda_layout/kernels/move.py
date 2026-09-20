"""Batch contiguous output tiles into one five-loop NDDMA and one GM store.

The host supplies shape-only tiling metadata; each tile contains at most4096
values. Cast precision and64-lane footprints match the original implementation.
"""
from ascriptor.a5 import *


@vf()
def narrow(source: Tensor, destination: Tensor, count: Var, multiply: Var, factor: Var):
    value = Reg(DT.float)
    packed = Reg(DT.bfloat16)
    full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
    config = CastConfig(round_mode=RoundMode.TO_EVEN)
    for index in range(count / 64):
        value <<= source[0:1, index*64:index*64+64]
        if multiply != 0:
            value <<= value * factor
        packed <<= value.astype(DT.bfloat16, config)
        reg_to_ub_downsample(destination[0:1, index*64:index*64+64], packed, mask=full)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def widen(source: Tensor, destination: Tensor, count: Var):
    value = Reg(DT.float)
    for index in range(count / 64):
        value <<= source[0:1, index*64:index*64+64].unpack()
        destination[0:1, index*64:index*64+64] <<= value
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def zero_f32(destination: Tensor):
    value = Reg(DT.float)
    value <<= 0.0
    for index in range(64):
        destination[0:1, index*64:index*64+64] <<= value
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def zero_bf16(destination: Tensor):
    value = Reg(DT.bfloat16)
    low = MaskReg(DT.bfloat16, init_mode=MaskType.LOWHALF)
    value <<= 0.0
    for index in range(64):
        reg_to_ub_normal(destination[0:1, index*64:index*64+64], value, mask=low)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel(mode="vec")
def kda_layout_bf16_bf16_kernel(
    source: GM[bf16, (1, "Storage")], destination: GM[bf16, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
    TileN: i32, T0: i32, T1: i32, T2: i32, T3: i32, T4: i32,
):
    staging = Tensor(DT.bfloat16, [1, 4096], Position.UB)
    work_count = Var(N / TileN)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    source_span = Var(1 + (T0-1)*S0 + (T1-1)*S1 + (T2-1)*S2 + (T3-1)*S3 + (T4-1)*S4)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * TileN)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+source_span],
                            [S4, S3, S2, S1, S0],
                            [1, T4, T4*T3, T4*T3*T2, T4*T3*T2*T1],
                            [T4, T3, T2, T1, T0], fence="mte2")
            ub_to_gm_pad(destination[0, offset:offset+TileN], staging,
                         n_burst=1, burst_len_element=TileN,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_f32_f32_kernel(
    source: GM[f32, (1, "Storage")], destination: GM[f32, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
    TileN: i32, T0: i32, T1: i32, T2: i32, T3: i32, T4: i32,
):
    staging = Tensor(DT.float, [1, 4096], Position.UB)
    work_count = Var(N / TileN)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    source_span = Var(1 + (T0-1)*S0 + (T1-1)*S1 + (T2-1)*S2 + (T3-1)*S3 + (T4-1)*S4)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * TileN)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+source_span],
                            [S4, S3, S2, S1, S0],
                            [1, T4, T4*T3, T4*T3*T2, T4*T3*T2*T1],
                            [T4, T3, T2, T1, T0], fence="mte2")
            ub_to_gm_pad(destination[0, offset:offset+TileN], staging,
                         n_burst=1, burst_len_element=TileN,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_bf16_f32_kernel(
    source: GM[bf16, (1, "Storage")], destination: GM[f32, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
    TileN: i32, T0: i32, T1: i32, T2: i32, T3: i32, T4: i32,
):
    staging = Tensor(DT.bfloat16, [1, 4096], Position.UB)
    target = Tensor(DT.float, [1, 4096], Position.UB)
    work_count = Var(N / TileN)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    source_span = Var(1 + (T0-1)*S0 + (T1-1)*S1 + (T2-1)*S2 + (T3-1)*S3 + (T4-1)*S4)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * TileN)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+source_span],
                            [S4, S3, S2, S1, S0],
                            [1, T4, T4*T3, T4*T3*T2, T4*T3*T2*T1],
                            [T4, T3, T2, T1, T0], fence="mte2")
            widen(staging, target, Var(TileN))
            ub_to_gm_pad(destination[0, offset:offset+TileN], target,
                         n_burst=1, burst_len_element=TileN,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_f32_bf16_kernel(
    source: GM[f32, (1, "Storage")], destination: GM[bf16, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
    TileN: i32, T0: i32, T1: i32, T2: i32, T3: i32, T4: i32, multiply: i32, factor: f32,
):
    staging = Tensor(DT.float, [1, 4096], Position.UB)
    target = Tensor(DT.bfloat16, [1, 4096], Position.UB)
    work_count = Var(N / TileN)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    source_span = Var(1 + (T0-1)*S0 + (T1-1)*S1 + (T2-1)*S2 + (T3-1)*S3 + (T4-1)*S4)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * TileN)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+source_span],
                            [S4, S3, S2, S1, S0],
                            [1, T4, T4*T3, T4*T3*T2, T4*T3*T2*T1],
                            [T4, T3, T2, T1, T0], fence="mte2")
            narrow(staging, target, Var(TileN), Var(multiply), Var(factor))
            ub_to_gm_pad(destination[0, offset:offset+TileN], target,
                         n_burst=1, burst_len_element=TileN,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_zero_bf16_kernel(destination: GM[bf16, (1, "N")], N: i32):
    staging = Tensor(DT.bfloat16, [1, 4096], Position.UB)
    count = CeilDiv(N, 4096)
    per_vec = CeilDiv(count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, count)
    with auto_sync():
        zero_bf16(staging)
        for work in range(begin, end):
            offset = Var(work * 4096)
            active = Min(4096, N-offset)
            ub_to_gm_pad(destination[0, offset:offset+active], staging,
                         n_burst=1, burst_len_element=active,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_zero_f32_kernel(destination: GM[f32, (1, "N")], N: i32):
    staging = Tensor(DT.float, [1, 4096], Position.UB)
    count = CeilDiv(N, 4096)
    per_vec = CeilDiv(count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, count)
    with auto_sync():
        zero_f32(staging)
        for work in range(begin, end):
            offset = Var(work * 4096)
            active = Min(4096, N-offset)
            ub_to_gm_pad(destination[0, offset:offset+active], staging,
                         n_burst=1, burst_len_element=active,
                         src_stride=0, dst_stride_element=0)
    return destination
