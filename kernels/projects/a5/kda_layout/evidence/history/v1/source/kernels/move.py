"""Disjoint 64-element vector work items; source strides are element strides.

The GM storage-span view is metadata only. NDDMA performs all gather/reordering;
VF casts preserve the existing FP32/BF16 precision boundary. No cube resources.
"""
from ascriptor.a5 import *


@vf()
def narrow(source: Tensor, destination: Tensor, multiply: Var, factor: Var):
    value = Reg(DT.float)
    packed = Reg(DT.bfloat16)
    full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
    config = CastConfig(round_mode=RoundMode.TO_EVEN)
    value <<= source[0:1, 0:64]
    if multiply != 0:
        value <<= value * factor
    packed <<= value.astype(DT.bfloat16, config)
    reg_to_ub_downsample(destination[0:1, 0:64], packed, mask=full)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def widen(source: Tensor, destination: Tensor):
    value = Reg(DT.float)
    value <<= source[0:1, 0:64].unpack()
    destination[0:1, 0:64] <<= value
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def zero_f32(destination: Tensor):
    value = Reg(DT.float)
    value <<= 0.0
    destination[0:1, 0:64] <<= value
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def zero_bf16(destination: Tensor):
    value = Reg(DT.bfloat16)
    low = MaskReg(DT.bfloat16, init_mode=MaskType.LOWHALF)
    value <<= 0.0
    reg_to_ub_normal(destination[0:1, 0:64], value, mask=low)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel(mode="vec")
def kda_layout_bf16_bf16_kernel(
    source: GM[bf16, (1, "Storage")], destination: GM[bf16, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
):
    staging = Tensor(DT.bfloat16, [1, 128], Position.UB)
    work_count = Var(N / 64)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * 64)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+63*S4+1],
                            [S4], [1], [64], fence="mte2")
            ub_to_gm_pad(destination[0, offset:offset+64], staging,
                         n_burst=1, burst_len_element=64,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_f32_f32_kernel(
    source: GM[f32, (1, "Storage")], destination: GM[f32, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
):
    staging = Tensor(DT.float, [1, 64], Position.UB)
    work_count = Var(N / 64)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * 64)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+63*S4+1],
                            [S4], [1], [64], fence="mte2")
            ub_to_gm_pad(destination[0, offset:offset+64], staging,
                         n_burst=1, burst_len_element=64,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_bf16_f32_kernel(
    source: GM[bf16, (1, "Storage")], destination: GM[f32, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32,
):
    staging = Tensor(DT.bfloat16, [1, 128], Position.UB)
    target = Tensor(DT.float, [1, 64], Position.UB)
    work_count = Var(N / 64)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * 64)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+63*S4+1],
                            [S4], [1], [64], fence="mte2")
            widen(staging, target)
            ub_to_gm_pad(destination[0, offset:offset+64], target,
                         n_burst=1, burst_len_element=64,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_f32_bf16_kernel(
    source: GM[f32, (1, "Storage")], destination: GM[bf16, (1, "N")],
    Storage: i32, N: i32, D1: i32, D2: i32, D3: i32, D4: i32,
    S0: i32, S1: i32, S2: i32, S3: i32, S4: i32, multiply: i32, factor: f32,
):
    staging = Tensor(DT.float, [1, 64], Position.UB)
    target = Tensor(DT.bfloat16, [1, 128], Position.UB)
    work_count = Var(N / 64)
    per_vec = CeilDiv(work_count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, work_count)
    with auto_sync():
        for work in range(begin, end):
            offset = Var(work * 64)
            i4 = Var(offset % D4)
            rest3 = Var(offset / D4)
            i3 = Var(rest3 % D3)
            rest2 = Var(rest3 / D3)
            i2 = Var(rest2 % D2)
            rest1 = Var(rest2 / D2)
            i1 = Var(rest1 % D1)
            i0 = Var(rest1 / D1)
            source_offset = Var(i0*S0 + i1*S1 + i2*S2 + i3*S3 + i4*S4)
            gm_to_ub_nd_dma(staging, source[0, source_offset:source_offset+63*S4+1],
                            [S4], [1], [64], fence="mte2")
            narrow(staging, target, Var(multiply), Var(factor))
            ub_to_gm_pad(destination[0, offset:offset+64], target,
                         n_burst=1, burst_len_element=64,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_zero_bf16_kernel(destination: GM[bf16, (1, "N")], N: i32):
    staging = Tensor(DT.bfloat16, [1, 128], Position.UB)
    count = Var(N / 64)
    per_vec = CeilDiv(count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, count)
    with auto_sync():
        for work in range(begin, end):
            zero_bf16(staging)
            ub_to_gm_pad(destination[0, work*64:work*64+64], staging,
                         n_burst=1, burst_len_element=64,
                         src_stride=0, dst_stride_element=0)
    return destination


@kernel(mode="vec")
def kda_layout_zero_f32_kernel(destination: GM[f32, (1, "N")], N: i32):
    staging = Tensor(DT.float, [1, 64], Position.UB)
    count = Var(N / 64)
    per_vec = CeilDiv(count, GetVecNum())
    begin = Var(per_vec * GetVecIdx())
    end = Min(begin + per_vec, count)
    with auto_sync():
        for work in range(begin, end):
            zero_f32(staging)
            ub_to_gm_pad(destination[0, work*64:work*64+64], staging,
                         n_burst=1, burst_len_element=64,
                         src_stride=0, dst_stride_element=0)
    return destination
