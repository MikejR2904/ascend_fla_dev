"""Typed A5 raw-input preparation, with fixed row reductions and large DMA tiles.

Factories specialize static dtype closures. Each entry receives its permanent
operator name before IR creation, so chunk and decode may use independent bd.
"""
from ascriptor.a5 import *


def make_norm(name, input_dtype, output_dtype):
    @vf()
    def normalize(source: Tensor, destination: Tensor, rows: Var):
        x0 = Reg(DT.float)
        x1 = Reg(DT.float)
        square0 = Reg(DT.float)
        square1 = Reg(DT.float)
        sum0 = Reg(DT.float)
        sum1 = Reg(DT.float)
        denominator = Reg(DT.float)
        broadcast = Reg(DT.float)
        result = Reg(DT.float)
        index0 = Reg(DT.uint32)
        index0.fill(0)
        packed = Reg(DT.bfloat16)
        full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
        config = CastConfig(round_mode=RoundMode.TO_EVEN)
        for row in range(rows):
            if input_dtype == bf16:
                x0 <<= source[0, row*128:row*128+64].unpack()
                x1 <<= source[0, row*128+64:row*128+128].unpack()
            else:
                x0 <<= source[0, row*128:row*128+64]
                x1 <<= source[0, row*128+64:row*128+128]
            square0 <<= x0 * x0
            square1 <<= x1 * x1
            cadd(sum0, square0)
            cadd(sum1, square1)
            denominator <<= sum0 + sum1
            denominator <<= denominator + 1e-6
            denominator <<= denominator.sqrt()
            gather(broadcast, denominator, index0)
            result <<= x0 / broadcast
            if output_dtype == bf16:
                packed <<= result.astype(DT.bfloat16, config)
                reg_to_ub_downsample(destination[0, row*128:row*128+64], packed, mask=full)
            else:
                destination[0, row*128:row*128+64] <<= result
            result <<= x1 / broadcast
            if output_dtype == bf16:
                packed <<= result.astype(DT.bfloat16, config)
                reg_to_ub_downsample(destination[0, row*128+64:row*128+128], packed, mask=full)
            else:
                destination[0, row*128+64:row*128+128] <<= result
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def norm_kernel(source: GM[input_dtype, (1, 'N')],
                    destination: GM[output_dtype, (1, 'N')], N: i32):
        incoming = Tensor(input_dtype, [1, 4096], Position.UB)
        outgoing = Tensor(output_dtype, [1, 4096], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                offset = Var(work * 4096)
                count = Min(4096, N - offset)
                gm_to_ub_pad(incoming, source[0, offset:offset+count], n_burst=1,
                             burst_len_element=count, src_stride_element=0, dst_stride=0)
                normalize(incoming, outgoing, Var(count / 128))
                ub_to_gm_pad(destination[0, offset:offset+count], outgoing, n_burst=1,
                             burst_len_element=count, src_stride=0, dst_stride_element=0)
        return destination

    norm_kernel.name = name
    return norm_kernel


def make_gate(name, input_dtype, a_dtype, bias_dtype):
    @vf()
    def transform(source: Tensor, avec: Tensor, bias: Tensor,
                  destination: Tensor, rows: Var):
        decay = Reg(DT.float)
        narrow_decay = Reg(DT.bfloat16)
        if a_dtype == bf16:
            ub_to_reg_single(narrow_decay, avec[0, 0:1])
            decay <<= narrow_decay.astype(DT.float)
        else:
            ub_to_reg_single(decay, avec[0, 0:1])
        decay <<= decay.exp()
        decay <<= -decay
        bias0 = Reg(DT.float)
        bias1 = Reg(DT.float)
        if bias_dtype == bf16:
            bias0 <<= bias[0, 0:64].unpack()
            bias1 <<= bias[0, 64:128].unpack()
        else:
            bias0 <<= bias[0, 0:64]
            bias1 <<= bias[0, 64:128]
        value = Reg(DT.float)
        u = Reg(DT.float)
        argument = Reg(DT.float)
        exponential = Reg(DT.float)
        plus_one = Reg(DT.float)
        denominator = Reg(DT.float)
        correction = Reg(DT.float)
        logarithm = Reg(DT.float)
        result = Reg(DT.float)
        one = Reg(DT.float)
        threshold = Reg(DT.float)
        one.fill(1.)
        threshold.fill(20.)
        upper = MaskReg(DT.float)
        tiny = MaskReg(DT.float)
        for row in range(rows):
            for half in unroll(2):
                if input_dtype == bf16:
                    value <<= source[0, row*128+half*64:row*128+half*64+64].unpack()
                else:
                    value <<= source[0, row*128+half*64:row*128+half*64+64]
                if half == 0:
                    u <<= value + bias0
                else:
                    u <<= value + bias1
                compare(upper, u, 20., CompareMode.GT)
                select(argument, threshold, u, upper)
                exponential <<= argument.exp()
                plus_one <<= exponential + 1.
                denominator <<= plus_one - 1.
                compare(tiny, plus_one, 1., CompareMode.EQ)
                select(denominator, one, denominator, tiny)
                correction <<= exponential / denominator
                logarithm <<= plus_one.ln()
                result <<= logarithm * correction
                select(result, exponential, result, tiny)
                select(result, u, result, upper)
                result <<= decay * result
                destination[0, row*128+half*64:row*128+half*64+64] <<= result
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def gate_kernel(source: GM[input_dtype, (1, 'N')],
                    alog: GM[a_dtype, (1, 'HV')], bias: GM[bias_dtype, (1, 'Channels')],
                    destination: GM[f32, (1, 'N')], N: i32, HV: i32, Channels: i32, BT: i32):
        incoming = Tensor(input_dtype, [1, 4096], Position.UB)
        avec = Tensor(a_dtype, [1, 16], Position.UB)
        bvec = Tensor(bias_dtype, [1, 128], Position.UB)
        outgoing = Tensor(DT.float, [1, 4096], Position.UB)
        chunks = CeilDiv(BT, 32)
        works = Var(HV * chunks)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                head = Var(work / chunks)
                start_row = Var((work % chunks) * 32)
                rows = Min(32, BT - start_row)
                offset = Var(start_row * Channels + head * 128)
                span = Var((rows - 1) * Channels + 128)
                gm_to_ub_pad(avec, alog[0, head:head+1], n_burst=1,
                             burst_len_element=1, src_stride_element=0, dst_stride=0, pad=0.)
                gm_to_ub_pad(bvec, bias[0, head*128:head*128+128], n_burst=1,
                             burst_len_element=128, src_stride_element=0, dst_stride=0)
                gm_to_ub_pad(incoming, source[0, offset:offset+span], n_burst=rows,
                             burst_len_element=128, src_stride_element=Channels-128, dst_stride=0)
                transform(incoming, avec, bvec, outgoing, Var(rows))
                ub_to_gm_pad(destination[0, offset:offset+span], outgoing, n_burst=rows,
                             burst_len_element=128, src_stride=0, dst_stride_element=Channels-128)
        return destination

    gate_kernel.name = name
    return gate_kernel


def make_beta(name, input_dtype):
    @vf()
    def sigmoid(source: Tensor, destination: Tensor, count: Var):
        x = Reg(DT.float)
        result = Reg(DT.float)
        one = Reg(DT.float)
        one.fill(1.)
        for index in range(count / 64):
            if input_dtype == bf16:
                x <<= source[0, index*64:index*64+64].unpack()
            else:
                x <<= source[0, index*64:index*64+64]
            result <<= -x
            result <<= result.exp()
            result <<= result + 1.
            result <<= one / result
            destination[0, index*64:index*64+64] <<= result
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def beta_kernel(source: GM[input_dtype, (1, 'N')],
                    destination: GM[f32, (1, 'N')], N: i32):
        incoming = Tensor(input_dtype, [1, 4096], Position.UB)
        outgoing = Tensor(DT.float, [1, 4096], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                offset = Var(work * 4096)
                count = Min(4096, N - offset)
                padded = Align64(count)
                gm_to_ub_nd_dma(incoming, source[0, offset:offset+count], [1], [1], [count],
                                loop_right_pad=[padded-count], constant_value=0., fence='mte2')
                sigmoid(incoming, outgoing, Var(padded))
                ub_to_gm_pad(destination[0, offset:offset+count], outgoing, n_burst=1,
                             burst_len_element=count, src_stride=0, dst_stride_element=0)
        return destination

    beta_kernel.name = name
    return beta_kernel
