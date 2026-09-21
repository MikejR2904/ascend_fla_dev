"""Raw preparation derivatives with fixed work items and deterministic reductions."""
from ascriptor.a5 import *


def make_norm_backward(name, raw_dtype):
    @vf()
    def differentiate(source: Tensor, sensitivity: Tensor, destination: Tensor, rows: Var):
        x0 = Reg(DT.float)
        x1 = Reg(DT.float)
        d0 = Reg(DT.float)
        d1 = Reg(DT.float)
        sh = Reg(DT.float)
        sl = Reg(DT.float)
        dh = Reg(DT.float)
        dl = Reg(DT.float)
        ph = Reg(DT.float)
        pl = Reg(DT.float)
        qh = Reg(DT.float)
        ql = Reg(DT.float)
        a_hi = Reg(DT.float)
        a_lo = Reg(DT.float)
        b_hi = Reg(DT.float)
        b_lo = Reg(DT.float)
        split = Reg(DT.float)
        temp = Reg(DT.float)
        summed = Reg(DT.float)
        virtual_b = Reg(DT.float)
        virtual_a = Reg(DT.float)
        error_a = Reg(DT.float)
        error_b = Reg(DT.float)
        low = Reg(DT.float)
        numerator = Reg(DT.float)
        denominator = Reg(DT.float)
        result = Reg(DT.float)
        row_scale = Reg(DT.float)
        epsilon = Reg(DT.float)
        large = MaskReg(DT.float)
        index = Reg(DT.uint32)
        signed_index = Reg(DT.int)
        step = Reg(DT.uint32)
        partner = Reg(DT.uint32)
        index0 = Reg(DT.uint32)
        signed_index.arange(0)
        index <<= signed_index.reinterpret(DT.uint32)
        index0.fill(0)
        packed = Reg(DT.bfloat16)
        full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
        config = CastConfig(round_mode=RoundMode.TO_EVEN)
        for row in range(rows):
            if raw_dtype == bf16:
                x0 <<= source[0, row*128:row*128+64].unpack()
                x1 <<= source[0, row*128+64:row*128+128].unpack()
            else:
                x0 <<= source[0, row*128:row*128+64]
                x1 <<= source[0, row*128+64:row*128+128]
            d0 <<= sensitivity[0, row*128:row*128+64].unpack()
            d1 <<= sensitivity[0, row*128+64:row*128+128].unpack()
            # Binary scaling keeps the derivative's intermediate products in
            # range. Small rows retain scale1; no host preprocessing is used.
            temp <<= x0.abs()
            result <<= x1.abs()
            cmax(denominator, temp)
            cmax(numerator, result)
            vmax(denominator, denominator, numerator)
            gather(denominator, denominator, index0)
            compare(large, denominator, 16., CompareMode.GE)
            partner <<= denominator.reinterpret(DT.uint32)
            step.fill(0x7f800000)
            vand(partner, partner, step)
            step.fill(0x7f000000)
            partner <<= step - partner
            row_scale <<= partner.reinterpret(DT.float)
            vmaxs(row_scale, row_scale, 7.52316384526264e-37)
            numerator.fill(1.)
            select(row_scale, row_scale, numerator, large)
            x0 <<= x0 * row_scale
            x1 <<= x1 * row_scale
            epsilon <<= row_scale * 1.e-6
            epsilon <<= epsilon * row_scale
            # Two-component products and a fixed compensated K128 reduction.
            for left0, left1, right0, right1, high_out, low_out in (
                    (x0, x1, x0, x1, sh, sl), (x0, x1, d0, d1, dh, dl)):
                for a, b, high, residual in ((left0, right0, ph, pl), (left1, right1, qh, ql)):
                    high <<= a * b
                    if raw_dtype == bf16:
                        residual.fill(0.)
                    else:
                        split <<= a * 4097.
                        temp <<= split - a
                        a_hi <<= split - temp
                        a_lo <<= a - a_hi
                        split <<= b * 4097.
                        temp <<= split - b
                        b_hi <<= split - temp
                        b_lo <<= b - b_hi
                        residual <<= a_hi * b_hi
                        residual <<= residual - high
                        temp <<= a_hi * b_lo
                        residual <<= residual + temp
                        temp <<= a_lo * b_hi
                        residual <<= residual + temp
                        temp <<= a_lo * b_lo
                        residual <<= residual + temp
                # Keep the forward two64-cadd value as the leading component;
                # the explicit tree below supplies its missing low component.
                cadd(denominator, ph)
                cadd(result, qh)
                denominator <<= denominator + result
                gather(denominator, denominator, index0)
                # First combine corresponding lanes from the two64 halves,
                # then reduce those64 pairs in an explicit deterministic tree.
                for level in unroll(7):
                    if level > 0:
                        step.fill(64 >> level)
                        vxor(partner, index, step)
                        gather(qh, ph, partner)
                        gather(ql, pl, partner)
                    summed <<= ph + qh
                    virtual_b <<= summed - ph
                    virtual_a <<= summed - virtual_b
                    error_b <<= qh - virtual_b
                    error_a <<= ph - virtual_a
                    temp <<= error_a + error_b
                    low <<= pl + ql
                    low <<= low + temp
                    ph <<= summed + low
                    temp <<= ph - summed
                    pl <<= low - temp
                gather(high_out, ph, index0)
                gather(low_out, pl, index0)
                temp <<= high_out - denominator
                low_out <<= temp + low_out
                high_out <<= denominator
            # Keep the epsilon contribution outside the cancellation. The
            # numerator products also need their rounding residuals: improving
            # the row reduction alone cannot recover these small derivatives.
            for x, gy, offset in ((x0, d0, 0), (x1, d1, 64)):
                for a, b, high, residual in ((gy, sh, ph, pl), (x, dh, qh, ql)):
                    high <<= a * b
                    split <<= a * 4097.
                    temp <<= split - a
                    a_hi <<= split - temp
                    a_lo <<= a - a_hi
                    split <<= b * 4097.
                    temp <<= split - b
                    b_hi <<= split - temp
                    b_lo <<= b - b_hi
                    residual <<= a_hi * b_hi
                    residual <<= residual - high
                    temp <<= a_hi * b_lo
                    residual <<= residual + temp
                    temp <<= a_lo * b_hi
                    residual <<= residual + temp
                    temp <<= a_lo * b_lo
                    residual <<= residual + temp
                qh <<= -qh
                ql <<= -ql
                summed <<= ph + qh
                virtual_b <<= summed - ph
                virtual_a <<= summed - virtual_b
                error_b <<= qh - virtual_b
                error_a <<= ph - virtual_a
                temp <<= error_a + error_b
                low <<= pl + ql
                low <<= low + temp
                ph <<= summed + low
                temp <<= ph - summed
                pl <<= low - temp
                numerator <<= gy * sl
                temp <<= x * dl
                numerator <<= numerator - temp
                temp <<= gy * epsilon
                numerator <<= numerator + temp
                numerator <<= numerator + pl
                numerator <<= ph + numerator
                temp <<= sl + epsilon
                summed <<= sh + temp
                denominator <<= summed.sqrt()
                denominator <<= summed * denominator
                result <<= numerator / denominator
                result <<= result * row_scale
                if raw_dtype == bf16:
                    packed <<= result.astype(DT.bfloat16, config)
                    reg_to_ub_downsample(destination[0, row*128+offset:row*128+offset+64], packed, mask=full)
                else:
                    destination[0, row*128+offset:row*128+offset+64] <<= result
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def norm_backward(source: GM[raw_dtype, (1, 'N')], sensitivity: GM[bf16, (1, 'N')],
                      destination: GM[raw_dtype, (1, 'N')], N: i32):
        incoming = Tensor(raw_dtype, [1, 4096], Position.UB)
        gradient = Tensor(DT.bfloat16, [1, 4096], Position.UB)
        outgoing = Tensor(raw_dtype, [1, 4096], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                offset = Var(work * 4096)
                count = Min(4096, N-offset)
                gm_to_ub_pad(incoming, source[0, offset:offset+count], n_burst=1,
                             burst_len_element=count, src_stride_element=0, dst_stride=0)
                gm_to_ub_pad(gradient, sensitivity[0, offset:offset+count], n_burst=1,
                             burst_len_element=count, src_stride_element=0, dst_stride=0)
                differentiate(incoming, gradient, outgoing, Var(count / 128))
                ub_to_gm_pad(destination[0, offset:offset+count], outgoing, n_burst=1,
                             burst_len_element=count, src_stride=0, dst_stride_element=0)
        return destination

    norm_backward.name = name
    return norm_backward


def make_gate_backward(name, raw_dtype, a_dtype, bias_dtype):
    @vf()
    def contributions(source: Tensor, sensitivity: Tensor, avec: Tensor, bias: Tensor,
                      destination: Tensor, apart: Tensor, bpart: Tensor, partial: Tensor, rows: Var):
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
        gy = Reg(DT.float)
        u = Reg(DT.float)
        argument = Reg(DT.float)
        exponential = Reg(DT.float)
        denominator = Reg(DT.float)
        plus_one = Reg(DT.float)
        correction = Reg(DT.float)
        logarithm = Reg(DT.float)
        softplus = Reg(DT.float)
        sigmoid = Reg(DT.float)
        positive_sigmoid = Reg(DT.float)
        result = Reg(DT.float)
        one = Reg(DT.float)
        zero = Reg(DT.float)
        threshold = Reg(DT.float)
        one.fill(1.)
        zero.fill(0.)
        threshold.fill(20.)
        upper = MaskReg(DT.float)
        positive = MaskReg(DT.float)
        tiny = MaskReg(DT.float)
        packed = Reg(DT.bfloat16)
        full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
        config = CastConfig(round_mode=RoundMode.TO_EVEN)
        for row in range(rows):
            for half in unroll(2):
                if raw_dtype == bf16:
                    value <<= source[0, row*128+half*64:row*128+half*64+64].unpack()
                else:
                    value <<= source[0, row*128+half*64:row*128+half*64+64]
                gy <<= sensitivity[0, row*128+half*64:row*128+half*64+64]
                if half == 0:
                    u <<= value + bias0
                else:
                    u <<= value + bias1
                compare(upper, u, 20., CompareMode.GT)
                compare(positive, u, 0., CompareMode.GE)
                argument <<= u.abs()
                argument <<= -argument
                exponential <<= argument.exp()
                denominator <<= exponential + 1.
                sigmoid <<= exponential / denominator
                positive_sigmoid <<= one / denominator
                select(sigmoid, positive_sigmoid, sigmoid, positive)
                select(sigmoid, one, sigmoid, upper)
                result <<= gy * decay
                result <<= result * sigmoid
                bpart[0, row*128+half*64:row*128+half*64+64] <<= result
                if raw_dtype == bf16:
                    packed <<= result.astype(DT.bfloat16, config)
                    reg_to_ub_downsample(destination[0, row*128+half*64:row*128+half*64+64], packed, mask=full)
                else:
                    destination[0, row*128+half*64:row*128+half*64+64] <<= result
                select(argument, threshold, u, upper)
                exponential <<= argument.exp()
                plus_one <<= exponential + 1.
                denominator <<= plus_one - 1.
                compare(tiny, plus_one, 1., CompareMode.EQ)
                select(denominator, one, denominator, tiny)
                correction <<= exponential / denominator
                logarithm <<= plus_one.ln()
                softplus <<= logarithm * correction
                select(softplus, exponential, softplus, tiny)
                select(softplus, u, softplus, upper)
                result <<= gy * softplus
                apart[0, row*128+half*64:row*128+half*64+64] <<= result
        for row in range(rows, 32):
            for half in unroll(2):
                apart[0, row*128+half*64:row*128+half*64+64] <<= zero
                bpart[0, row*128+half*64:row*128+half*64+64] <<= zero
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        left = Reg(DT.float)
        right = Reg(DT.float)
        for level in unroll(5):
            for pair in range(32 // (2 << level)):
                for half in unroll(2):
                    left <<= apart[0, pair*(256 << level)+half*64:pair*(256 << level)+half*64+64]
                    right <<= apart[0, pair*(256 << level)+(128 << level)+half*64:pair*(256 << level)+(128 << level)+half*64+64]
                    left <<= left + right
                    apart[0, pair*(256 << level)+half*64:pair*(256 << level)+half*64+64] <<= left
                    left <<= bpart[0, pair*(256 << level)+half*64:pair*(256 << level)+half*64+64]
                    right <<= bpart[0, pair*(256 << level)+(128 << level)+half*64:pair*(256 << level)+(128 << level)+half*64+64]
                    left <<= left + right
                    bpart[0, pair*(256 << level)+half*64:pair*(256 << level)+half*64+64] <<= left
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        for half in unroll(2):
            left <<= apart[0, half*64:half*64+64]
            partial[0, half*64:half*64+64] <<= left
            left <<= bpart[0, half*64:half*64+64]
            partial[0, 128+half*64:192+half*64] <<= left
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def gate_backward(source: GM[raw_dtype, (1, 'N')], sensitivity: GM[f32, (1, 'N')],
                      alog: GM[a_dtype, (1, 'HV')], bias: GM[bias_dtype, (1, 'Channels')],
                      destination: GM[raw_dtype, (1, 'N')], partials: GM[f32, (1, 'P')],
                      N: i32, HV: i32, Channels: i32, BT: i32, P: i32):
        incoming = Tensor(raw_dtype, [1, 4096], Position.UB)
        gradient = Tensor(DT.float, [1, 4096], Position.UB)
        outgoing = Tensor(raw_dtype, [1, 4096], Position.UB)
        avec = Tensor(a_dtype, [1, 16], Position.UB)
        bvec = Tensor(bias_dtype, [1, 128], Position.UB)
        apart = Tensor(DT.float, [1, 4096], Position.UB)
        bpart = Tensor(DT.float, [1, 4096], Position.UB)
        partial = Tensor(DT.float, [1, 256], Position.UB)
        chunks = CeilDiv(BT, 32)
        works = Var(HV * chunks)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin + per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                head = Var(work / chunks)
                start = Var((work % chunks) * 32)
                rows = Min(32, BT-start)
                offset = Var(start*Channels+head*128)
                span = Var((rows-1)*Channels+128)
                gm_to_ub_pad(avec, alog[0, head:head+1], n_burst=1,
                             burst_len_element=1, src_stride_element=0, dst_stride=0, pad=0.)
                gm_to_ub_pad(bvec, bias[0, head*128:head*128+128], n_burst=1,
                             burst_len_element=128, src_stride_element=0, dst_stride=0)
                gm_to_ub_pad(incoming, source[0, offset:offset+span], n_burst=rows,
                             burst_len_element=128, src_stride_element=Channels-128, dst_stride=0)
                gm_to_ub_pad(gradient, sensitivity[0, offset:offset+span], n_burst=rows,
                             burst_len_element=128, src_stride_element=Channels-128, dst_stride=0)
                contributions(incoming, gradient, avec, bvec, outgoing, apart, bpart, partial, Var(rows))
                ub_to_gm_pad(destination[0, offset:offset+span], outgoing, n_burst=rows,
                             burst_len_element=128, src_stride=0, dst_stride_element=Channels-128)
                ub_to_gm_pad(partials[0, work*256:work*256+256], partial, n_burst=1,
                             burst_len_element=256, src_stride=0, dst_stride_element=0)
        return destination, partials

    gate_backward.name = name
    return gate_backward


def make_gate_reduce(name, a_dtype, bias_dtype):
    heads_per_work = 16 if a_dtype == bf16 else 8

    @vf()
    def initialize(state: Tensor):
        zero = Reg(DT.float)
        zero.fill(0.)
        for part in unroll(8):
            state[0, part*64:part*64+64] <<= zero
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @vf()
    def accumulate(partials: Tensor, state: Tensor, rows: Var):
        total = Reg(DT.float)
        compensation = Reg(DT.float)
        value = Reg(DT.float)
        adjusted = Reg(DT.float)
        updated = Reg(DT.float)
        difference = Reg(DT.float)
        for part in unroll(4):
            total <<= state[0, part*64:part*64+64]
            compensation <<= state[0, 256+part*64:320+part*64]
            for row in range(rows):
                value <<= partials[0, row*256+part*64:row*256+part*64+64]
                adjusted <<= value - compensation
                updated <<= total + adjusted
                difference <<= updated - total
                compensation <<= difference - adjusted
                total <<= updated
            state[0, part*64:part*64+64] <<= total
            state[0, 256+part*64:320+part*64] <<= compensation
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @vf()
    def finish(state: Tensor, avec: Tensor, aout: Tensor, bout: Tensor, slot: Var):
        v0 = Reg(DT.float)
        v1 = Reg(DT.float)
        sum0 = Reg(DT.float)
        sum1 = Reg(DT.float)
        decay = Reg(DT.float)
        result = Reg(DT.float)
        narrow = Reg(DT.bfloat16)
        full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
        config = CastConfig(round_mode=RoundMode.TO_EVEN)
        v0 <<= state[0, 0:64]
        v1 <<= state[0, 64:128]
        cadd(sum0, v0)
        cadd(sum1, v1)
        result <<= sum0 + sum1
        if a_dtype == bf16:
            ub_to_reg_single(narrow, avec[0, slot:slot+1])
            decay <<= narrow.astype(DT.float)
        else:
            ub_to_reg_single(decay, avec[0, slot:slot+1])
        decay <<= decay.exp()
        decay <<= -decay
        result <<= result * decay
        if a_dtype == bf16:
            narrow <<= result.astype(DT.bfloat16, config)
            reg_to_ub_single(aout[0, slot:slot+1], narrow)
        else:
            reg_to_ub_single(aout[0, slot:slot+1], result)
        for half in unroll(2):
            result <<= state[0, 128+half*64:192+half*64]
            if bias_dtype == bf16:
                narrow <<= result.astype(DT.bfloat16, config)
                reg_to_ub_downsample(bout[0, half*64:half*64+64], narrow, mask=full)
            else:
                bout[0, half*64:half*64+64] <<= result
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def gate_reduce(partials: GM[f32, (1, 'P')], alog: GM[a_dtype, (1, 'HV')],
                    da: GM[a_dtype, (1, 'HV')], db: GM[bias_dtype, (1, 'Channels')],
                    P: i32, HV: i32, Channels: i32, Chunks: i32):
        incoming = Tensor(DT.float, [1, 4096], Position.UB)
        state = Tensor(DT.float, [1, 512], Position.UB)
        avec = Tensor(a_dtype, [1, 16], Position.UB)
        aout = Tensor(a_dtype, [1, 16], Position.UB)
        bout = Tensor(bias_dtype, [1, 128], Position.UB)
        works = CeilDiv(HV, heads_per_work)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec * GetVecIdx())
        end = Min(begin+per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                head_begin = Var(work*heads_per_work)
                heads = Min(heads_per_work, HV-head_begin)
                gm_to_ub_pad(avec, alog[0, head_begin:head_begin+heads], n_burst=1,
                             burst_len_element=heads, src_stride_element=0, dst_stride=0, pad=0.)
                for slot in range(heads):
                    head = Var(head_begin+slot)
                    initialize(state)
                    for tile in range(CeilDiv(Chunks, 16)):
                        start = Var(tile*16)
                        rows = Min(16, Chunks-start)
                        offset = Var((head*Chunks+start)*256)
                        count = Var(rows*256)
                        gm_to_ub_pad(incoming, partials[0, offset:offset+count], n_burst=1,
                                     burst_len_element=count, src_stride_element=0, dst_stride=0)
                        accumulate(incoming, state, Var(rows))
                    finish(state, avec, aout, bout, Var(slot))
                    ub_to_gm_pad(db[0, head*128:head*128+128], bout, n_burst=1,
                                 burst_len_element=128, src_stride=0, dst_stride_element=0)
                ub_to_gm_pad(da[0, head_begin:head_begin+heads], aout, n_burst=1,
                             burst_len_element=heads, src_stride=0, dst_stride_element=0)
        return da, db

    gate_reduce.name = name
    return gate_reduce


def make_beta_backward(name, raw_dtype):
    @vf()
    def differentiate(probability: Tensor, sensitivity: Tensor, destination: Tensor, count: Var):
        s = Reg(DT.float)
        gy = Reg(DT.float)
        complement = Reg(DT.float)
        result = Reg(DT.float)
        one = Reg(DT.float)
        one.fill(1.)
        packed = Reg(DT.bfloat16)
        full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
        config = CastConfig(round_mode=RoundMode.TO_EVEN)
        for index in range(count / 64):
            s <<= probability[0, index*64:index*64+64]
            gy <<= sensitivity[0, index*64:index*64+64]
            complement <<= one - s
            result <<= gy * s
            result <<= result * complement
            if raw_dtype == bf16:
                packed <<= result.astype(DT.bfloat16, config)
                reg_to_ub_downsample(destination[0, index*64:index*64+64], packed, mask=full)
            else:
                destination[0, index*64:index*64+64] <<= result
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def beta_backward(probability: GM[f32, (1, 'N')], sensitivity: GM[f32, (1, 'N')],
                      destination: GM[raw_dtype, (1, 'N')], N: i32):
        probability_ub = Tensor(DT.float, [1, 4096], Position.UB)
        gradient = Tensor(DT.float, [1, 4096], Position.UB)
        outgoing = Tensor(raw_dtype, [1, 4096], Position.UB)
        works = CeilDiv(N, 4096)
        per_vec = CeilDiv(works, GetVecNum())
        begin = Var(per_vec*GetVecIdx())
        end = Min(begin+per_vec, works)
        with auto_sync():
            for work in range(begin, end):
                offset = Var(work*4096)
                count = Min(4096, N-offset)
                padded = Align64(count)
                gm_to_ub_nd_dma(probability_ub, probability[0, offset:offset+count], [1], [1], [count],
                                loop_right_pad=[padded-count], constant_value=0., fence='mte2')
                gm_to_ub_nd_dma(gradient, sensitivity[0, offset:offset+count], [1], [1], [count],
                                loop_right_pad=[padded-count], constant_value=0., fence='mte2')
                differentiate(probability_ub, gradient, outgoing, Var(padded))
                ub_to_gm_pad(destination[0, offset:offset+count], outgoing, n_burst=1,
                             burst_len_element=count, src_stride=0, dst_stride_element=0)
        return destination

    beta_backward.name = name
    return beta_backward
