"""Raw preparation derivatives with fixed work items and deterministic reductions."""
from ascriptor.a5 import *
import math
import struct


def _split_constant(value):
    high = struct.unpack('f', struct.pack('f', value))[0]
    return high, value - high


_EXP_COEFFICIENTS = tuple(_split_constant(1 / math.factorial(n)) for n in range(13, -1, -1))
_LOG_COEFFICIENTS = tuple(_split_constant(1 / n) for n in range(39, 0, -2))
_EXP_FIRST = _split_constant(1 / math.factorial(14))
_LOG_FIRST = _split_constant(1 / 41)


@func
def _dd_add(oh, ol, ah, al, bh, bl):
    s = Reg(DT.float)
    v = Reg(DT.float)
    e = Reg(DT.float)
    t = Reg(DT.float)
    h = Reg(DT.float)
    l = Reg(DT.float)
    overflow = MaskReg(DT.float)
    s <<= ah + bh
    v <<= s - ah
    t <<= s - v
    e <<= ah - t
    t <<= bh - v
    e <<= e + t
    t <<= al + bl
    e <<= e + t
    h <<= s + e
    t <<= h - s
    l <<= e - t
    # A finite two-component expansion cannot encode an infinite residual.
    # Keep the primary IEEE result instead of turning Inf into Inf-Inf NaN.
    e.fill(0.)
    t <<= h.abs()
    compare(overflow, t, 3.4028234663852886e38, CompareMode.GT)
    select(l, e, l, overflow)
    t <<= s.abs()
    compare(overflow, t, 3.4028234663852886e38, CompareMode.GT)
    select(h, s, h, overflow)
    select(l, e, l, overflow)
    oh <<= h
    ol <<= l


@func
def _dd_add_constant(oh, ol, ah, al, high, low):
    # Immediate operands avoid retaining every polynomial coefficient in a
    # vector register across the enclosing row loop.
    # Both Horner callers have abs(ah) < high: the exp product/coefficient
    # ratio is below .372, and the log ratio below .127. FastTwoSum therefore
    # recovers the same exact residual, with the low-term order unchanged.
    s = Reg(DT.float)
    e = Reg(DT.float)
    t = Reg(DT.float)
    h = Reg(DT.float)
    l = Reg(DT.float)
    s <<= ah + high
    t <<= s - high
    e <<= ah - t
    t <<= al + low
    e <<= e + t
    h <<= s + e
    t <<= h - s
    l <<= e - t
    oh <<= h
    ol <<= l


@func
def _dd_mul(oh, ol, ah, al, bh, bl, protect_overflow=True):
    # A fused FP32 product-minus-rounded-product retains the product residual
    # without mantissa splitting. Every call still carries both DD components;
    # cross-product addition and renormalization retain their original order.
    p = Reg(DT.float)
    e = Reg(DT.float)
    t = Reg(DT.float)
    h = Reg(DT.float)
    l = Reg(DT.float)
    p <<= ah * bh
    e <<= -p
    muladddst(e, ah, bh)
    t <<= ah * bl
    e <<= e + t
    t <<= al * bh
    e <<= e + t
    h <<= p + e
    t <<= h - p
    l <<= e - t
    # Only the bounded exp/log polynomial callsites disable this guard.
    if protect_overflow:
        zero = Reg(DT.float)
        overflow = MaskReg(DT.float)
        # A finite two-component expansion cannot encode an infinite residual.
        # Keep the primary IEEE result instead of turning Inf into Inf-Inf NaN.
        zero.fill(0.)
        t <<= h.abs()
        compare(overflow, t, 3.4028234663852886e38, CompareMode.GT)
        select(l, zero, l, overflow)
        t <<= p.abs()
        compare(overflow, t, 3.4028234663852886e38, CompareMode.GT)
        select(h, p, h, overflow)
        select(l, zero, l, overflow)
    oh <<= h
    ol <<= l


@func
def _dd_div(oh, ol, ah, al, bh, bl):
    q = Reg(DT.float)
    zero = Reg(DT.float)
    ph = Reg(DT.float)
    pl = Reg(DT.float)
    rh = Reg(DT.float)
    rl = Reg(DT.float)
    correction = Reg(DT.float)
    zero.fill(0.)
    q <<= ah / bh
    _dd_mul(ph, pl, bh, bl, q, zero)
    ph <<= -ph
    pl <<= -pl
    _dd_add(rh, rl, ah, al, ph, pl)
    correction <<= rh + rl
    correction <<= correction / bh
    _dd_add(oh, ol, q, zero, correction, zero)


@func
def _dd_exp_negative(oh, ol, xh, xl):
    # x <= 0. The polynomial is only evaluated on the proved bounded branch.
    h = Reg(DT.float)
    l = Reg(DT.float)
    zh = Reg(DT.float)
    zl = Reg(DT.float)
    fallback = Reg(DT.float)
    floor = Reg(DT.float)
    zero = Reg(DT.float)
    outside = MaskReg(DT.float)
    floor.fill(-80.)
    zero.fill(0.)
    compare(outside, xh, -80., CompareMode.LT)
    select(zh, floor, xh, outside)
    select(zl, zero, xl, outside)
    zh <<= zh * .00390625
    zl <<= zl * .00390625
    h.fill(_EXP_FIRST[0])
    l.fill(_EXP_FIRST[1])
    for high, low in _EXP_COEFFICIENTS:
        _dd_mul(h, l, h, l, zh, zl, protect_overflow=False)
        _dd_add_constant(h, l, h, l, high, low)
    for iteration in range(8):
        _dd_mul(h, l, h, l, h, l, protect_overflow=False)
    fallback <<= xh.exp()
    select(oh, fallback, h, outside)
    select(ol, zero, l, outside)


@func
def _dd_softplus_sigmoid(sh, sl, dh, dl, uh, ul):
    zero = Reg(DT.float)
    one = Reg(DT.float)
    two = Reg(DT.float)
    xh = Reg(DT.float)
    xl = Reg(DT.float)
    eh = Reg(DT.float)
    el = Reg(DT.float)
    bh = Reg(DT.float)
    bl = Reg(DT.float)
    nh = Reg(DT.float)
    nl = Reg(DT.float)
    zh = Reg(DT.float)
    zl = Reg(DT.float)
    zz_h = Reg(DT.float)
    zz_l = Reg(DT.float)
    ph = Reg(DT.float)
    pl = Reg(DT.float)
    positive = MaskReg(DT.float)
    upper = MaskReg(DT.float)
    zero.fill(0.)
    one.fill(1.)
    two.fill(2.)
    compare(positive, uh, 0., CompareMode.GE)
    compare(upper, uh, 20., CompareMode.GT)
    xh <<= -uh
    xl <<= -ul
    select(xh, xh, uh, positive)
    select(xl, xl, ul, positive)
    _dd_exp_negative(eh, el, xh, xl)
    _dd_add(bh, bl, one, zero, eh, el)
    select(nh, one, eh, positive)
    select(nl, zero, el, positive)
    _dd_div(dh, dl, nh, nl, bh, bl)
    select(dh, one, dh, upper)
    select(dl, zero, dl, upper)
    # log(1+e) = 2*atanh(e/(2+e)); |e/(2+e)| <= 1/3.
    _dd_add(bh, bl, two, zero, eh, el)
    _dd_div(zh, zl, eh, el, bh, bl)
    _dd_mul(zz_h, zz_l, zh, zl, zh, zl, protect_overflow=False)
    ph.fill(_LOG_FIRST[0])
    pl.fill(_LOG_FIRST[1])
    for high, low in _LOG_COEFFICIENTS:
        _dd_mul(ph, pl, ph, pl, zz_h, zz_l, protect_overflow=False)
        _dd_add_constant(ph, pl, ph, pl, high, low)
    _dd_mul(ph, pl, ph, pl, zh, zl, protect_overflow=False)
    ph <<= ph * 2.
    pl <<= pl * 2.
    select(nh, uh, zero, positive)
    select(nl, ul, zero, positive)
    _dd_add(sh, sl, ph, pl, nh, nl)
    select(sh, uh, sh, upper)
    select(sl, ul, sl, upper)


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
        forward_overflow = MaskReg(DT.float)
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
            # D-PM-55: preserve the unchanged forward's FP32 overflow boundary.
            # Use its exact two64 cadd / add / epsilon operation order.
            ph <<= x0 * x0
            qh <<= x1 * x1
            cadd(denominator, ph)
            cadd(numerator, qh)
            temp <<= denominator + numerator
            temp <<= temp + 1.e-6
            gather(temp, temp, index0)
            compare(forward_overflow, temp, 3.4028234663852886e38, CompareMode.GT)
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
                temp.fill(0.)
                select(result, temp, result, forward_overflow)
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
                      destination: Tensor, terms: Tensor, partial: Tensor, rows: Var):
        decay = Reg(DT.float)
        narrow = Reg(DT.bfloat16)
        if a_dtype == bf16:
            ub_to_reg_single(narrow, avec[0, 0:1])
            decay <<= narrow.astype(DT.float)
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
        uh = Reg(DT.float)
        ul = Reg(DT.float)
        sh = Reg(DT.float)
        sl = Reg(DT.float)
        dh = Reg(DT.float)
        dl = Reg(DT.float)
        ah = Reg(DT.float)
        al = Reg(DT.float)
        bh = Reg(DT.float)
        bl = Reg(DT.float)
        result = Reg(DT.float)
        zero = Reg(DT.float)
        zero.fill(0.)
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
                    _dd_add(uh, ul, value, zero, bias0, zero)
                else:
                    _dd_add(uh, ul, value, zero, bias1, zero)
                _dd_softplus_sigmoid(sh, sl, dh, dl, uh, ul)
                _dd_mul(ah, al, gy, zero, sh, sl)
                _dd_mul(bh, bl, gy, zero, dh, dl)
                # Preserve both components until the fixed global reduction.
                terms[0, row*128+half*64:row*128+half*64+64] <<= ah
                terms[0, 4096+row*128+half*64:4096+row*128+half*64+64] <<= al
                terms[0, 8192+row*128+half*64:8192+row*128+half*64+64] <<= bh
                terms[0, 12288+row*128+half*64:12288+row*128+half*64+64] <<= bl
                _dd_mul(dh, dl, bh, bl, decay, zero)
                result <<= dh + dl
                if raw_dtype == bf16:
                    narrow <<= result.astype(DT.bfloat16, config)
                    reg_to_ub_downsample(destination[0, row*128+half*64:row*128+half*64+64], narrow, mask=full)
                else:
                    destination[0, row*128+half*64:row*128+half*64+64] <<= result
        for row in range(rows, 32):
            for part in unroll(4):
                for half in unroll(2):
                    terms[0, part*4096+row*128+half*64:part*4096+row*128+half*64+64] <<= zero
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        for level in unroll(5):
            for pair in range(32 // (2 << level)):
                for part in unroll(2):
                    for half in unroll(2):
                        ah <<= terms[0, part*8192+pair*(256 << level)+half*64:part*8192+pair*(256 << level)+half*64+64]
                        al <<= terms[0, part*8192+4096+pair*(256 << level)+half*64:part*8192+4096+pair*(256 << level)+half*64+64]
                        bh <<= terms[0, part*8192+pair*(256 << level)+(128 << level)+half*64:part*8192+pair*(256 << level)+(128 << level)+half*64+64]
                        bl <<= terms[0, part*8192+4096+pair*(256 << level)+(128 << level)+half*64:part*8192+4096+pair*(256 << level)+(128 << level)+half*64+64]
                        _dd_add(ah, al, ah, al, bh, bl)
                        terms[0, part*8192+pair*(256 << level)+half*64:part*8192+pair*(256 << level)+half*64+64] <<= ah
                        terms[0, part*8192+4096+pair*(256 << level)+half*64:part*8192+4096+pair*(256 << level)+half*64+64] <<= al
            vf_barrier(VfPipe.STORE, VfPipe.LOAD)
        for part in unroll(4):
            for half in unroll(2):
                result <<= terms[0, part*4096+half*64:part*4096+half*64+64]
                partial[0, part*128+half*64:part*128+half*64+64] <<= result
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
        terms = Tensor(DT.float, [1, 16384], Position.UB)
        partial = Tensor(DT.float, [1, 512], Position.UB)
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
                contributions(incoming, gradient, avec, bvec, outgoing, terms, partial, Var(rows))
                ub_to_gm_pad(destination[0, offset:offset+span], outgoing, n_burst=rows,
                             burst_len_element=128, src_stride=0, dst_stride_element=Channels-128)
                ub_to_gm_pad(partials[0, work*512:work*512+512], partial, n_burst=1,
                             burst_len_element=512, src_stride=0, dst_stride_element=0)
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
        ah = Reg(DT.float)
        al = Reg(DT.float)
        bh = Reg(DT.float)
        bl = Reg(DT.float)
        for part in unroll(2):
            for half in unroll(2):
                ah <<= state[0, part*256+half*64:part*256+half*64+64]
                al <<= state[0, part*256+128+half*64:part*256+128+half*64+64]
                for row in range(rows):
                    bh <<= partials[0, row*512+part*256+half*64:row*512+part*256+half*64+64]
                    bl <<= partials[0, row*512+part*256+128+half*64:row*512+part*256+128+half*64+64]
                    _dd_add(ah, al, ah, al, bh, bl)
                state[0, part*256+half*64:part*256+half*64+64] <<= ah
                state[0, part*256+128+half*64:part*256+128+half*64+64] <<= al
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @vf()
    def finish(state: Tensor, avec: Tensor, aout: Tensor, bout: Tensor, slot: Var):
        ah = Reg(DT.float)
        al = Reg(DT.float)
        bh = Reg(DT.float)
        bl = Reg(DT.float)
        result = Reg(DT.float)
        decay = Reg(DT.float)
        zero = Reg(DT.float)
        narrow = Reg(DT.bfloat16)
        signed_index = Reg(DT.int)
        index = Reg(DT.uint32)
        partner = Reg(DT.uint32)
        step = Reg(DT.uint32)
        full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
        config = CastConfig(round_mode=RoundMode.TO_EVEN)
        zero.fill(0.)
        if a_dtype == bf16:
            ub_to_reg_single(narrow, avec[0, slot:slot+1])
            decay <<= narrow.astype(DT.float)
        else:
            ub_to_reg_single(decay, avec[0, slot:slot+1])
        decay <<= decay.exp()
        decay <<= -decay
        ah <<= state[0, 0:64]
        al <<= state[0, 128:192]
        bh <<= state[0, 64:128]
        bl <<= state[0, 192:256]
        _dd_add(ah, al, ah, al, bh, bl)
        signed_index.arange(0)
        index <<= signed_index.reinterpret(DT.uint32)
        for level in unroll(6):
            step.fill(1 << level)
            vxor(partner, index, step)
            gather(bh, ah, partner)
            gather(bl, al, partner)
            _dd_add(ah, al, ah, al, bh, bl)
        _dd_mul(ah, al, ah, al, decay, zero)
        result <<= ah + al
        if a_dtype == bf16:
            narrow <<= result.astype(DT.bfloat16, config)
            reg_to_ub_single(aout[0, slot:slot+1], narrow)
        else:
            reg_to_ub_single(aout[0, slot:slot+1], result)
        for half in unroll(2):
            ah <<= state[0, 256+half*64:320+half*64]
            al <<= state[0, 384+half*64:448+half*64]
            _dd_mul(ah, al, ah, al, decay, zero)
            result <<= ah + al
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
                    for tile in range(CeilDiv(Chunks, 8)):
                        start = Var(tile*8)
                        rows = Min(8, Chunks-start)
                        offset = Var((head*Chunks+start)*512)
                        count = Var(rows*512)
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
    def differentiate(source: Tensor, probability: Tensor, sensitivity: Tensor, destination: Tensor, count: Var):
        s = Reg(DT.float)
        x = Reg(DT.float)
        gy = Reg(DT.float)
        eh = Reg(DT.float)
        el = Reg(DT.float)
        dh = Reg(DT.float)
        dl = Reg(DT.float)
        nh = Reg(DT.float)
        nl = Reg(DT.float)
        result = Reg(DT.float)
        one = Reg(DT.float)
        zero = Reg(DT.float)
        saturated_zero = MaskReg(DT.float)
        saturated_one = MaskReg(DT.float)
        one.fill(1.)
        zero.fill(0.)
        packed = Reg(DT.bfloat16)
        full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
        config = CastConfig(round_mode=RoundMode.TO_EVEN)
        for index in range(count / 64):
            if raw_dtype == bf16:
                x <<= source[0, index*64:index*64+64].unpack()
            else:
                x <<= source[0, index*64:index*64+64]
            s <<= probability[0, index*64:index*64+64]
            gy <<= sensitivity[0, index*64:index*64+64]
            compare(saturated_zero, s, 0., CompareMode.EQ)
            compare(saturated_one, s, 1., CompareMode.EQ)
            x <<= x.abs()
            x <<= -x
            _dd_exp_negative(eh, el, x, zero)
            _dd_add(dh, dl, one, zero, eh, el)
            _dd_mul(dh, dl, dh, dl, dh, dl)
            _dd_mul(nh, nl, gy, zero, eh, el)
            _dd_div(nh, nl, nh, nl, dh, dl)
            result <<= nh + nl
            # Only the actual saved forward result decides saturation.
            select(result, zero, result, saturated_zero)
            select(result, zero, result, saturated_one)
            if raw_dtype == bf16:
                packed <<= result.astype(DT.bfloat16, config)
                reg_to_ub_downsample(destination[0, index*64:index*64+64], packed, mask=full)
            else:
                destination[0, index*64:index*64+64] <<= result
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def beta_backward(source: GM[raw_dtype, (1, 'N')], probability: GM[f32, (1, 'N')], sensitivity: GM[f32, (1, 'N')],
                      destination: GM[raw_dtype, (1, 'N')], N: i32):
        source_ub = Tensor(raw_dtype, [1, 4096], Position.UB)
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
                gm_to_ub_nd_dma(source_ub, source[0, offset:offset+count], [1], [1], [count],
                                loop_right_pad=[padded-count], constant_value=0., fence='mte2')
                gm_to_ub_nd_dma(probability_ub, probability[0, offset:offset+count], [1], [1], [count],
                                loop_right_pad=[padded-count], constant_value=0., fence='mte2')
                gm_to_ub_nd_dma(gradient, sensitivity[0, offset:offset+count], [1], [1], [count],
                                loop_right_pad=[padded-count], constant_value=0., fence='mte2')
                differentiate(source_ub, probability_ub, gradient, outgoing, Var(padded))
                ub_to_gm_pad(destination[0, offset:offset+count], outgoing, n_burst=1,
                             burst_len_element=count, src_stride=0, dst_stride_element=0)
        return destination

    beta_backward.name = name
    return beta_backward
