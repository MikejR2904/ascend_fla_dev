"""BF16 cube output products, FP32 accumulation, direct BF16 GM output."""
from ascriptor.a5 import *


@vf()
def cast_rows(source: Tensor, out: Tensor, rows: Var, halves: Var):
    r = Reg(DT.float)
    b = Reg(DT.bfloat16)
    full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
    cfg = CastConfig(round_mode=RoundMode.TO_EVEN)
    for rr in range(rows):
        for hh in range(halves):
            r <<= source[rr:rr+1, hh*64:hh*64+64]
            b <<= r.astype(DT.bfloat16, cfg)
            reg_to_ub_downsample(out[rr:rr+1, hh*64:hh*64+64], b, mask=full)


@vf()
def weighted_query(q: Tensor, gate: Tensor, out: Tensor):
    r = Reg(DT.float)
    g = Reg(DT.float)
    b = Reg(DT.bfloat16)
    full = MaskReg(DT.bfloat16, init_mode=MaskType.ALL)
    cfg = CastConfig(round_mode=RoundMode.TO_EVEN)
    for rr in range(32):
        for hh in unroll(2):
            r <<= q[rr:rr+1, hh*64:hh*64+64]
            g <<= gate[rr:rr+1, hh*64:hh*64+64]
            g <<= g.exp()
            r <<= r * g
            b <<= r.astype(DT.bfloat16, cfg)
            reg_to_ub_downsample(out[rr:rr+1, hh*64:hh*64+64], b, mask=full)


@kernel(mode='mix')
def pkda_bf16_output(
    qn: GM[f32, ("B", "N", "H", 64, 128)],
    gc: GM[f32, ("B", "N", "H", 64, 128)],
    score: GM[f32, ("B", "N", "H", 64, 64)],
    states: GM[f32, ("B", "N", "H", 128, 128)],
    delta: GM[f32, ("B", "N", "H", 64, 128)],
    o: GM[bf16, ("B", "T", "H", 128)],
    B: i32, T: i32, H: i32, N: i32,
):
    qu = Tensor(DT.float, [32, 128], Position.UB)
    gu = Tensor(DT.float, [32, 128], Position.UB)
    au = Tensor(DT.float, [32, 64], Position.UB)
    su = Tensor(DT.float, [64, 128], Position.UB)
    du = Tensor(DT.float, [32, 128], Position.UB)
    qb = Tensor(DT.bfloat16, [32, 128], Position.UB)
    ab = Tensor(DT.bfloat16, [32, 64], Position.UB)
    sb = Tensor(DT.bfloat16, [64, 128], Position.UB)
    db = Tensor(DT.bfloat16, [32, 128], Position.UB)
    middle = Tensor(DT.float, [32, 128], Position.UB)
    out = Tensor(DT.bfloat16, [32, 128], Position.UB)
    ql = Tensor(DT.bfloat16, [64, 128], Position.L1)
    al = Tensor(DT.bfloat16, [64, 64], Position.L1)
    sl = Tensor(DT.bfloat16, [128, 128], Position.L1)
    dl = Tensor(DT.bfloat16, [64, 128], Position.L1)
    product = Tensor(DT.float, [64, 128], Position.L0C)
    q_ready = VcMutex(0, guards=ql, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    a_ready = VcMutex(1, guards=al, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    s_ready = VcMutex(2, guards=sl, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    d_ready = VcMutex(3, guards=dl, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
    result_ready = CvMutex(4, guards=middle, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)
    per = CeilDiv(B*N*H, GetCubeNum())
    begin = Var(per*GetCubeIdx())
    end = Min(begin+per, B*N*H)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item%H)
            cc = Var((item//H)%N)
            bb = Var(item//(N*H))
            rr = Var(GetSubBlockIdx()*32)
            sr = Var(GetSubBlockIdx()*64)
            q_ready.lock()
            qu[:, :] <<= qn[bb, cc, hh, rr:rr+32, :]
            gu[:, :] <<= gc[bb, cc, hh, rr:rr+32, :]
            weighted_query(qu, gu, qb)
            ql[rr:rr+32, :] <<= qb[:, :]
            q_ready.ready()
            a_ready.lock()
            au[:, :] <<= score[bb, cc, hh, rr:rr+32, :]
            cast_rows(au, ab, 32, 1)
            al[rr:rr+32, :] <<= ab[:, :]
            a_ready.ready()
            s_ready.lock()
            su[:, :] <<= states[bb, cc, hh, sr:sr+64, :]
            cast_rows(su, sb, 64, 2)
            sl[sr:sr+64, :] <<= sb[:, :]
            s_ready.ready()
            d_ready.lock()
            du[:, :] <<= delta[bb, cc, hh, rr:rr+32, :]
            cast_rows(du, db, 32, 2)
            dl[rr:rr+32, :] <<= db[:, :]
            d_ready.ready()

            q_ready.wait()
            s_ready.wait()
            matmul(product, ql, sl.T, m=64, n=128, k=128)
            a_ready.wait()
            d_ready.wait()
            matmul(product, al, dl.T, m=64, n=128, k=64, is_init=False)
            result_ready.lock()
            middle[:, :] <<= product[:, :]
            result_ready.ready()
            q_ready.free()
            a_ready.free()
            s_ready.free()
            d_ready.free()

            result_ready.wait()
            cast_rows(middle, out, 32, 2)
            result_ready.free()
            for ii in range(32):
                tt = Var(cc*64+rr+ii)
                if tt < T:
                    o[bb, tt:tt+1, hh, :] <<= out[ii:ii+1, :]
    return o
