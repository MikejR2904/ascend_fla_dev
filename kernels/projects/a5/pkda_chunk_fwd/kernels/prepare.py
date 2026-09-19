"""ATK diagonal recurrence fused with KDA workspace preparation.

Pinned FLA naive semantics consume q/k as supplied: no hidden normalization.
Each vector participant owns complete heads so A carries across every chunk.
"""
from ascriptor.a5 import *

D = 128
C = 64
LOG_X = 0.4054651081081644  # ln(1.5); FP32 arithmetic in the VF.


@vf()
def zero_row(x: Tensor):
    z = RegList(DT.float, 2)
    z <<= 0.0
    x[0:1, 0:D] <<= z


@vf()
def prepare_row(q: Tensor, k: Tensor, v: Tensor, g: Tensor,
                ga: Tensor, ba: Tensor, beta: Tensor, center: Tensor,
                a: Tensor, p: Tensor, bk: Tensor, prefix: Tensor, scale: Var):
    qr = RegList(DT.float, 2)
    kr = RegList(DT.float, 2)
    vr = RegList(DT.float, 2)
    ar = RegList(DT.float, 2)
    tmp = RegList(DT.float, 2)
    weight = RegList(DT.float, 2)
    decay = RegList(DT.float, 2)
    pref = RegList(DT.float, 2)
    scalar_ga = Reg(DT.float)
    scalar_ba = Reg(DT.float)
    scalar_beta = Reg(DT.float)
    scalar_center = Reg(DT.float)
    scalar_ga <<= ga[0:1, 0:1].single()
    scalar_ga <<= scalar_ga.exp()
    scalar_ba <<= ba[0:1, 0:1].single()
    scalar_beta <<= beta[0:1, 0:1].single()
    scalar_center <<= center[0:1, 0:1].single()
    qr <<= q[0:1, 0:D]
    kr <<= k[0:1, 0:D]
    vr <<= v[0:1, 0:D]
    ar <<= a[0:1, 0:D]
    ar <<= ar * scalar_ga
    tmp <<= kr * kr
    tmp <<= tmp * scalar_ba
    ar <<= ar + tmp
    # Update A before using it to construct the bounded diagonal M.
    tmp <<= ar + 1e-6
    tmp <<= tmp.ln()
    tmp <<= tmp - scalar_center
    weight <<= tmp.abs()
    weight <<= weight + 1.0
    tmp <<= tmp / weight
    tmp <<= tmp * -LOG_X
    weight <<= tmp.exp()
    tmp <<= kr * weight
    p[0:1, 0:D] <<= tmp
    # Original k predicts; only p=k*M writes. Never replace both keys.
    tmp <<= kr * scalar_beta
    bk[0:1, 0:D] <<= tmp
    vr <<= vr * scalar_beta
    qr <<= qr * scale
    decay <<= g[0:1, 0:D]
    pref <<= prefix[0:1, 0:D]
    pref <<= pref + decay
    q[0:1, 0:D] <<= qr
    v[0:1, 0:D] <<= vr
    a[0:1, 0:D] <<= ar
    prefix[0:1, 0:D] <<= pref
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def pkda_chunk_prepare(
    q: GM[f32, ("B", "T", "H", 128)], k: GM[f32, ("B", "T", "H", 128)],
    v: GM[f32, ("B", "T", "H", 128)], g: GM[f32, ("B", "T", "H", 128)],
    g_atk: GM[f32, ("B", "T", "H")], beta_atk: GM[f32, ("B", "T", "H")],
    beta: GM[f32, ("B", "T", "H")], initial_A_state: GM[f32, ("B", "H", 128)],
    log_atk_scale: GM[f32, (1, "H")],
    qn: GM[f32, ("B", "N", "H", 64, 128)], kn: GM[f32, ("B", "N", "H", 64, 128)],
    gc: GM[f32, ("B", "N", "H", 64, 128)], bk: GM[f32, ("B", "N", "H", 64, 128)],
    wv: GM[f32, ("B", "N", "H", 64, 128)], final_A_state: GM[f32, ("B", "H", 128)],
    B: i32, T: i32, H: i32, N: i32, scale: f32,
):
    qu = Tensor(DT.float, [1, D], Position.UB)
    ku = Tensor(DT.float, [1, D], Position.UB)
    vu = Tensor(DT.float, [1, D], Position.UB)
    gu = Tensor(DT.float, [1, D], Position.UB)
    bu = Tensor(DT.float, [1, D], Position.UB)
    pu = Tensor(DT.float, [1, D], Position.UB)
    au = Tensor(DT.float, [1, D], Position.UB)
    pku = Tensor(DT.float, [1, D], Position.UB)
    gau = Tensor(DT.float, [1, 8], Position.UB)
    bau = Tensor(DT.float, [1, 8], Position.UB)
    betau = Tensor(DT.float, [1, 8], Position.UB)
    centeru = Tensor(DT.float, [1, 8], Position.UB)
    per = CeilDiv(B * H, GetVecNum())
    begin = Var(per * GetVecIdx())
    end = Min(begin + per, B * H)
    with auto_sync():
        for item in range(begin, end):
            hh = Var(item % H)
            bb = Var(item // H)
            au[:, :] <<= initial_A_state[bb, hh:hh+1, :]
            centeru[:, 0:1] <<= log_atk_scale[0:1, hh:hh+1]
            for cc in range(N):
                zero_row(pu)
                for ii in range(C):
                    tt = Var(cc * C + ii)
                    if tt < T:
                        qu[:, :] <<= q[bb, tt:tt+1, hh, :]
                        ku[:, :] <<= k[bb, tt:tt+1, hh, :]
                        vu[:, :] <<= v[bb, tt:tt+1, hh, :]
                        gu[:, :] <<= g[bb, tt:tt+1, hh, :]
                        gau[:, 0:1] <<= g_atk[bb, tt:tt+1, hh:hh+1]
                        bau[:, 0:1] <<= beta_atk[bb, tt:tt+1, hh:hh+1]
                        betau[:, 0:1] <<= beta[bb, tt:tt+1, hh:hh+1]
                        prepare_row(qu, ku, vu, gu, gau, bau, betau, centeru,
                                    au, pku, bu, pu, scale)
                    else:
                        zero_row(qu)
                        zero_row(pku)
                        zero_row(vu)
                        zero_row(bu)
                    qn[bb, cc, hh, ii:ii+1, :] <<= qu[:, :]
                    kn[bb, cc, hh, ii:ii+1, :] <<= pku[:, :]
                    gc[bb, cc, hh, ii:ii+1, :] <<= pu[:, :]
                    bk[bb, cc, hh, ii:ii+1, :] <<= bu[:, :]
                    wv[bb, cc, hh, ii:ii+1, :] <<= vu[:, :]
            final_A_state[bb, hh:hh+1, :] <<= au[:, :]
    return qn, kn, gc, bk, wv, final_A_state
