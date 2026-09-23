"""GDN-2 recurrent BACKWARD kernel (A2 / c220). Reverse recurrence validated vs
torch autograd (bwd_golden). Phase-2-only: per-step forward states are provided as the `states` input
(produced by the training forward); the reverse recurrence runs directly. 2 full [K,V] UB tiles + GM checkpoint.
All primitives validated on a2 silicon."""
from ascriptor.a2 import *

K = 128
V = 128
GROUP = 64
NGROUP = V // GROUP
ROWBLK = V // 8
QK_EPS = 1e-6
Q_SCALE = 1.0 / 11.313708498984761
FULL = (1 << 64) - 1


def _spread8(sp8, x_row):
    brcb(sp8, x_row[0:1, 0:K], repeat=K // 8, dst_blk_stride=1, dst_rep_stride=8)


def _rowscale(dst, src, sp8):
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(dst[0:K, s:s + GROUP], src[0:K, s:s + GROUP], sp8, repeat=K, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=ROWBLK, src1_blk_stride=1, src1_rep_stride=ROWBLK,
            src2_blk_stride=0, src2_rep_stride=1)


def _mul_rowvec(dst, src, row):
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(dst[0:K, s:s + GROUP], src[0:K, s:s + GROUP], row[0:1, s:s + GROUP], repeat=K,
            count_per_rep=GROUP, dst_blk_stride=1, dst_rep_stride=ROWBLK, src1_blk_stride=1,
            src1_rep_stride=ROWBLK, src2_blk_stride=1, src2_rep_stride=0)


def _outer_acc(acc, sp8, row):
    """acc[k,v] += sp8[k] * row[0,v] via fused multiply-accumulate. muladddst
    (dst = dst + src1*src2) is bit-identical to the outer-product-then-add it
    replaces on a2, at 2 ops instead of 4 (no [K,V] outer-product temporary)."""
    for gi in range(NGROUP):
        s = gi * GROUP
        muladddst(acc[0:K, s:s + GROUP], sp8, row[0:1, s:s + GROUP], repeat=K, count_per_rep=GROUP,
                  dst_blk_stride=1, dst_rep_stride=ROWBLK, src1_blk_stride=0, src1_rep_stride=1,
                  src2_blk_stride=1, src2_rep_stride=0)


def _mul_full(dst, a, bt):
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(dst[0:K, s:s + GROUP], a[0:K, s:s + GROUP], bt[0:K, s:s + GROUP], repeat=K,
            count_per_rep=GROUP, dst_blk_stride=1, dst_rep_stride=ROWBLK, src1_blk_stride=1,
            src1_rep_stride=ROWBLK, src2_blk_stride=1, src2_rep_stride=ROWBLK)


def _reduce_rows(scr, out_row):
    add(scr[0:64, 0:V], scr[0:64, 0:V], scr[64:128, 0:V])
    add(scr[0:32, 0:V], scr[0:32, 0:V], scr[32:64, 0:V])
    add(scr[0:16, 0:V], scr[0:16, 0:V], scr[16:32, 0:V])
    add(scr[0:8, 0:V], scr[0:8, 0:V], scr[8:16, 0:V])
    add(scr[0:4, 0:V], scr[0:4, 0:V], scr[4:8, 0:V])
    add(scr[0:2, 0:V], scr[0:2, 0:V], scr[2:4, 0:V])
    add(scr[0:1, 0:V], scr[0:1, 0:V], scr[1:2, 0:V])
    out_row[0:1, 0:V] <<= scr[0:1, 0:V]


def _reduce_cols(scr, row_out):
    for half in (64, 32, 16, 8):
        add(scr[0:K, 0:half], scr[0:K, 0:half], scr[0:K, half:2 * half], repeat=K,
            count_per_rep=half, dst_blk_stride=1, dst_rep_stride=ROWBLK, src1_blk_stride=1,
            src1_rep_stride=ROWBLK, src2_blk_stride=1, src2_rep_stride=ROWBLK)
    cadd(row_out[0:1, 0:K], scr[0:K, 0:8], repeat=K, count_per_rep=8, src_blk_stride=1,
         src_rep_stride=ROWBLK, dst_rep_stride=1)


def _rsqrt_newton(y, x, tmp):
    rsqrt(y, x)
    mul(tmp, y, y); mul(tmp, tmp, x); muls(tmp, tmp, -0.5); adds(tmp, tmp, 1.5); mul(y, y, tmp)


def _row_sum(row_in, sq, red2, out1, square, other):
    # out1[0,0] = sum over K of (row_in^2 if square else row_in*other)
    if square:
        mul(sq, row_in[0:1, 0:K], row_in[0:1, 0:K])
    else:
        mul(sq, row_in[0:1, 0:K], other[0:1, 0:K])
    dup(red2, 0.0); dup(out1, 0.0)
    cadd(red2[0:1, 0:2], sq, repeat=2, src_blk_stride=1, src_rep_stride=8, count_per_rep=64)
    cadd(out1[0:1, 0:1], red2[0:1, 0:2], repeat=1, count_per_rep=2)


def _bcast64(scale88, elem1, block88):
    brcb(block88, elem1[0:1, 0:1], repeat=1, dst_blk_stride=1, dst_rep_stride=8)
    brcb(scale88, block88[0:1, 0:8], repeat=1, dst_blk_stride=1, dst_rep_stride=8)


def _scale_row(row, scale88):
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(row[0:1, s:s + GROUP], row[0:1, s:s + GROUP], scale88)


def _norm_r(dst, src, r, sq, red2, sy, block88, scale88, extra):
    # r[0,0] = 1/sqrt(sum(src^2)+eps) (kept for the l2norm backward);
    # dst = src * r [* Q_SCALE if extra]; src unchanged.
    _row_sum(src, sq, red2, sy, True, None)   # sy = sum(src^2) (kept as rsqrt input)
    adds(sy, sy, QK_EPS)
    _rsqrt_newton(r, sy, red2)                # r = rsqrt(sy); sy preserved for correct Newton
    if extra:
        muls(sy, r, Q_SCALE); _bcast64(scale88, sy, block88)   # reuse sy (sum no longer needed)
    else:
        _bcast64(scale88, r, block88)
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(dst[0:1, s:s + GROUP], src[0:1, s:s + GROUP], scale88)


def _l2n_bwd_r(dx, xraw, dxn, r, sq, red2, cc, prod, dotv, block88, scale88, cscl):
    # dx = r*dxn - (r^3 * sum(dxn*xraw)) * xraw, reusing the precomputed r.
    _row_sum(dxn, sq, red2, dotv, False, xraw)
    mul(cc, r, r); mul(cc, cc, r); mul(cc, cc, dotv)   # c = r^3 * dot
    _bcast64(scale88, r, block88)
    _bcast64(cscl, cc, block88)
    for gi in range(NGROUP):
        s = gi * GROUP
        mul(dx[0:1, s:s + GROUP], dxn[0:1, s:s + GROUP], scale88)
        mul(prod[0:1, s:s + GROUP], xraw[0:1, s:s + GROUP], cscl)
        sub(dx[0:1, s:s + GROUP], dx[0:1, s:s + GROUP], prod[0:1, s:s + GROUP])


@kernel(mode="vec", block_dim=40)
def gdn2_recurrent_bwd_kernel(
    q: GM[f32, ("B", "T", "H", 128)], k: GM[f32, ("B", "T", "H", 128)],
    v: GM[f32, ("B", "T", "H", 128)], g: GM[f32, ("B", "T", "H", 128)],
    erase_gate: GM[f32, ("B", "T", "H", 128)], w: GM[f32, ("B", "T", "H", 128)],
    initial_state: GM[f32, ("B", "H", 128, 128)],
    dout: GM[f32, ("B", "T", "H", 128)], dfinal: GM[f32, ("B", "H", 128, 128)],
    states: GM[f32, ("B", "H", "TP1", 128, 128)],
    delta_ckpt: GM[f32, ("B", "T", "H", 128)],
    dq: GM[f32, ("B", "T", "H", 128)], dk: GM[f32, ("B", "T", "H", 128)],
    dv: GM[f32, ("B", "T", "H", 128)], dg: GM[f32, ("B", "T", "H", 128)],
    db: GM[f32, ("B", "T", "H", 128)], dw: GM[f32, ("B", "T", "H", 128)],
    dh0: GM[f32, ("B", "H", 128, 128)], B: i32, T: i32, H: i32,
):
    A = Tensor(DT.float, [K, V], Position.UB)
    Bt = Tensor(DT.float, [K, V], Position.UB)
    qraw = Tensor(DT.float, [1, K], Position.UB); kraw = Tensor(DT.float, [1, K], Position.UB)
    qn = Tensor(DT.float, [1, K], Position.UB); kn = Tensor(DT.float, [1, K], Position.UB)
    vr = Tensor(DT.float, [1, V], Position.UB); wr = Tensor(DT.float, [1, V], Position.UB)
    gr = Tensor(DT.float, [1, K], Position.UB); br = Tensor(DT.float, [1, K], Position.UB)
    dor = Tensor(DT.float, [1, V], Position.UB)
    eg = Tensor(DT.float, [1, K], Position.UB); bk = Tensor(DT.float, [1, K], Position.UB)
    sp8 = Tensor(DT.float, [K, 8], Position.UB)
    sq = Tensor(DT.float, [1, K], Position.UB); red2 = Tensor(DT.float, [1, 64], Position.UB)
    sy = Tensor(DT.float, [1, 64], Position.UB); cc = Tensor(DT.float, [1, 64], Position.UB)
    rq = Tensor(DT.float, [1, 64], Position.UB); rk = Tensor(DT.float, [1, 64], Position.UB)
    dotv = Tensor(DT.float, [1, 64], Position.UB); prod = Tensor(DT.float, [1, K], Position.UB)
    block88 = Tensor(DT.float, [8, 8], Position.UB); scale88 = Tensor(DT.float, [8, 8], Position.UB)
    cscl = Tensor(DT.float, [8, 8], Position.UB)
    erase = Tensor(DT.float, [1, V], Position.UB); delta = Tensor(DT.float, [1, V], Position.UB)
    ddelta = Tensor(DT.float, [1, V], Position.UB); derase = Tensor(DT.float, [1, V], Position.UB)
    dqn = Tensor(DT.float, [1, K], Position.UB); dkn = Tensor(DT.float, [1, K], Position.UB)
    dbk = Tensor(DT.float, [1, K], Position.UB); dgr = Tensor(DT.float, [1, K], Position.UB)
    dxq = Tensor(DT.float, [1, K], Position.UB); dxk = Tensor(DT.float, [1, K], Position.UB)

    work = B * H
    per = CeilDiv(work, GetVecNum())
    wb = Var(per * GetVecIdx()); we = Min(wb + per, work)
    with auto_sync():
        set_mask(FULL, FULL)
        for wk in range(wb, we):
            h = Var(wk % H); bi = Var(wk // H)
            # PHASE 2: reverse
            A[0:K, 0:V] <<= dfinal[bi, h, 0:K, 0:V]
            for tr in range(T):
                t = Var(T - 1 - tr)
                qraw[0:1, 0:K] <<= q[bi, t, h, 0:K]; kraw[0:1, 0:K] <<= k[bi, t, h, 0:K]
                vr[0:1, 0:V] <<= v[bi, t, h, 0:V]; gr[0:1, 0:K] <<= g[bi, t, h, 0:K]
                br[0:1, 0:K] <<= erase_gate[bi, t, h, 0:K]; wr[0:1, 0:V] <<= w[bi, t, h, 0:V]
                dor[0:1, 0:V] <<= dout[bi, t, h, 0:V]
                _norm_r(qn, qraw, rq, sq, red2, sy, block88, scale88, True)
                _norm_r(kn, kraw, rk, sq, red2, sy, block88, scale88, False)
                exp(eg, gr); mul(bk, br, kn)
                # dqn = S_new @ do
                Bt[0:K, 0:V] <<= states[bi, h, t + 1, 0:K, 0:V]
                _mul_rowvec(Bt, Bt, dor); _reduce_cols(Bt, dqn)
                # dS += qn ^ do
                _spread8(sp8, qn); _outer_acc(A, sp8, dor)
                # ddelta = kn @ dS  (reduce over K)
                Bt[0:K, 0:V] <<= A[0:K, 0:V]; _spread8(sp8, kn); _rowscale(Bt, Bt, sp8)
                _reduce_rows(Bt, ddelta)
                mul(prod, ddelta, wr); dv[bi, t, h, 0:V] <<= prod[0:1, 0:V]
                mul(prod, ddelta, vr); dw[bi, t, h, 0:V] <<= prod[0:1, 0:V]
                muls(derase, ddelta, -1.0)
                # delta from the forward checkpoint (was: reload states[t], apply
                # eg+bk rowscales and a row-reduce to recompute erase, then wv-erase)
                delta[0:1, 0:V] <<= delta_ckpt[bi, t, h, 0:V]
                Bt[0:K, 0:V] <<= A[0:K, 0:V]; _mul_rowvec(Bt, Bt, delta); _reduce_cols(Bt, dkn)
                # dbk = S_dec @ derase
                Bt[0:K, 0:V] <<= states[bi, h, t, 0:K, 0:V]; _spread8(sp8, eg); _rowscale(Bt, Bt, sp8)
                _mul_rowvec(Bt, Bt, derase); _reduce_cols(Bt, dbk)
                # dS += bk ^ derase
                _spread8(sp8, bk); _outer_acc(A, sp8, derase)
                mul(prod, dbk, kn); db[bi, t, h, 0:K] <<= prod[0:1, 0:K]
                mul(prod, dbk, br); add(dkn, dkn, prod)
                # dg = reduce_v(dS * S_prev) * eg
                Bt[0:K, 0:V] <<= states[bi, h, t, 0:K, 0:V]; _mul_full(Bt, Bt, A); _reduce_cols(Bt, dgr)
                mul(dgr, dgr, eg); dg[bi, t, h, 0:K] <<= dgr[0:1, 0:K]
                # dS *= eg
                _spread8(sp8, eg); _rowscale(A, A, sp8)
                # dq, dk via l2norm backward (dqn scaled by Q_SCALE)
                muls(dqn, dqn, Q_SCALE)
                _l2n_bwd_r(dxq, qraw, dqn, rq, sq, red2, cc, prod, dotv, block88, scale88, cscl)
                _l2n_bwd_r(dxk, kraw, dkn, rk, sq, red2, cc, prod, dotv, block88, scale88, cscl)
                dq[bi, t, h, 0:K] <<= dxq[0:1, 0:K]
                dk[bi, t, h, 0:K] <<= dxk[0:1, 0:K]
            dh0[bi, h, 0:K, 0:V] <<= A[0:K, 0:V]
    return dq, dk, dv, dg, db, dw, dh0
