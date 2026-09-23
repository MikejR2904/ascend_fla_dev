"""Training GDN-2 forward with per-step state checkpoints (for the backward).
One-launch GDN-2 recurrent forward for short inference sequences (A2 / c220).

Port of the a5 unit into the A2 tensor-vector model. The a5 source used the
``@vf`` register value-function model (c310-only); A2 has no ``@vf``, so this
rewrite uses the tensor-vector free functions (``exp``/``mul``/``add``/``brcb``/
``cadd``) documented in ``docs/api/a2-vectors.md`` and validated on b3 by the
``a2_reduce_broadcast`` example, whose exact reduce/broadcast idiom is reused.

State is kept as ``state[K, V]`` (K-major, matching the GM ``initial_state`` /
``final_state`` layout, so no transpose is needed). The vector unit works on
64 FP32 lanes per repeat, so V=128 is handled in two 64-lane groups. A per-key
scalar is broadcast to a 64-lane block with two ``brcb`` stages and reused for
both groups. ``brcb`` reads a full 8-element block from its source, so the four
vectors it broadcasts from (q/k/eg/bk) are padded by 8 and zero-initialised.
Reductions over K are halving tree-adds over the K rows.
"""

from ascriptor.a2 import *

HEAD_DIM = 128
VALUE_DIM = 128
PAD = 192                   # brcb reads an 8-block from any k; 64-multiple for full-width ops
GROUP = 64
NGROUP = VALUE_DIM // GROUP
ROWBLK = VALUE_DIM // 8      # a [128,128] tile row spans this many 32-byte blocks
T_MAX = 16
QK_EPS = 1e-6
Q_SCALE = 1.0 / 11.313708498984761
FULL_MASK = (1 << 64) - 1


def _bcast64(scale88, src_elem, block88):
    """Broadcast a single UB element ``src_elem`` to 64 lanes in ``scale88``
    ([8,8]) via two brcb stages.

    NOTE (silicon TODO): ``brcb`` requires a 32-byte-aligned source. For the
    per-key calls the source sits at offset ``kk*4`` (not 32B aligned); the
    functional sim accepts this but real hardware raises an aivec alignment
    exception. The fix is to pre-spread each vector so element k lands at a
    32B-aligned offset, then read the aligned block here. Sim+pipesim pass;
    aclnn silicon is blocked on this alignment restructure."""
    brcb(block88, src_elem, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
    brcb(scale88, block88[0:1, 0:8], repeat=1, dst_blk_stride=1, dst_rep_stride=8)


def _scale_row(row_slice, scale88):
    """Multiply a [1,VALUE_DIM] row in place by the 64-lane broadcast scalar."""
    for grp in range(NGROUP):
        s = grp * GROUP
        mul(row_slice[0:1, s:s + GROUP], row_slice[0:1, s:s + GROUP], scale88)


def _scale_into(dst_slice, src_slice, scale88):
    """dst = src * scale (per-row scalar), one 64-lane group at a time."""
    for grp in range(NGROUP):
        s = grp * GROUP
        mul(dst_slice[0:1, s:s + GROUP], src_slice[0:1, s:s + GROUP], scale88)


def _spread8(sp8, x_row):
    """Build the per-key spread ``sp8[k, 0:8] = x_row[0, k]`` for k in 0..127.

    A single ``brcb`` reads the whole row from offset 0 (32B aligned) and
    broadcasts each element into its own 8-lane block, so downstream reads of
    ``sp8[k]`` land at the 32B-aligned offset ``k*8``. ``brcb`` consumes 8
    source elements per repeat, so 128 keys take ``repeat=16``."""
    brcb(sp8, x_row[0:1, 0:HEAD_DIM], repeat=HEAD_DIM // 8, dst_blk_stride=1, dst_rep_stride=8)


def _rowscale(dst, src, sp8):
    """dst[k,v] = src[k,v] * value[k], where ``value[k]`` sits in sp8[k,0:8].

    Vectorized over all 128 rows in two 64-lane groups (a 64-lane group is one
    vector repeat, so the per-row scalar advances one sp8 row per repeat via
    ``src2_rep_stride=1``, reused across the group with ``src2_blk_stride=0``)."""
    for g in range(NGROUP):
        s = g * GROUP
        mul(dst[0:HEAD_DIM, s:s + GROUP], src[0:HEAD_DIM, s:s + GROUP], sp8,
            repeat=HEAD_DIM, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=ROWBLK,
            src1_blk_stride=1, src1_rep_stride=ROWBLK,
            src2_blk_stride=0, src2_rep_stride=1)


def _outer(dst, sp8, col_row):
    """dst[k,v] = value[k] * col_row[0, v]  (rank-1 outer product).

    src1 = sp8 broadcast per row (blk_stride 0, rep_stride 1); src2 = col_row
    broadcast down every row (rep_stride 0). dst rows stride the full tile."""
    for g in range(NGROUP):
        s = g * GROUP
        mul(dst[0:HEAD_DIM, s:s + GROUP], sp8, col_row[0:1, s:s + GROUP],
            repeat=HEAD_DIM, count_per_rep=GROUP,
            dst_blk_stride=1, dst_rep_stride=ROWBLK,
            src1_blk_stride=0, src1_rep_stride=1,
            src2_blk_stride=1, src2_rep_stride=0)


def _reduce_rows(scratch, out_row):
    """Sum scratch[0:128, 0:V] over the K rows into out_row[0:1,0:V]."""
    add(scratch[0:64, 0:VALUE_DIM], scratch[0:64, 0:VALUE_DIM], scratch[64:128, 0:VALUE_DIM])
    add(scratch[0:32, 0:VALUE_DIM], scratch[0:32, 0:VALUE_DIM], scratch[32:64, 0:VALUE_DIM])
    add(scratch[0:16, 0:VALUE_DIM], scratch[0:16, 0:VALUE_DIM], scratch[16:32, 0:VALUE_DIM])
    add(scratch[0:8, 0:VALUE_DIM], scratch[0:8, 0:VALUE_DIM], scratch[8:16, 0:VALUE_DIM])
    add(scratch[0:4, 0:VALUE_DIM], scratch[0:4, 0:VALUE_DIM], scratch[4:8, 0:VALUE_DIM])
    add(scratch[0:2, 0:VALUE_DIM], scratch[0:2, 0:VALUE_DIM], scratch[2:4, 0:VALUE_DIM])
    add(scratch[0:1, 0:VALUE_DIM], scratch[0:1, 0:VALUE_DIM], scratch[1:2, 0:VALUE_DIM])
    out_row[0:1, 0:VALUE_DIM] <<= scratch[0:1, 0:VALUE_DIM]


def _rsqrt_newton(y, x, tmp):
    """y = 1/sqrt(x) to ~fp32 accuracy: refine the hardware fast rsqrt with one
    Newton step y <- y*(1.5 - 0.5*x*y^2). The hardware rsqrt alone is only
    ~2^-11 relative, which overshoots the unit's 1e-4 relative-L2 gate."""
    rsqrt(y, x)
    mul(tmp, y, y)
    mul(tmp, tmp, x)
    muls(tmp, tmp, -0.5)
    adds(tmp, tmp, 1.5)
    mul(y, y, tmp)


def _l2_sum(sq_row, red2, sum8):
    """sum(sq_row[0:128]) into sum8[0,0] (sum8[0:8] all valid for a later brcb).

    ``cadd`` accumulates into its destination, so both reduction targets are
    zeroed first (as the a2_reduce_broadcast example does before each row)."""
    dup(red2, 0.0)
    dup(sum8, 0.0)
    cadd(red2[0:1, 0:2], sq_row, repeat=2, src_blk_stride=1,
         src_rep_stride=8, count_per_rep=64)
    cadd(sum8[0:1, 0:1], red2[0:1, 0:2], repeat=1, count_per_rep=2)


@kernel(mode="vec", block_dim=40)
def gdn2_fwd_states_kernel(
    q: GM[f32, ("B", "T", "H", 128)],
    k: GM[f32, ("B", "T", "H", 128)],
    v: GM[f32, ("B", "T", "H", 128)],
    g: GM[f32, ("B", "T", "H", 128)],
    erase_gate: GM[f32, ("B", "T", "H", 128)],
    w: GM[f32, ("B", "T", "H", 128)],
    initial_state: GM[f32, ("B", "H", 128, 128)],
    o: GM[f32, ("B", "T", "H", 128)],
    final_state: GM[f32, ("B", "H", 128, 128)],
    states: GM[f32, ("B", "H", "TP1", 128, 128)],
    delta_ckpt: GM[f32, ("B", "T", "H", 128)],
    B: i32,
    T: i32,
    H: i32,
    TP1: i32,
):
    state = Tensor(DT.float, [HEAD_DIM, VALUE_DIM], Position.UB)    # [k, v]
    scratch = Tensor(DT.float, [HEAD_DIM, VALUE_DIM], Position.UB)  # reduction workspace

    q_row = Tensor(DT.float, [1, PAD], Position.UB)   # brcb source -> padded
    k_row = Tensor(DT.float, [1, PAD], Position.UB)   # brcb source -> padded
    eg_row = Tensor(DT.float, [1, PAD], Position.UB)  # exp(g), brcb source
    bk_row = Tensor(DT.float, [1, PAD], Position.UB)  # b * k_norm, brcb source
    v_row = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    g_row = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    b_row = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    w_row = Tensor(DT.float, [1, VALUE_DIM], Position.UB)

    sq_row = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    red2 = Tensor(DT.float, [1, 64], Position.UB)
    sum8 = Tensor(DT.float, [1, 64], Position.UB)
    sy = Tensor(DT.float, [1, 64], Position.UB)   # refined 1/sqrt result
    block88 = Tensor(DT.float, [8, 8], Position.UB)
    scale88 = Tensor(DT.float, [8, 8], Position.UB)
    sp8 = Tensor(DT.float, [HEAD_DIM, 8], Position.UB)   # per-key spread (aligned)
    erase = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    delta = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    out_row = Tensor(DT.float, [1, VALUE_DIM], Position.UB)

    work_count = B * H
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        set_mask(FULL_MASK, FULL_MASK)
        # zero-init padded brcb sources and the sum tail once; per-token writes
        # only touch [0:128] / [0:1], leaving the block tails at 0.
        dup(q_row, 0.0)
        dup(k_row, 0.0)
        dup(eg_row, 0.0)
        dup(bk_row, 0.0)
        dup(sum8, 0.0)
        for work in range(work_begin, work_end):
            h_idx = Var(work % H)
            b_idx = Var(work // H)
            state[0:HEAD_DIM, 0:VALUE_DIM] <<= initial_state[
                b_idx, h_idx, 0:HEAD_DIM, 0:VALUE_DIM
            ]
            states[b_idx, h_idx, 0, 0:HEAD_DIM, 0:VALUE_DIM] <<= state[0:HEAD_DIM, 0:VALUE_DIM]

            for t in range(T):
                q_row[0:1, 0:HEAD_DIM] <<= q[b_idx, t:t + 1, h_idx, 0:HEAD_DIM]
                k_row[0:1, 0:HEAD_DIM] <<= k[b_idx, t:t + 1, h_idx, 0:HEAD_DIM]
                v_row[0:1, 0:VALUE_DIM] <<= v[b_idx, t:t + 1, h_idx, 0:VALUE_DIM]
                g_row[0:1, 0:HEAD_DIM] <<= g[b_idx, t:t + 1, h_idx, 0:HEAD_DIM]
                b_row[0:1, 0:HEAD_DIM] <<= erase_gate[b_idx, t:t + 1, h_idx, 0:HEAD_DIM]
                w_row[0:1, 0:VALUE_DIM] <<= w[b_idx, t:t + 1, h_idx, 0:VALUE_DIM]

                # ---- q normalize: q = q / sqrt(sum(q^2)+eps) * Q_SCALE ----
                # Do the eps/rsqrt/scale math on the [1,64] sum row (unary ops
                # misbehave on the [8,8] broadcast tile); broadcast afterwards.
                mul(sq_row, q_row[0:1, 0:HEAD_DIM], q_row[0:1, 0:HEAD_DIM])
                _l2_sum(sq_row, red2, sum8)
                adds(sum8, sum8, QK_EPS)
                _rsqrt_newton(sy, sum8, red2)
                muls(sy, sy, Q_SCALE)
                _bcast64(scale88, sy[0:1, 0:1], block88)
                _scale_row(q_row, scale88)

                # ---- k normalize: k = k / sqrt(sum(k^2)+eps) ----
                mul(sq_row, k_row[0:1, 0:HEAD_DIM], k_row[0:1, 0:HEAD_DIM])
                _l2_sum(sq_row, red2, sum8)
                adds(sum8, sum8, QK_EPS)
                _rsqrt_newton(sy, sum8, red2)
                _bcast64(scale88, sy[0:1, 0:1], block88)
                _scale_row(k_row, scale88)

                # ---- gates ----
                exp(eg_row[0:1, 0:HEAD_DIM], g_row)
                mul(bk_row[0:1, 0:HEAD_DIM], b_row, k_row[0:1, 0:HEAD_DIM])

                # ---- S1 = Diag(exp(g)) * S0 ; erase = (b*k_norm)^T @ S1 ----
                _spread8(sp8, eg_row)
                _rowscale(state, state, sp8)          # decay: state[k,:] *= eg[k]
                _spread8(sp8, bk_row)
                _rowscale(scratch, state, sp8)        # scratch[k,:] = state[k,:] * bk[k]
                _reduce_rows(scratch, erase)          # erase[v] = sum_k scratch[k,v]

                # ---- delta = w * v - erase ----
                mul(delta, w_row, v_row)
                sub(delta, delta, erase)
                # checkpoint delta so the backward reads it instead of recomputing
                # erase (a states load + 2 rowscales + a row-reduce) each step.
                delta_ckpt[b_idx, t:t + 1, h_idx, 0:VALUE_DIM] <<= delta[0:1, 0:VALUE_DIM]

                # ---- S2 = S1 + k_norm ^ delta ; o = q_norm^T @ S2 ----
                _spread8(sp8, k_row)
                _outer(scratch, sp8, delta)           # scratch[k,v] = k_norm[k] * delta[v]
                # state += k ^ delta, split by rows into full-width contiguous halves
                # (<=255 repeats, no strided-view stride hints needed).
                add(state[0:64, 0:VALUE_DIM], state[0:64, 0:VALUE_DIM], scratch[0:64, 0:VALUE_DIM])
                add(state[64:HEAD_DIM, 0:VALUE_DIM], state[64:HEAD_DIM, 0:VALUE_DIM],
                    scratch[64:HEAD_DIM, 0:VALUE_DIM])
                _spread8(sp8, q_row)
                _rowscale(scratch, state, sp8)        # scratch[k,:] = state[k,:] * q[k]
                _reduce_rows(scratch, out_row)        # o[v] = sum_k scratch[k,v]

                o[b_idx, t:t + 1, h_idx, 0:VALUE_DIM] <<= out_row[0:1, 0:VALUE_DIM]
                states[b_idx, h_idx, t + 1, 0:HEAD_DIM, 0:VALUE_DIM] <<= state[0:HEAD_DIM, 0:VALUE_DIM]

            final_state[b_idx, h_idx, 0:HEAD_DIM, 0:VALUE_DIM] <<= state[
                0:HEAD_DIM, 0:VALUE_DIM
            ]

    return o, final_state, states, delta_ckpt
