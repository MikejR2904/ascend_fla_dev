"""A2 blocked inverse of the unit lower-triangular chunk matrix: ``inv = (I - S)^-1`` (BF16 out).

Port of ascriptor ``kernels/projects/a5/kda_fwd/kernels/triangular_inverse.py`` (read-only upstream) to the A2
(c220) facade. ``S`` is the FP32 strictly-lower ``strict`` matrix from the scores kernel. The 64x64 matrix is
four 16x16 diagonal blocks ``D_i = (I - S_ii)^-1`` (forward substitution on the vector side) and six
off-diagonal blocks built on the cube, exactly the A5 schedule::

    X10 = D1 S10 D0            X21 = (D2 S21) D1          X32 = (D3 S32) D2
    X20 = (D2 S20) D0 + (D2 S21) X10
    X31 = (D3 S31) D1 + (D3 S32) X21
    X30 = (D3 S30) D0 + (D3 S31) X10 + (D3 S32) X20

What changes on A2, and why:

* No ``@vf``: the forward substitution is one ``muls`` + ``add`` per (row, earlier row) pair on a 16-wide UB
  row, driven by a scalar read of ``S[i, k]``, in the A5 accumulation order (``acc = 0 + p0 + p1 + ...``).
* No UB -> L1 on A2: the diagonal inverses reach the cube through two-slot GM workspaces, one ring per stage
  mutex (``VcMutex``, MTE3 -> MTE2), each sub-block taking its lock right before its own write; the
  off-diagonal ``S`` blocks are loaded by the cube straight from GM.
* No FP32 L0C -> L1 on c220 (the CCE backend has no spelling for it; quantising to BF16 would change the A5
  FP32 intermediate precision): each FP32 product is published through a GM workspace, with an explicit
  FIX -> MTE2 event, because autosync does not order a same-core GM read-after-write (pipesim reports it).
* A2-01 / library M10-081: every ``is_init=False`` FP32 accumulate is preceded by ``barrier(Pipe.M)``.
"""

from ascriptor.a2 import *

SIZE = 64

BLOCK = 16

#: rows of the per-chunk publication workspace: one 16-row region per published product
PUB_T1, PUB_X10, PUB_P21, PUB_X21, PUB_P32, PUB_T2, PUB_X20, PUB_P31, PUB_T3 = range(9)
PUB_REGIONS = 9


@func()
def _invert_diag(src_ub: Tensor, diag_ub: Tensor, tmp_ub: Tensor, s_val: Var, one: Var):
    dup(diag_ub[0:BLOCK, 0:BLOCK], 0.0, count=BLOCK * BLOCK)
    one.SetValueTo(diag_ub[0:1, 0:1])
    for i in range(1, BLOCK):
        for k in range(0, i):
            s_val.GetValueFrom(src_ub[i:i + 1, k:k + 1])
            muls(tmp_ub[0:1, 0:BLOCK], diag_ub[k:k + 1, 0:BLOCK], s_val, count=BLOCK)
            add(diag_ub[i:i + 1, 0:BLOCK], diag_ub[i:i + 1, 0:BLOCK], tmp_ub[0:1, 0:BLOCK], count=BLOCK)
        one.SetValueTo(diag_ub[i:i + 1, i:i + 1])


@func()
def _publish(l1_dst: Tensor, region: Tensor, l0c_src: Tensor, ev):
    region <<= l0c_src
    ev.set()
    ev.wait()
    l1_dst <<= region


@kernel()
def tril_inverse64_a2_kernel(
    a: GM[f32, ('B', 'HV', 'C', 64, 64)],
    inv: GM[bf16, ('B', 'HV', 'C', 64, 64)],
    B: i32,
    HV: i32,
    C: i32,
):
    d01_ws = GMBuff(DT.float, [2 * BLOCK, BLOCK], slots=2, name="diag01_ws")
    d23_ws = GMBuff(DT.float, [2 * BLOCK, BLOCK], slots=2, name="diag23_ws")
    pub_ws = GMBuff(DT.float, [PUB_REGIONS * BLOCK, BLOCK], slots=2, name="pub_ws")
    stage0 = VcMutex(0, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    stage1 = VcMutex(1, depth=2, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.MTE2)
    pub_ev = DEvent(Pipe.FIX, Pipe.MTE2)  # independent publishes can put two FIX sets in flight

    src_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    diag_a_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    diag_b_ub = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)
    tmp_ub = Tensor(DT.float, [1, BLOCK], Position.UB)
    diag_a_bf = Tensor(DT.bfloat16, [BLOCK, BLOCK], Position.UB)
    diag_b_bf = Tensor(DT.bfloat16, [BLOCK, BLOCK], Position.UB)
    zero_bf = Tensor(DT.bfloat16, [BLOCK, BLOCK], Position.UB)
    zero_f = Tensor(DT.float, [BLOCK, BLOCK], Position.UB)

    l1_d0 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_d1 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_d2 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_d3 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_s10 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_s20 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_s21 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_s30 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_s31 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_s32 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_t = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_x10 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_x20 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_x21 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_p21 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_p31 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l1_p32 = Tensor(DT.float, [BLOCK, BLOCK], Position.L1)
    l0c = DBuff(DT.float, [BLOCK, BLOCK], Position.L0C)

    total = B * HV * C
    per_core = CeilDiv(total, GetCubeNum())
    chunk_begin = Var(per_core * GetCubeIdx())
    chunk_end = Min(chunk_begin + per_core, total)
    beat = Var(0)
    l0c_cnt = Var(0)
    s_val = Var(0.0, dtype=DT.float)
    one = Var(1.0, dtype=DT.float)

    with auto_sync():
        with vec_scope():
            dup(zero_f[0:BLOCK, 0:BLOCK], 0.0, count=BLOCK * BLOCK)  # dup has no BF16 form on A2
            cast(zero_bf[0:BLOCK, 0:BLOCK], zero_f[0:BLOCK, 0:BLOCK], round_mode=RoundMode.TO_EVEN, count=BLOCK * BLOCK)

    for chunk in range(chunk_begin, chunk_end):
        with auto_sync():
            c_idx = Var(chunk % C)
            tmp = Var(chunk // C)
            hv_idx = Var(tmp % HV)
            b_idx = Var(tmp // HV)

            # ---- vector: D0 / D1 (stage 0) ----
            if GetSubBlockIdx() == 0:
                src_ub <<= a[b_idx, hv_idx, c_idx, 0:BLOCK, 0:BLOCK]
                _invert_diag(src_ub, diag_a_ub, tmp_ub, s_val, one)
                stage0.lock()
                d01_ws[beat][0:BLOCK, 0:BLOCK] <<= diag_a_ub[0:BLOCK, 0:BLOCK]
                stage0.ready()
            if GetSubBlockIdx() == 1:
                src_ub <<= a[b_idx, hv_idx, c_idx, BLOCK:2 * BLOCK, BLOCK:2 * BLOCK]
                _invert_diag(src_ub, diag_a_ub, tmp_ub, s_val, one)
                stage0.lock()
                d01_ws[beat][BLOCK:2 * BLOCK, 0:BLOCK] <<= diag_a_ub[0:BLOCK, 0:BLOCK]
                stage0.ready()

            # ---- vector: D2 / D3 (stage 1) and the diagonal / zero output blocks ----
            if GetSubBlockIdx() == 0:
                src_ub <<= a[b_idx, hv_idx, c_idx, 2 * BLOCK:3 * BLOCK, 2 * BLOCK:3 * BLOCK]
                _invert_diag(src_ub, diag_b_ub, tmp_ub, s_val, one)
                stage1.lock()
                d23_ws[beat][0:BLOCK, 0:BLOCK] <<= diag_b_ub[0:BLOCK, 0:BLOCK]
                stage1.ready()
                cast(diag_a_bf[0:BLOCK, 0:BLOCK], diag_a_ub[0:BLOCK, 0:BLOCK], round_mode=RoundMode.TO_EVEN, count=BLOCK * BLOCK)
                cast(diag_b_bf[0:BLOCK, 0:BLOCK], diag_b_ub[0:BLOCK, 0:BLOCK], round_mode=RoundMode.TO_EVEN, count=BLOCK * BLOCK)
                inv[b_idx, hv_idx, c_idx, 0:BLOCK, 0:BLOCK] <<= diag_a_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, 2 * BLOCK:3 * BLOCK, 2 * BLOCK:3 * BLOCK] <<= diag_b_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, 0:BLOCK, BLOCK:2 * BLOCK] <<= zero_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, 0:BLOCK, 2 * BLOCK:3 * BLOCK] <<= zero_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, 0:BLOCK, 3 * BLOCK:SIZE] <<= zero_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, BLOCK:2 * BLOCK, 2 * BLOCK:3 * BLOCK] <<= zero_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, BLOCK:2 * BLOCK, 3 * BLOCK:SIZE] <<= zero_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, 2 * BLOCK:3 * BLOCK, 3 * BLOCK:SIZE] <<= zero_bf[0:BLOCK, 0:BLOCK]
            if GetSubBlockIdx() == 1:
                src_ub <<= a[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, 3 * BLOCK:SIZE]
                _invert_diag(src_ub, diag_b_ub, tmp_ub, s_val, one)
                stage1.lock()
                d23_ws[beat][BLOCK:2 * BLOCK, 0:BLOCK] <<= diag_b_ub[0:BLOCK, 0:BLOCK]
                stage1.ready()
                cast(diag_a_bf[0:BLOCK, 0:BLOCK], diag_a_ub[0:BLOCK, 0:BLOCK], round_mode=RoundMode.TO_EVEN, count=BLOCK * BLOCK)
                cast(diag_b_bf[0:BLOCK, 0:BLOCK], diag_b_ub[0:BLOCK, 0:BLOCK], round_mode=RoundMode.TO_EVEN, count=BLOCK * BLOCK)
                inv[b_idx, hv_idx, c_idx, BLOCK:2 * BLOCK, BLOCK:2 * BLOCK] <<= diag_a_bf[0:BLOCK, 0:BLOCK]
                inv[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, 3 * BLOCK:SIZE] <<= diag_b_bf[0:BLOCK, 0:BLOCK]

            # ---- cube: off-diagonal S blocks come straight from GM ----
            l1_s10 <<= a[b_idx, hv_idx, c_idx, BLOCK:2 * BLOCK, 0:BLOCK]
            l1_s20 <<= a[b_idx, hv_idx, c_idx, 2 * BLOCK:3 * BLOCK, 0:BLOCK]
            l1_s21 <<= a[b_idx, hv_idx, c_idx, 2 * BLOCK:3 * BLOCK, BLOCK:2 * BLOCK]
            l1_s30 <<= a[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, 0:BLOCK]
            l1_s31 <<= a[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, BLOCK:2 * BLOCK]
            l1_s32 <<= a[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, 2 * BLOCK:3 * BLOCK]

            stage0.wait()
            l1_d0 <<= d01_ws[beat][0:BLOCK, 0:BLOCK]
            l1_d1 <<= d01_ws[beat][BLOCK:2 * BLOCK, 0:BLOCK]
            stage0.free()

            # X10 = (D1 S10) D0
            matmul(l0c[l0c_cnt], l1_d1, l1_s10.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_t, pub_ws[beat][PUB_T1 * BLOCK:(PUB_T1 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            l0c_cnt += 1
            matmul(l0c[l0c_cnt], l1_t, l1_d0.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_x10, pub_ws[beat][PUB_X10 * BLOCK:(PUB_X10 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            inv[b_idx, hv_idx, c_idx, BLOCK:2 * BLOCK, 0:BLOCK] <<= l0c[l0c_cnt]
            l0c_cnt += 1

            stage1.wait()
            l1_d2 <<= d23_ws[beat][0:BLOCK, 0:BLOCK]
            l1_d3 <<= d23_ws[beat][BLOCK:2 * BLOCK, 0:BLOCK]
            stage1.free()

            # P21 = D2 S21 ; X21 = P21 D1
            matmul(l0c[l0c_cnt], l1_d2, l1_s21.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_p21, pub_ws[beat][PUB_P21 * BLOCK:(PUB_P21 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            l0c_cnt += 1
            matmul(l0c[l0c_cnt], l1_p21, l1_d1.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_x21, pub_ws[beat][PUB_X21 * BLOCK:(PUB_X21 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            inv[b_idx, hv_idx, c_idx, 2 * BLOCK:3 * BLOCK, BLOCK:2 * BLOCK] <<= l0c[l0c_cnt]
            l0c_cnt += 1

            # P32 = D3 S32 ; X32 = P32 D2
            matmul(l0c[l0c_cnt], l1_d3, l1_s32.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_p32, pub_ws[beat][PUB_P32 * BLOCK:(PUB_P32 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            l0c_cnt += 1
            matmul(l0c[l0c_cnt], l1_p32, l1_d2.T, m=BLOCK, n=BLOCK, k=BLOCK)
            inv[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, 2 * BLOCK:3 * BLOCK] <<= l0c[l0c_cnt]
            l0c_cnt += 1

            # X20 = (D2 S20) D0 + P21 X10
            matmul(l0c[l0c_cnt], l1_d2, l1_s20.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_t, pub_ws[beat][PUB_T2 * BLOCK:(PUB_T2 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            l0c_cnt += 1
            matmul(l0c[l0c_cnt], l1_t, l1_d0.T, m=BLOCK, n=BLOCK, k=BLOCK)
            barrier(Pipe.M)
            matmul(l0c[l0c_cnt], l1_p21, l1_x10.T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            _publish(l1_x20, pub_ws[beat][PUB_X20 * BLOCK:(PUB_X20 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            inv[b_idx, hv_idx, c_idx, 2 * BLOCK:3 * BLOCK, 0:BLOCK] <<= l0c[l0c_cnt]
            l0c_cnt += 1

            # P31 = D3 S31 ; X31 = P31 D1 + P32 X21
            matmul(l0c[l0c_cnt], l1_d3, l1_s31.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_p31, pub_ws[beat][PUB_P31 * BLOCK:(PUB_P31 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            l0c_cnt += 1
            matmul(l0c[l0c_cnt], l1_p31, l1_d1.T, m=BLOCK, n=BLOCK, k=BLOCK)
            barrier(Pipe.M)
            matmul(l0c[l0c_cnt], l1_p32, l1_x21.T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            inv[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, BLOCK:2 * BLOCK] <<= l0c[l0c_cnt]
            l0c_cnt += 1

            # X30 = (D3 S30) D0 + P31 X10 + P32 X20
            matmul(l0c[l0c_cnt], l1_d3, l1_s30.T, m=BLOCK, n=BLOCK, k=BLOCK)
            _publish(l1_t, pub_ws[beat][PUB_T3 * BLOCK:(PUB_T3 + 1) * BLOCK, 0:BLOCK], l0c[l0c_cnt], pub_ev)
            l0c_cnt += 1
            matmul(l0c[l0c_cnt], l1_t, l1_d0.T, m=BLOCK, n=BLOCK, k=BLOCK)
            barrier(Pipe.M)
            matmul(l0c[l0c_cnt], l1_p31, l1_x10.T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            barrier(Pipe.M)
            matmul(l0c[l0c_cnt], l1_p32, l1_x20.T, m=BLOCK, n=BLOCK, k=BLOCK, is_init=False)
            inv[b_idx, hv_idx, c_idx, 3 * BLOCK:SIZE, 0:BLOCK] <<= l0c[l0c_cnt]
            l0c_cnt += 1

            beat += 1

    return inv
