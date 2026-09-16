"""One-launch GDN-2 recurrent forward for short inference sequences.

The layer has already activated ``g``, ``b`` and ``w``. This kernel owns q/k
normalization, channel-wise state decay, erase/write update, output read and the
FP32 final state. Each vector participant owns complete heads, so no state row
or reduction crosses cores.
"""

from ascriptor.a5 import *


HEAD_DIM = 128
VALUE_DIM = 128
REGS = HEAD_DIM // 64
T_MAX = 16
QK_EPS = 1e-6
Q_SCALE = 1.0 / 11.313708498984761


@vf()
def gdn2_recurrent_token_vf(
    state_ub: Tensor,
    q_ub: Tensor,
    k_ub: Tensor,
    v_ub: Tensor,
    g_ub: Tensor,
    b_ub: Tensor,
    w_ub: Tensor,
    norm_ub: Tensor,
    o_ub: Tensor,
):
    """Advance one token while ``state_ub[K,V]`` remains resident in UB."""
    q_values = RegList(DT.float, REGS)
    k_values = RegList(DT.float, REGS)
    q_square = RegList(DT.float, REGS)
    k_square = RegList(DT.float, REGS)
    q_sum = Reg(DT.float)
    k_sum = Reg(DT.float)
    q_inv_norm = Reg(DT.float)
    k_inv_norm = Reg(DT.float)
    one = Reg(DT.float)

    q_values <<= q_ub[0:1, 0:HEAD_DIM]
    k_values <<= k_ub[0:1, 0:HEAD_DIM]
    q_square <<= q_values * q_values
    k_square <<= k_values * k_values
    q_sum <<= q_square.cadd()
    k_sum <<= k_square.cadd()
    norm_ub[0:1, 0:1] <<= q_sum.single_value()
    norm_ub[0:1, 1:2] <<= k_sum.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    q_inv_norm <<= norm_ub[0:1, 0:1].single()
    k_inv_norm <<= norm_ub[0:1, 1:2].single()
    q_inv_norm <<= q_inv_norm + QK_EPS
    k_inv_norm <<= k_inv_norm + QK_EPS
    q_inv_norm <<= q_inv_norm.sqrt()
    k_inv_norm <<= k_inv_norm.sqrt()
    one <<= 1.0
    q_inv_norm <<= one / q_inv_norm
    k_inv_norm <<= one / k_inv_norm
    q_inv_norm <<= q_inv_norm * Q_SCALE
    q_values <<= q_values * q_inv_norm
    k_values <<= k_values * k_inv_norm
    q_ub[0:1, 0:HEAD_DIM] <<= q_values
    k_ub[0:1, 0:HEAD_DIM] <<= k_values
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    row = RegList(DT.float, REGS)
    erase = RegList(DT.float, REGS)
    output = RegList(DT.float, REGS)
    delta = RegList(DT.float, REGS)
    write_gate = RegList(DT.float, REGS)
    temporary = RegList(DT.float, REGS)

    decay = Reg(DT.float)
    g_value = Reg(DT.float)
    k_value = Reg(DT.float)
    b_value = Reg(DT.float)
    erase_weight = Reg(DT.float)
    q_value = Reg(DT.float)

    erase <<= 0.0
    output <<= 0.0

    # S1 = Diag(exp(g)) * S0; erase = (b * k_norm)^T * S1.
    for kk in range(HEAD_DIM):
        g_value <<= g_ub[0:1, kk:kk + 1].single()
        decay <<= g_value.exp()
        k_value <<= k_ub[0:1, kk:kk + 1].single()
        b_value <<= b_ub[0:1, kk:kk + 1].single()
        erase_weight <<= k_value * b_value
        row <<= state_ub[kk:kk + 1, 0:VALUE_DIM]
        row <<= row * decay
        state_ub[kk:kk + 1, 0:VALUE_DIM] <<= row
        temporary <<= row * erase_weight
        erase <<= erase + temporary

    # delta = w * v - erase.
    delta <<= v_ub[0:1, 0:VALUE_DIM]
    write_gate <<= w_ub[0:1, 0:VALUE_DIM]
    delta <<= delta * write_gate
    delta <<= delta - erase

    # S2 = S1 + k_norm * delta^T; o = (q_norm * scale)^T * S2.
    for kk in range(HEAD_DIM):
        k_value <<= k_ub[0:1, kk:kk + 1].single()
        q_value <<= q_ub[0:1, kk:kk + 1].single()
        row <<= state_ub[kk:kk + 1, 0:VALUE_DIM]
        temporary <<= delta * k_value
        row <<= row + temporary
        state_ub[kk:kk + 1, 0:VALUE_DIM] <<= row
        temporary <<= row * q_value
        output <<= output + temporary

    o_ub[0:1, 0:VALUE_DIM] <<= output
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def gdn2_fused_recurrent_kernel(
    q: GM[f32, ("B", "T", "H", 128)],
    k: GM[f32, ("B", "T", "H", 128)],
    v: GM[f32, ("B", "T", "H", 128)],
    g: GM[f32, ("B", "T", "H", 128)],
    erase_gate: GM[f32, ("B", "T", "H", 128)],
    w: GM[f32, ("B", "T", "H", 128)],
    initial_state: GM[f32, ("B", "H", 128, 128)],
    o: GM[f32, ("B", "T", "H", 128)],
    final_state: GM[f32, ("B", "H", 128, 128)],
    B: i32,
    T: i32,
    H: i32,
):
    """Assign complete heads to vector participants and keep state across T."""
    state_ub = Tensor(DT.float, [HEAD_DIM, VALUE_DIM], Position.UB)
    q_ub = DBuff(DT.float, [1, HEAD_DIM], Position.UB)
    k_ub = DBuff(DT.float, [1, HEAD_DIM], Position.UB)
    v_ub = DBuff(DT.float, [1, VALUE_DIM], Position.UB)
    g_ub = DBuff(DT.float, [1, HEAD_DIM], Position.UB)
    b_ub = DBuff(DT.float, [1, HEAD_DIM], Position.UB)
    w_ub = DBuff(DT.float, [1, VALUE_DIM], Position.UB)
    norm_ub = DBuff(DT.float, [1, 8], Position.UB)
    o_ub = DBuff(DT.float, [1, VALUE_DIM], Position.UB)

    work_count = B * H
    work_per_vec = CeilDiv(work_count, GetVecNum())
    work_begin = Var(work_per_vec * GetVecIdx())
    work_end = Min(work_begin + work_per_vec, work_count)

    with auto_sync():
        for work in range(work_begin, work_end):
            h_idx = Var(work % H)
            b_idx = Var(work // H)
            state_ub[0:HEAD_DIM, 0:VALUE_DIM] <<= initial_state[
                b_idx, h_idx, 0:HEAD_DIM, 0:VALUE_DIM
            ]

            for t in range(T):
                slot = Var(work * T + t)
                q_ub[slot][0:1, 0:HEAD_DIM] <<= q[b_idx, t:t + 1, h_idx, 0:HEAD_DIM]
                k_ub[slot][0:1, 0:HEAD_DIM] <<= k[b_idx, t:t + 1, h_idx, 0:HEAD_DIM]
                v_ub[slot][0:1, 0:VALUE_DIM] <<= v[b_idx, t:t + 1, h_idx, 0:VALUE_DIM]
                g_ub[slot][0:1, 0:HEAD_DIM] <<= g[b_idx, t:t + 1, h_idx, 0:HEAD_DIM]
                b_ub[slot][0:1, 0:HEAD_DIM] <<= erase_gate[
                    b_idx, t:t + 1, h_idx, 0:HEAD_DIM
                ]
                w_ub[slot][0:1, 0:VALUE_DIM] <<= w[b_idx, t:t + 1, h_idx, 0:VALUE_DIM]

                gdn2_recurrent_token_vf(
                    state_ub,
                    q_ub[slot],
                    k_ub[slot],
                    v_ub[slot],
                    g_ub[slot],
                    b_ub[slot],
                    w_ub[slot],
                    norm_ub[slot],
                    o_ub[slot],
                )
                o[b_idx, t:t + 1, h_idx, 0:VALUE_DIM] <<= o_ub[slot][
                    0:1, 0:VALUE_DIM
                ]

            final_state[b_idx, h_idx, 0:HEAD_DIM, 0:VALUE_DIM] <<= state_ub[
                0:HEAD_DIM, 0:VALUE_DIM
            ]

    return o, final_state
