"""One-launch BF16 GDN-2 decode core, raw gates, and output norm.

The fixed real-model shape gives every A5 vector participant one complete head.
The FP32 state stays resident in UB while BF16 model activations are widened in
registers.  Gate activation, q/k normalization, recurrence, RMSNorm, swish and
the sole output narrowing therefore share one launch and one precision boundary.
"""

from ascriptor.a5 import *


HEAD_DIM = 128
VALUE_DIM = 128
REGS = HEAD_DIM // 64
QK_EPS = 1e-6
NORM_EPS = 1e-5
Q_SCALE = 1.0 / 11.313708498984761
INV_VALUE_DIM = 1.0 / VALUE_DIM
STATE_TILE_ROWS = 64


@vf()
def prepare_decode_inputs_vf(
    q_raw_ub: Tensor,
    k_raw_ub: Tensor,
    v_raw_ub: Tensor,
    f_raw_ub: Tensor,
    b_raw_ub: Tensor,
    w_raw_ub: Tensor,
    decay_rate_ub: Tensor,
    dt_bias_ub: Tensor,
    norm_ub: Tensor,
    q_ub: Tensor,
    k_ub: Tensor,
    v_ub: Tensor,
    decay_ub: Tensor,
    b_ub: Tensor,
    w_ub: Tensor,
):
    """Widen BF16 rows, normalize q/k and activate all recurrence gates."""
    q_values = RegList(DT.float, REGS)
    k_values = RegList(DT.float, REGS)
    q_square = RegList(DT.float, REGS)
    k_square = RegList(DT.float, REGS)
    q_sum = Reg(DT.float)
    k_sum = Reg(DT.float)
    q_inv_norm = Reg(DT.float)
    k_inv_norm = Reg(DT.float)
    one = Reg(DT.float)

    q_values <<= q_raw_ub[0:1, 0:HEAD_DIM]
    k_values <<= k_raw_ub[0:1, 0:HEAD_DIM]
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

    values = RegList(DT.float, REGS)
    temporary = RegList(DT.float, REGS)
    positive = RegList(DT.float, REGS)
    invariant = RegList(DT.float, REGS)

    # Stable softplus(x) = max(x, 0) + log(1 + exp(-abs(x))).  The
    # precomputed negative decay rate is already broadcast across the row.
    values <<= f_raw_ub[0:1, 0:HEAD_DIM]
    invariant <<= dt_bias_ub[0:1, 0:HEAD_DIM]
    values <<= values + invariant
    positive <<= values.vmaxs(0.0)
    temporary <<= values.abs()
    temporary <<= -temporary
    temporary <<= temporary.exp()
    temporary <<= temporary + 1.0
    temporary <<= temporary.ln()
    values <<= positive + temporary
    invariant <<= decay_rate_ub[0:1, 0:HEAD_DIM]
    values <<= values * invariant
    values <<= values.exp()
    decay_ub[0:1, 0:HEAD_DIM] <<= values

    # sigmoid(b_raw), with allow_neg_eigval fixed false by the unit contract.
    values <<= b_raw_ub[0:1, 0:HEAD_DIM]
    temporary <<= -values
    temporary <<= temporary.exp()
    temporary <<= temporary + 1.0
    values <<= 1.0
    values <<= values / temporary
    b_ub[0:1, 0:HEAD_DIM] <<= values

    # sigmoid(w_raw).
    values <<= w_raw_ub[0:1, 0:VALUE_DIM]
    temporary <<= -values
    temporary <<= temporary.exp()
    temporary <<= temporary + 1.0
    values <<= 1.0
    values <<= values / temporary
    w_ub[0:1, 0:VALUE_DIM] <<= values

    values <<= v_raw_ub[0:1, 0:VALUE_DIM]
    v_ub[0:1, 0:VALUE_DIM] <<= values
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def init_recurrent_accumulators_vf(erase_ub: Tensor, output_ub: Tensor):
    """Initialize the two cross-tile FP32 accumulators exactly once."""
    erase = RegList(DT.float, REGS)
    output = RegList(DT.float, REGS)
    erase <<= 0.0
    output <<= 0.0
    erase_ub[0:1, 0:VALUE_DIM] <<= erase
    output_ub[0:1, 0:VALUE_DIM] <<= output
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def decay_and_erase_tile_vf(
    state_input_ub: Tensor,
    state_decayed_ub: Tensor,
    q_ub: Tensor,
    k_ub: Tensor,
    decay_ub: Tensor,
    b_ub: Tensor,
    erase_ub: Tensor,
    row_begin: Var,
):
    """First pass: decay one 64-row state tile and extend erase."""
    row = RegList(DT.float, REGS)
    erase = RegList(DT.float, REGS)
    temporary = RegList(DT.float, REGS)
    decay = Reg(DT.float)
    k_value = Reg(DT.float)
    b_value = Reg(DT.float)
    erase_weight = Reg(DT.float)
    erase <<= erase_ub[0:1, 0:VALUE_DIM]

    # S1 = Diag(decay) * S0; erase = (sigmoid(b_raw) * k_norm)^T * S1.
    for row_offset in range(STATE_TILE_ROWS):
        kk = Var(row_begin + row_offset)
        decay <<= decay_ub[0:1, kk:kk + 1].single()
        k_value <<= k_ub[0:1, kk:kk + 1].single()
        b_value <<= b_ub[0:1, kk:kk + 1].single()
        erase_weight <<= k_value * b_value
        row <<= state_input_ub[row_offset:row_offset + 1, 0:VALUE_DIM]
        row <<= row * decay
        state_decayed_ub[row_offset:row_offset + 1, 0:VALUE_DIM] <<= row
        temporary <<= row * erase_weight
        erase <<= erase + temporary
    erase_ub[0:1, 0:VALUE_DIM] <<= erase
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def make_delta_vf(
    v_ub: Tensor,
    w_ub: Tensor,
    erase_ub: Tensor,
    delta_ub: Tensor,
):
    """Finish delta after both decay/erase tiles have contributed."""
    delta = RegList(DT.float, REGS)
    write_gate = RegList(DT.float, REGS)
    erase = RegList(DT.float, REGS)

    delta <<= v_ub[0:1, 0:VALUE_DIM]
    write_gate <<= w_ub[0:1, 0:VALUE_DIM]
    delta <<= delta * write_gate
    erase <<= erase_ub[0:1, 0:VALUE_DIM]
    delta <<= delta - erase
    delta_ub[0:1, 0:VALUE_DIM] <<= delta
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def update_and_output_tile_vf(
    state_decayed_ub: Tensor,
    state_output_ub: Tensor,
    q_ub: Tensor,
    k_ub: Tensor,
    delta_ub: Tensor,
    output_ub: Tensor,
    row_begin: Var,
):
    """Second pass: update one 64-row tile and extend output."""
    row = RegList(DT.float, REGS)
    output = RegList(DT.float, REGS)
    delta = RegList(DT.float, REGS)
    temporary = RegList(DT.float, REGS)
    k_value = Reg(DT.float)
    q_value = Reg(DT.float)
    delta <<= delta_ub[0:1, 0:VALUE_DIM]
    output <<= output_ub[0:1, 0:VALUE_DIM]

    # S2 = S1 + k_norm * delta^T; recurrent_o = q_norm^T * S2.
    for row_offset in range(STATE_TILE_ROWS):
        kk = Var(row_begin + row_offset)
        k_value <<= k_ub[0:1, kk:kk + 1].single()
        q_value <<= q_ub[0:1, kk:kk + 1].single()
        row <<= state_decayed_ub[row_offset:row_offset + 1, 0:VALUE_DIM]
        temporary <<= delta * k_value
        row <<= row + temporary
        state_output_ub[row_offset:row_offset + 1, 0:VALUE_DIM] <<= row
        temporary <<= row * q_value
        output <<= output + temporary

    output_ub[0:1, 0:VALUE_DIM] <<= output
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def norm_gate_and_narrow_vf(
    recurrent_o_ub: Tensor,
    output_gate_raw_ub: Tensor,
    norm_weight_ub: Tensor,
    norm_ub: Tensor,
    output_ub: Tensor,
):
    """FP32 RMSNorm and swish, followed by the only BF16 narrowing."""
    values = RegList(DT.float, REGS)
    square = RegList(DT.float, REGS)
    weight = RegList(DT.float, REGS)
    gate = RegList(DT.float, REGS)
    denominator = RegList(DT.float, REGS)
    total = Reg(DT.float)
    inv_norm = Reg(DT.float)
    one = Reg(DT.float)

    values <<= recurrent_o_ub[0:1, 0:VALUE_DIM]
    square <<= values * values
    total <<= square.cadd()
    norm_ub[0:1, 0:1] <<= total.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    inv_norm <<= norm_ub[0:1, 0:1].single()
    inv_norm <<= inv_norm * INV_VALUE_DIM
    inv_norm <<= inv_norm + NORM_EPS
    inv_norm <<= inv_norm.sqrt()
    one <<= 1.0
    inv_norm <<= one / inv_norm
    values <<= values * inv_norm

    weight <<= norm_weight_ub[0:1, 0:VALUE_DIM]
    values <<= values * weight
    gate <<= output_gate_raw_ub[0:1, 0:VALUE_DIM]
    denominator <<= -gate
    denominator <<= denominator.exp()
    denominator <<= denominator + 1.0
    gate <<= gate / denominator
    values <<= values * gate

    round_to_bf16 = CastConfig(
        round_mode=RoundMode.TO_EVEN,
        reg_layout=RegLayout.ZERO,
        name="gdn2_decode_output_bf16",
    )
    low = Reg(DT.bfloat16)
    high = Reg(DT.bfloat16)
    packed = Reg(DT.bfloat16)
    unused = Reg(DT.bfloat16)
    low <<= values[0].astype(DT.bfloat16, round_to_bf16)
    high <<= values[1].astype(DT.bfloat16, round_to_bf16)
    deinterleave(packed, unused, low, high)
    reg_to_ub_normal(output_ub[0:1, 0:VALUE_DIM], packed)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def gdn2_fused_decode_kernel(
    q: GM[bf16, (1, 1, 16, 128)],
    k: GM[bf16, (1, 1, 16, 128)],
    v: GM[bf16, (1, 1, 16, 128)],
    f_raw: GM[bf16, (1, 1, 16, 128)],
    b_raw: GM[bf16, (1, 1, 16, 128)],
    w_raw: GM[bf16, (1, 1, 16, 128)],
    output_gate: GM[bf16, (1, 1, 16, 128)],
    decay_rate: GM[f32, (16, 128)],
    dt_bias: GM[f32, (16, 128)],
    norm_weight: GM[f32, (128,)],
    initial_state: GM[f32, (1, 16, 128, 128)],
    o: GM[bf16, (1, 1, 16, 128)],
    final_state: GM[f32, (1, 16, 128, 128)],
):
    """Assign one complete real-model head to each vector participant."""
    state_decayed_ub = Tensor(DT.float, [HEAD_DIM, VALUE_DIM], Position.UB)
    state_slot0_ub = Tensor(DT.float, [STATE_TILE_ROWS, VALUE_DIM], Position.UB)
    state_slot1_ub = Tensor(DT.float, [STATE_TILE_ROWS, VALUE_DIM], Position.UB)

    q_raw_ub = Tensor(DT.bfloat16, [1, HEAD_DIM], Position.UB)
    k_raw_ub = Tensor(DT.bfloat16, [1, HEAD_DIM], Position.UB)
    v_raw_ub = Tensor(DT.bfloat16, [1, VALUE_DIM], Position.UB)
    f_raw_ub = Tensor(DT.bfloat16, [1, HEAD_DIM], Position.UB)
    b_raw_ub = Tensor(DT.bfloat16, [1, HEAD_DIM], Position.UB)
    w_raw_ub = Tensor(DT.bfloat16, [1, VALUE_DIM], Position.UB)
    output_gate_raw_ub = Tensor(DT.bfloat16, [1, VALUE_DIM], Position.UB)

    decay_rate_ub = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    dt_bias_ub = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    norm_weight_ub = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    q_ub = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    k_ub = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    v_ub = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    decay_ub = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    b_ub = Tensor(DT.float, [1, HEAD_DIM], Position.UB)
    w_ub = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    erase_ub = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    delta_ub = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    recurrent_o_ub = Tensor(DT.float, [1, VALUE_DIM], Position.UB)
    norm_ub = Tensor(DT.float, [1, 8], Position.UB)
    output_ub = Tensor(DT.bfloat16, [1, VALUE_DIM], Position.UB)

    head = Var(GetVecIdx())
    row0 = Var(0)
    row1 = Var(STATE_TILE_ROWS)
    with auto_sync():
        if head < 16:
            q_raw_ub[0:1, 0:HEAD_DIM] <<= q[0, 0:1, head, 0:HEAD_DIM]
            k_raw_ub[0:1, 0:HEAD_DIM] <<= k[0, 0:1, head, 0:HEAD_DIM]
            v_raw_ub[0:1, 0:VALUE_DIM] <<= v[0, 0:1, head, 0:VALUE_DIM]
            f_raw_ub[0:1, 0:HEAD_DIM] <<= f_raw[0, 0:1, head, 0:HEAD_DIM]
            b_raw_ub[0:1, 0:HEAD_DIM] <<= b_raw[0, 0:1, head, 0:HEAD_DIM]
            w_raw_ub[0:1, 0:VALUE_DIM] <<= w_raw[0, 0:1, head, 0:VALUE_DIM]
            output_gate_raw_ub[0:1, 0:VALUE_DIM] <<= output_gate[
                0, 0:1, head, 0:VALUE_DIM
            ]
            decay_rate_ub[0:1, 0:HEAD_DIM] <<= decay_rate[head, 0:HEAD_DIM]
            dt_bias_ub[0:1, 0:HEAD_DIM] <<= dt_bias[head, 0:HEAD_DIM]
            norm_weight_ub[0:1, 0:VALUE_DIM] <<= norm_weight[0:VALUE_DIM]

            state_slot0_ub[0:STATE_TILE_ROWS, 0:VALUE_DIM] <<= initial_state[
                0, head, 0:STATE_TILE_ROWS, 0:VALUE_DIM
            ]

            prepare_decode_inputs_vf(
                q_raw_ub,
                k_raw_ub,
                v_raw_ub,
                f_raw_ub,
                b_raw_ub,
                w_raw_ub,
                decay_rate_ub,
                dt_bias_ub,
                norm_ub,
                q_ub,
                k_ub,
                v_ub,
                decay_ub,
                b_ub,
                w_ub,
            )
            init_recurrent_accumulators_vf(erase_ub, recurrent_o_ub)

            # Static slot identities let MTE2 fill slot 1 while Vector consumes
            # slot 0; no dynamic DBuff alias joins these independent operations.
            state_slot1_ub[0:STATE_TILE_ROWS, 0:VALUE_DIM] <<= initial_state[
                0, head, STATE_TILE_ROWS:HEAD_DIM, 0:VALUE_DIM
            ]
            decay_and_erase_tile_vf(
                state_slot0_ub,
                state_decayed_ub[0:STATE_TILE_ROWS, 0:VALUE_DIM],
                q_ub,
                k_ub,
                decay_ub,
                b_ub,
                erase_ub,
                row0,
            )
            decay_and_erase_tile_vf(
                state_slot1_ub,
                state_decayed_ub[STATE_TILE_ROWS:HEAD_DIM, 0:VALUE_DIM],
                q_ub,
                k_ub,
                decay_ub,
                b_ub,
                erase_ub,
                row1,
            )

            make_delta_vf(v_ub, w_ub, erase_ub, delta_ub)
            update_and_output_tile_vf(
                state_decayed_ub[0:STATE_TILE_ROWS, 0:VALUE_DIM],
                state_slot0_ub,
                q_ub,
                k_ub,
                delta_ub,
                recurrent_o_ub,
                row0,
            )
            final_state[0, head, 0:STATE_TILE_ROWS, 0:VALUE_DIM] <<= state_slot0_ub[
                0:STATE_TILE_ROWS, 0:VALUE_DIM
            ]

            # MTE3 drains slot 0 while Vector produces the disjoint slot 1.
            update_and_output_tile_vf(
                state_decayed_ub[STATE_TILE_ROWS:HEAD_DIM, 0:VALUE_DIM],
                state_slot1_ub,
                q_ub,
                k_ub,
                delta_ub,
                recurrent_o_ub,
                row1,
            )
            final_state[
                0, head, STATE_TILE_ROWS:HEAD_DIM, 0:VALUE_DIM
            ] <<= state_slot1_ub[0:STATE_TILE_ROWS, 0:VALUE_DIM]

            norm_gate_and_narrow_vf(
                recurrent_o_ub,
                output_gate_raw_ub,
                norm_weight_ub,
                norm_ub,
                output_ub,
            )

            o[0, 0:1, head, 0:VALUE_DIM] <<= output_ub[0:1, 0:VALUE_DIM]

    return o, final_state
