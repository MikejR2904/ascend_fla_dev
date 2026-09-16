"""Numerically conservative fused RMSNorm2, W1/W2 and SwiGLU decode.

The two vector subblocks redundantly compute the same FP32 RMS reduction and
publish disjoint 1152-element halves of the checkpoint-faithful BF16 normalized
input to L1. Cube then streams paired W1/W2 tiles. The C2V epilogue preserves
both observable BF16 boundaries of the vendor composition: GEMV output and
SiLU output. Only the final BF16 hidden activation reaches GM; W3 stays out.
"""

from ascriptor.a5 import *


K = 2304
K_HALF = K // 2
INTERMEDIATE = 6208
CHANNELS_PER_ITEM = 64
ITEMS = INTERMEDIATE // CHANNELS_PER_ITEM
PAIR_N = 2 * CHANNELS_PER_ITEM
PHYSICAL_M = 16
K_TILE = 256
K_TILES = K // K_TILE
BLOCK_DIM = 28
INV_K = 1.0 / K
EPS = 1e-5


@vf()
def normalize_half_vf(
    x_ub: Tensor,
    gamma_half_ub: Tensor,
    normalized_half_ub: Tensor,
    inverse_ub: Tensor,
    global_k_begin: Var,
):
    """Compute the full RMS scalar, then one disjoint BF16 K half."""
    norm_values = RegList(DT.float, K // 64)
    values = RegList(DT.float, 2)
    gamma = RegList(DT.float, 2)
    inverse = Reg(DT.float)
    one = Reg(DT.float)
    norm_values <<= x_ub[0:1, 0:K]
    norm_values <<= norm_values * norm_values
    inverse <<= norm_values.cadd()
    inverse <<= inverse * INV_K
    inverse <<= inverse + EPS
    inverse <<= inverse.sqrt()
    one <<= 1.0
    inverse <<= one / inverse
    inverse_ub[0:1, 0:1] <<= inverse.single_value()
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)
    inverse <<= inverse_ub[0:1, 0:1].single()

    round_to_bf16 = CastConfig(
        round_mode=RoundMode.TO_EVEN,
        reg_layout=RegLayout.ZERO,
        name="gdn2_norm2_output_bf16",
    )
    low = Reg(DT.bfloat16)
    high = Reg(DT.bfloat16)
    packed = Reg(DT.bfloat16)
    unused = Reg(DT.bfloat16)
    for chunk in range(K_HALF // 128):
        local_begin = chunk * 128
        global_begin = Var(global_k_begin + local_begin)
        values <<= x_ub[0:1, global_begin:global_begin + 128]
        gamma <<= gamma_half_ub[0:1, local_begin:local_begin + 128]
        values <<= values * inverse
        values <<= values * gamma
        low <<= values[0].astype(DT.bfloat16, round_to_bf16)
        high <<= values[1].astype(DT.bfloat16, round_to_bf16)
        deinterleave(packed, unused, low, high)
        reg_to_ub_normal(
            normalized_half_ub[0:1, local_begin:local_begin + 128], packed
        )
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@vf()
def swiglu_epilogue_vf(
    product_bf16_ub: Tensor,
    silu_bf16_ub: Tensor,
    hidden_bf16_ub: Tensor,
):
    """Preserve GEMV-BF16 and SiLU-BF16 before the final BF16 multiply."""
    left = Reg(DT.float)
    right = Reg(DT.float)
    denominator = Reg(DT.float)
    zero = Reg(DT.float)

    left <<= product_bf16_ub[0:1, 0:CHANNELS_PER_ITEM]
    denominator <<= -left
    denominator <<= denominator.exp()
    denominator <<= denominator + 1.0
    left <<= left / denominator

    zero <<= 0.0
    round_silu = CastConfig(
        round_mode=RoundMode.TO_EVEN,
        reg_layout=RegLayout.ZERO,
        name="gdn2_swiglu_silu_bf16",
    )
    low = Reg(DT.bfloat16)
    high = Reg(DT.bfloat16)
    packed = Reg(DT.bfloat16)
    unused = Reg(DT.bfloat16)
    low <<= left.astype(DT.bfloat16, round_silu)
    high <<= zero.astype(DT.bfloat16, round_silu)
    deinterleave(packed, unused, low, high)
    reg_to_ub_normal(silu_bf16_ub[0:1, 0:128], packed)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    left <<= silu_bf16_ub[0:1, 0:CHANNELS_PER_ITEM]
    right <<= product_bf16_ub[0:1, CHANNELS_PER_ITEM:PAIR_N]
    left <<= left * right
    round_hidden = CastConfig(
        round_mode=RoundMode.TO_EVEN,
        reg_layout=RegLayout.ZERO,
        name="gdn2_swiglu_hidden_bf16",
    )
    low <<= left.astype(DT.bfloat16, round_hidden)
    high <<= zero.astype(DT.bfloat16, round_hidden)
    deinterleave(packed, unused, low, high)
    reg_to_ub_normal(hidden_bf16_ub[0:1, 0:128], packed)
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel(mode="mix", block_dim=BLOCK_DIM)
def gdn2_norm2_w12_swiglu_kernel(
    x: GM[bf16, (1, K)],
    gamma: GM[bf16, (K,)],
    paired_weight: GM[bf16, (ITEMS, PAIR_N, K)],
    hidden: GM[bf16, (1, INTERMEDIATE)],
):
    """Run the fixed B=T=1, D=2304, I=6208 conservative boundary."""
    x_l1 = Tensor(DT.bfloat16, [PHYSICAL_M, K], Position.L1)
    weight_l1 = DBuff(DT.bfloat16, [PAIR_N, K_TILE], Position.L1)
    product_l0c = DBuff(DT.float, [PHYSICAL_M, PAIR_N], Position.L0C)

    x_ub = Tensor(DT.bfloat16, [1, K], Position.UB)
    gamma_half_ub = Tensor(DT.bfloat16, [1, K_HALF], Position.UB)
    normalized_half_ub = Tensor(DT.bfloat16, [1, K_HALF], Position.UB)
    inverse_ub = Tensor(DT.float, [1, 8], Position.UB)
    product_bf16_ub = DBuff(DT.bfloat16, [PHYSICAL_M, PAIR_N], Position.UB)
    silu_bf16_ub = DBuff(DT.bfloat16, [1, 128], Position.UB)
    hidden_bf16_ub = DBuff(DT.bfloat16, [1, 128], Position.UB)

    normalized_handoff = VcMutex(
        0,
        guards=x_l1,
        src_end_pipe=Pipe.MTE3,
        dst_end_pipe=Pipe.M,
    )
    product_handoff = CvMutex(
        1,
        guards=product_bf16_ub,
        src_end_pipe=Pipe.FIX,
        dst_end_pipe=Pipe.V,
    )

    weight_slot = Var(0)
    product_slot = Var(0)
    item_begin = Var(ITEMS * GetCubeIdx() // GetCubeNum())
    item_end = Var(ITEMS * (GetCubeIdx() + 1) // GetCubeNum())
    with auto_sync():
        normalized_handoff.lock()
        global_k_begin = Var(GetSubBlockIdx() * K_HALF)
        x_ub[0:1, 0:K] <<= x[0:1, 0:K]
        gamma_half_ub[0:1, 0:K_HALF] <<= gamma[
            global_k_begin:global_k_begin + K_HALF
        ]
        normalize_half_vf(
            x_ub,
            gamma_half_ub,
            normalized_half_ub,
            inverse_ub,
            global_k_begin,
        )
        x_l1[0:1, global_k_begin:global_k_begin + K_HALF] <<= (
            normalized_half_ub[0:1, 0:K_HALF].nz()
        )
        normalized_handoff.ready()

        normalized_handoff.wait()
        for item in range(item_begin, item_end):
            for k_tile in range(K_TILES):
                k_begin = Var(k_tile * K_TILE)
                weight_l1[weight_slot] <<= paired_weight[
                    item,
                    0:PAIR_N,
                    k_begin:k_begin + K_TILE,
                ]
                matmul(
                    product_l0c[product_slot],
                    x_l1[0:PHYSICAL_M, k_begin:k_begin + K_TILE],
                    weight_l1[weight_slot],
                    m=PHYSICAL_M,
                    n=PAIR_N,
                    k=K_TILE,
                    splitn=64,
                    is_init=(k_tile == 0),
                )
                weight_slot += 1

            product_handoff.lock()
            # One subblock needs both matching 64-channel halves for SiLU×W2.
            l0c_to_ub(
                product_bf16_ub[product_slot],
                product_l0c[product_slot],
                M=PHYSICAL_M,
                N=PAIR_N,
                N_dst=PAIR_N,
                M_src=PHYSICAL_M,
                dual_mode=DualMode.SINGLE,
                sub_block_id=0,
                scale=1.0,
            )
            product_handoff.ready()

            product_handoff.wait()
            if GetSubBlockIdx() == 0:
                swiglu_epilogue_vf(
                    product_bf16_ub[product_slot],
                    silu_bf16_ub[product_slot],
                    hidden_bf16_ub[product_slot],
                )
            product_handoff.free()
            if GetSubBlockIdx() == 0:
                channel_begin = Var(item * CHANNELS_PER_ITEM)
                hidden[0:1, channel_begin:channel_begin + CHANNELS_PER_ITEM] <<= (
                    hidden_bf16_ub[product_slot][0:1, 0:CHANNELS_PER_ITEM]
                )
            product_slot += 1
        normalized_handoff.free()
    return hidden
