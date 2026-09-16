"""One-launch BF16 packed q/k/v short convolution for GDN-2 decode.

The released 1.3B model fixes 6144 packed depthwise channels and width four.
Each A5 vector participant owns 384 complete channels.  Cache and weight rows
arrive channel-major in GM, are transposed into four contiguous tap rows in UB,
and are never shared between participants.
"""

from ascriptor.a5 import *


CHANNELS = 6144
WIDTH = 4
BLOCK_DIM = 8
VECTOR_PARTICIPANTS = 2 * BLOCK_DIM
CHANNELS_PER_PARTICIPANT = CHANNELS // VECTOR_PARTICIPANTS
VECTOR_TILE = 128
TILES_PER_PARTICIPANT = CHANNELS_PER_PARTICIPANT // VECTOR_TILE


@vf()
def short_conv_tile_vf(
    x_ub: Tensor,
    cache_t_ub: Tensor,
    weight_t_ub: Tensor,
    y_ub: Tensor,
    new_cache_ub: Tensor,
    channel_begin: Var,
):
    """Compute and publish one 128-channel tile entirely in registers."""
    old1 = Reg(DT.bfloat16)
    old2 = Reg(DT.bfloat16)
    old3 = Reg(DT.bfloat16)
    current = Reg(DT.bfloat16)

    old1 <<= cache_t_ub[1:2, channel_begin:channel_begin + VECTOR_TILE]
    old2 <<= cache_t_ub[2:3, channel_begin:channel_begin + VECTOR_TILE]
    old3 <<= cache_t_ub[3:4, channel_begin:channel_begin + VECTOR_TILE]
    current <<= x_ub[0:1, channel_begin:channel_begin + VECTOR_TILE]

    values = RegList(DT.float, 2)
    tap = RegList(DT.float, 2)
    weight = RegList(DT.float, 2)
    denominator = RegList(DT.float, 2)

    values <<= cache_t_ub[1:2, channel_begin:channel_begin + VECTOR_TILE]
    weight <<= weight_t_ub[0:1, channel_begin:channel_begin + VECTOR_TILE]
    values <<= values * weight

    tap <<= cache_t_ub[2:3, channel_begin:channel_begin + VECTOR_TILE]
    weight <<= weight_t_ub[1:2, channel_begin:channel_begin + VECTOR_TILE]
    tap <<= tap * weight
    values <<= values + tap

    tap <<= cache_t_ub[3:4, channel_begin:channel_begin + VECTOR_TILE]
    weight <<= weight_t_ub[2:3, channel_begin:channel_begin + VECTOR_TILE]
    tap <<= tap * weight
    values <<= values + tap

    tap <<= x_ub[0:1, channel_begin:channel_begin + VECTOR_TILE]
    weight <<= weight_t_ub[3:4, channel_begin:channel_begin + VECTOR_TILE]
    tap <<= tap * weight
    values <<= values + tap

    # SiLU(x) = x / (1 + exp(-x)).  The contract bounds generated values so
    # the direct exponential stays finite; the real model is well inside it.
    denominator <<= -values
    denominator <<= denominator.exp()
    denominator <<= denominator + 1.0
    values <<= values / denominator

    round_to_bf16 = CastConfig(
        round_mode=RoundMode.TO_EVEN,
        reg_layout=RegLayout.ZERO,
        name="gdn2_short_conv_output_bf16",
    )
    low = Reg(DT.bfloat16)
    high = Reg(DT.bfloat16)
    packed = Reg(DT.bfloat16)
    unused = Reg(DT.bfloat16)
    low <<= values[0].astype(DT.bfloat16, round_to_bf16)
    high <<= values[1].astype(DT.bfloat16, round_to_bf16)
    deinterleave(packed, unused, low, high)
    reg_to_ub_normal(
        y_ub[0:1, channel_begin:channel_begin + VECTOR_TILE], packed
    )

    # Weave four 128-lane BF16 tap registers into compact channel-major rows.
    # First make AB/CD pairs at b16 width, then interleave those pairs at b32
    # width.  Reinterpreting the final registers gives
    #   [old1[c], old2[c], old3[c], current[c]]
    # for 32 consecutive channels per register, without a UB transpose.
    ab0 = Reg(DT.bfloat16)
    ab1 = Reg(DT.bfloat16)
    cd0 = Reg(DT.bfloat16)
    cd1 = Reg(DT.bfloat16)
    interleave(ab0, ab1, old1, old2)
    interleave(cd0, cd1, old3, current)

    packed0 = Reg(DT.uint32)
    packed1 = Reg(DT.uint32)
    packed2 = Reg(DT.uint32)
    packed3 = Reg(DT.uint32)
    interleave(
        packed0,
        packed1,
        ab0.reinterpret(DT.uint32),
        cd0.reinterpret(DT.uint32),
    )
    interleave(
        packed2,
        packed3,
        ab1.reinterpret(DT.uint32),
        cd1.reinterpret(DT.uint32),
    )

    cache_element_begin = Var(channel_begin * WIDTH)
    reg_to_ub_normal(
        new_cache_ub[cache_element_begin], packed0.reinterpret(DT.bfloat16)
    )
    reg_to_ub_normal(
        new_cache_ub[cache_element_begin + 128], packed1.reinterpret(DT.bfloat16)
    )
    reg_to_ub_normal(
        new_cache_ub[cache_element_begin + 256], packed2.reinterpret(DT.bfloat16)
    )
    reg_to_ub_normal(
        new_cache_ub[cache_element_begin + 384], packed3.reinterpret(DT.bfloat16)
    )
    vf_barrier(VfPipe.STORE, VfPipe.LOAD)


@kernel()
def gdn2_short_conv_decode_kernel(
    x: GM[bf16, (1, CHANNELS)],
    cache: GM[bf16, (CHANNELS, WIDTH)],
    weight: GM[bf16, (CHANNELS, WIDTH)],
    y: GM[bf16, (1, CHANNELS)],
    new_cache: GM[bf16, (CHANNELS, WIDTH)],
):
    """Run the fixed B1/T1/D6144/W4 packed short-conv decode."""
    x_ub = Tensor(DT.bfloat16, [1, CHANNELS_PER_PARTICIPANT], Position.UB)
    cache_t_ub = Tensor(
        DT.bfloat16, [WIDTH, CHANNELS_PER_PARTICIPANT], Position.UB
    )
    weight_t_ub = Tensor(
        DT.bfloat16, [WIDTH, CHANNELS_PER_PARTICIPANT], Position.UB
    )
    y_ub = Tensor(DT.bfloat16, [1, CHANNELS_PER_PARTICIPANT], Position.UB)
    new_cache_ub = Tensor(
        DT.bfloat16, [CHANNELS_PER_PARTICIPANT, WIDTH], Position.UB
    )

    participant = Var(GetVecIdx())
    gm_channel_begin = Var(participant * CHANNELS_PER_PARTICIPANT)
    new_cache_flat = new_cache.reshape([CHANNELS * WIDTH])
    with auto_sync():
        if participant < VECTOR_PARTICIPANTS:
            x_ub[0:1, 0:CHANNELS_PER_PARTICIPANT] <<= x[
                0:1, gm_channel_begin:gm_channel_begin + CHANNELS_PER_PARTICIPANT
            ]
            gm_to_ub_nd_dma_transpose(
                cache_t_ub,
                cache[
                    gm_channel_begin:gm_channel_begin + CHANNELS_PER_PARTICIPANT,
                    0:WIDTH,
                ],
            )
            gm_to_ub_nd_dma_transpose(
                weight_t_ub,
                weight[
                    gm_channel_begin:gm_channel_begin + CHANNELS_PER_PARTICIPANT,
                    0:WIDTH,
                ],
            )

            for tile in range(TILES_PER_PARTICIPANT):
                channel_begin = Var(tile * VECTOR_TILE)
                short_conv_tile_vf(
                    x_ub,
                    cache_t_ub,
                    weight_t_ub,
                    y_ub,
                    new_cache_ub,
                    channel_begin,
                )

            y[0:1, gm_channel_begin:gm_channel_begin + CHANNELS_PER_PARTICIPANT] <<= y_ub[
                0:1, 0:CHANNELS_PER_PARTICIPANT
            ]
            # A channel-major cache row is only eight bytes.  Flatten the whole
            # GM output and issue one aligned contiguous burst instead of 384
            # unaligned row bursts.
            ub_to_gm_pad(
                new_cache_flat[
                    gm_channel_begin * WIDTH:
                    (gm_channel_begin + CHANNELS_PER_PARTICIPANT) * WIDTH
                ],
                new_cache_ub,
                1,
                CHANNELS_PER_PARTICIPANT * WIDTH,
                0,
                0,
            )

    return y, new_cache
