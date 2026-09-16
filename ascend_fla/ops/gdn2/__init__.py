"""GDN-2 operators with repository-owned public APIs."""

from .fused_decode import (
    GDN2_DECODE_BLOCK_DIM,
    GDN2_DECODE_HEADS,
    fused_decode_gdn2,
    prepare as _prepare_fused_decode,
)

from .fused_recurrent import (
    GDN2_RECURRENT_BLOCK_DIM,
    GDN2_RECURRENT_T_MAX,
    fused_recurrent_gdn2,
    prepare as _prepare_recurrent,
)
from .short_conv_decode import (
    GDN2_SHORT_CONV_BLOCK_DIM,
    GDN2_SHORT_CONV_CHANNELS,
    GDN2_SHORT_CONV_WIDTH,
    fused_short_conv_decode_gdn2,
    prepare as _prepare_short_conv_decode,
)
from .mixed_mlp import (
    GDN2_MIXED_MLP_BLOCK_DIM,
    GDN2_MIXED_MLP_HIDDEN_SIZE,
    GDN2_MIXED_MLP_INTERMEDIATE_SIZE,
    fused_norm2_w12_swiglu_gdn2,
    prepare as _prepare_mixed_mlp,
)


def prepare(
    *,
    device: str = "a5",
    block_dim: int = GDN2_RECURRENT_BLOCK_DIM,
    mixed_mlp: bool = False,
) -> None:
    """Compile every GDN-2 CCE kernel before CANN's first operator lookup."""
    _prepare_recurrent(device=device, block_dim=block_dim)
    _prepare_fused_decode(device=device, block_dim=block_dim)
    _prepare_short_conv_decode(device=device, block_dim=block_dim)
    if mixed_mlp:
        _prepare_mixed_mlp(device=device, block_dim=GDN2_MIXED_MLP_BLOCK_DIM)

__all__ = [
    "GDN2_RECURRENT_BLOCK_DIM",
    "GDN2_RECURRENT_T_MAX",
    "GDN2_DECODE_BLOCK_DIM",
    "GDN2_DECODE_HEADS",
    "GDN2_SHORT_CONV_BLOCK_DIM",
    "GDN2_SHORT_CONV_CHANNELS",
    "GDN2_SHORT_CONV_WIDTH",
    "GDN2_MIXED_MLP_BLOCK_DIM",
    "GDN2_MIXED_MLP_HIDDEN_SIZE",
    "GDN2_MIXED_MLP_INTERMEDIATE_SIZE",
    "fused_decode_gdn2",
    "fused_recurrent_gdn2",
    "fused_short_conv_decode_gdn2",
    "fused_norm2_w12_swiglu_gdn2",
    "prepare",
]
