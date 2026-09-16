"""Host-side ABI gates for the CCE GDN-2 recurrent operator."""
from __future__ import annotations

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ascend_fla.models import GDN2Config, GDN2ForCausalLM  # noqa: E402
from ascend_fla.ops.gdn2 import (  # noqa: E402
    GDN2_DECODE_HEADS,
    GDN2_RECURRENT_BLOCK_DIM,
    GDN2_RECURRENT_T_MAX,
    GDN2_SHORT_CONV_CHANNELS,
    GDN2_SHORT_CONV_WIDTH,
    fused_decode_gdn2,
    fused_recurrent_gdn2,
    fused_short_conv_decode_gdn2,
    prepare,
)


def _inputs(time: int = 1, heads: int = 1) -> tuple[torch.Tensor, ...]:
    shape = (1, time, heads, 128)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    g = -torch.rand(shape)
    b = torch.rand(shape)
    w = torch.rand(shape)
    state = torch.randn(1, heads, 128, 128)
    return q, k, v, g, b, w, state


def test_declared_cce_domain_is_fixed_before_compile() -> None:
    q, k, v, g, b, w, state = _inputs()
    with pytest.raises(ValueError, match="must be on NPU"):
        fused_recurrent_gdn2(q, k, v, g, b, w, initial_state=state)

    q, k, v, g, b, w, state = _inputs(time=GDN2_RECURRENT_T_MAX + 1)
    with pytest.raises(ValueError, match="longer sequences"):
        fused_recurrent_gdn2(q, k, v, g, b, w, initial_state=state)

    q, k, v, g, b, w, state = _inputs(heads=2)
    with pytest.raises(ValueError, match="H in"):
        fused_recurrent_gdn2(q, k, v, g, b, w, initial_state=state)


def test_backend_options_do_not_fallback() -> None:
    with pytest.raises(ValueError, match="block_dim"):
        prepare(block_dim=GDN2_RECURRENT_BLOCK_DIM // 2)
    with pytest.raises(ValueError, match="core_block_dim"):
        GDN2ForCausalLM(
            GDN2Config(
                vocab_size=32,
                padding_multiple=8,
                block_size=16,
                n_layer=1,
                n_embd=16,
                intermediate_size=24,
                num_heads=1,
                num_v_heads=1,
                head_dim=128,
            ),
            core_backend="cce",
            core_block_dim=4,
        )
    with pytest.raises(ValueError, match="NPU device"):
        GDN2ForCausalLM.from_checkpoint(
            pathlib.Path("does-not-need-to-exist.pth"),
            device="cpu",
            core_backend="cce",
        )


def test_fused_decode_host_gate_rejects_non_npu_inputs() -> None:
    shape = (1, 1, GDN2_DECODE_HEADS, 128)
    raw = [torch.zeros(shape, dtype=torch.bfloat16) for _ in range(7)]
    decay = -torch.ones(GDN2_DECODE_HEADS, 128)
    bias = torch.zeros(GDN2_DECODE_HEADS, 128)
    weight = torch.ones(128)
    state = torch.zeros(1, GDN2_DECODE_HEADS, 128, 128)
    with torch.inference_mode(), pytest.raises(ValueError, match="must be on NPU"):
        fused_decode_gdn2(*raw, decay, bias, weight, initial_state=state)


def test_short_conv_decode_host_gate_rejects_non_npu_inputs() -> None:
    x = torch.zeros(1, 1, GDN2_SHORT_CONV_CHANNELS, dtype=torch.bfloat16)
    cache = torch.zeros(
        1,
        GDN2_SHORT_CONV_CHANNELS,
        GDN2_SHORT_CONV_WIDTH,
        dtype=torch.bfloat16,
    )
    weight = torch.zeros(
        GDN2_SHORT_CONV_CHANNELS,
        1,
        GDN2_SHORT_CONV_WIDTH,
        dtype=torch.bfloat16,
    )
    with torch.inference_mode(), pytest.raises(ValueError, match="must be on NPU"):
        fused_short_conv_decode_gdn2(x, cache, weight)
