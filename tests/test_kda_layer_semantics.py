"""KDA host semantics, independent of the NPU kernel implementation.

Source: FLA v0.5.2 (9c8e42e), modules/l2norm.py and layers/kda.py.
The operator boundary is captured; projections/activation/normalization run.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ascend_fla.layers.kda import KimiDeltaAttention
from ascend_fla.ops.kda import autograd


def _capture(monkeypatch):
    captured = {}

    def forward(q, k, v, g, beta, *args, **kwargs):
        captured.update(q=q, k=k, v=v, g=g, beta=beta)
        return torch.zeros_like(v), None

    monkeypatch.setattr(autograd, "chunk_kda_fwd", forward)
    return captured


@pytest.mark.parametrize("norm", [0., 1e-4, 1e-2, 1., 30.])
def test_layer_l2norm_uses_additive_squared_epsilon(monkeypatch, norm):
    torch.manual_seed(244)
    layer = KimiDeltaAttention(hidden_size=128, num_heads=1, use_short_conv=True)
    # Replace only the convolution output, so this isolates normalization from
    # the independent no-short-convolution activation regression below.
    direction = torch.linspace(-1., 1., 128, dtype=torch.float64)
    vector = (direction / direction.norm() * norm).float()
    x = vector.view(1, 1, 128).expand(1, 64, 128).contiguous()
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        monkeypatch.setattr(getattr(layer, name), "forward", lambda *a, **kw: (x, None))
    captured = _capture(monkeypatch)
    with torch.no_grad():
        layer(torch.zeros(1, 64, 128))
    expected = (x.double() / torch.sqrt(x.double().square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
    for name in ("q", "k"):
        torch.testing.assert_close(captured[name].view_as(expected), expected, rtol=0, atol=0)
    if norm == 1e-4:
        assert .099 < captured['q'][0, 0, 0].float().norm().item() < .100


@pytest.mark.parametrize("with_cache", [False, True])
def test_layer_without_short_conv_activates_all_three_projections(monkeypatch, with_cache):
    torch.manual_seed(244)
    layer = KimiDeltaAttention(hidden_size=128, num_heads=1, num_v_heads=2, use_short_conv=False)
    x = torch.randn(1, 64, 128)
    captured = _capture(monkeypatch)
    cache = {} if with_cache else None
    with torch.no_grad():
        expected = {}
        for name in ("q", "k", "v"):
            projection = getattr(layer, name + "_proj")(x)
            z = F.silu(projection).double().view(1, 64, -1, 128)
            if name != "v":
                z = z / torch.sqrt(z.square().sum(-1, keepdim=True) + 1e-6)
            expected[name] = z.bfloat16()
        layer(x, cache=cache)
    for name in expected:
        torch.testing.assert_close(captured[name], expected[name], rtol=0, atol=0)
    if with_cache:
        assert cache['conv_state'] == (None, None, None)
