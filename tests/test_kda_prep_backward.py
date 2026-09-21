"""Training graph/ABI checks with explicit test-only CPU vendor substitutes."""
import itertools
from types import SimpleNamespace

import pytest
import torch

from ascend_fla.ops.kda import autograd, chunk, chunk_bwd


@pytest.fixture
def preparation_abi(monkeypatch):
    events = []

    class PrepABI:
        @staticmethod
        def _check_source(name, value):
            assert value.device.type == 'cpu' and value.is_contiguous()
            assert value.dtype in (torch.bfloat16, torch.float32)

        @staticmethod
        def prepare_backward(*args):
            events.append('raw_backward_ready')

        @staticmethod
        def norm(value, dtype, **options):
            assert events[:3] == ['forward_ready', 'backward_ready', 'raw_backward_ready']
            events.append('norm')
            z = value.float()
            return (z / (z.square().sum(-1, keepdim=True) + 1e-6).sqrt()).to(dtype)

        @staticmethod
        def gate(value, alog, bias, **options):
            assert events[:3] == ['forward_ready', 'backward_ready', 'raw_backward_ready']
            events.append('gate')
            return -alog.float().exp().view(-1, 1) * torch.nn.functional.softplus(
                value.float() + bias.float().view(value.shape[-2:]))

        @staticmethod
        def beta(value, **options):
            assert events[:3] == ['forward_ready', 'backward_ready', 'raw_backward_ready']
            events.append('beta')
            return value.float().sigmoid()

        @staticmethod
        def norm_backward(value, sensitivity, **options):
            events.append('norm_backward')
            assert sensitivity.dtype == torch.bfloat16 and sensitivity.is_contiguous()
            with torch.enable_grad():
                leaf = value.detach().clone().requires_grad_()
                return torch.autograd.grad(PrepABI.norm(leaf, torch.bfloat16), leaf, sensitivity)[0]

        @staticmethod
        def gate_backward(value, alog, bias, sensitivity, **options):
            events.append('gate_backward')
            assert sensitivity.dtype == torch.float32 and sensitivity.is_contiguous()
            with torch.enable_grad():
                leaves = [t.detach().clone().requires_grad_() for t in (value, alog, bias)]
                return torch.autograd.grad(PrepABI.gate(*leaves), leaves, sensitivity)

        @staticmethod
        def beta_backward(probability, sensitivity, dtype, **options):
            events.append('beta_backward')
            assert sensitivity.dtype == torch.float32 and sensitivity.is_contiguous()
            return (sensitivity * probability * (1 - probability)).to(dtype)

    monkeypatch.setattr(autograd, '_prep_runtime', lambda: PrepABI)
    monkeypatch.setattr(chunk, '_compiled_chain', lambda *a: events.append('forward_ready'))
    monkeypatch.setattr(chunk_bwd, '_compiled_chain', lambda *a: events.append('backward_ready'))
    monkeypatch.setattr(autograd, '_layout_runtime', lambda: SimpleNamespace(
        cast=lambda x, dtype, **kw: x.to(dtype).contiguous()))
    monkeypatch.setattr(chunk, '_prepare_inputs', lambda *a, **kw: pytest.fail('old host preparation called'))
    monkeypatch.setattr(autograd, '_prepare_inputs', lambda *a, **kw: pytest.fail('old host training called'))
    return events


def inputs(dtype):
    generator = torch.Generator().manual_seed(8008)
    q = torch.randn((1, 64, 2, 128), generator=generator).to(dtype)
    k = torch.randn(q.shape, generator=generator).to(dtype)
    g = torch.randn((1, 64, 4, 128), generator=generator).to(dtype)
    beta = torch.randn((1, 64, 4), generator=generator).to(dtype)
    a = torch.linspace(-2., -1., 4).to(dtype)
    bias = torch.linspace(-.2, .2, 512).to(dtype)
    return q, k, g, beta, a, bias


def independent(values, flags):
    q, k, g, beta, a, bias = values
    if flags[0]:
        x, y = q.float(), k.float()
        q = (x / (torch.sum(x.square(), -1, keepdim=True) + 1e-6).sqrt()).bfloat16()
        k = (y / (torch.sum(y.square(), -1, keepdim=True) + 1e-6).sqrt()).bfloat16()
    if flags[1]:
        g = -torch.exp(a.float()).view(-1, 1) * torch.nn.functional.softplus(
            g.float() + bias.float().view(g.shape[-2:]))
    if flags[2]:
        beta = torch.sigmoid(beta.float())
    return q, k, g, beta


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=3)))
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@pytest.mark.parametrize('selected', [(0, 1, 2, 3, 4, 5), (0, 2, 5), (4,), (5,)])
def test_flag_and_partial_gradient_graph(preparation_abi, flags, dtype, selected):
    values = tuple(t.requires_grad_(i in selected) for i, t in enumerate(inputs(dtype)))
    before = [t.detach().clone() for t in values]
    reference_values = tuple(t.detach().clone().requires_grad_(i in selected) for i, t in enumerate(values))
    options = dict(use_qk_l2norm_in_kernel=flags[0], use_gate_in_kernel=flags[1],
                   use_beta_sigmoid_in_kernel=flags[2])
    result = autograd._prepare_training_inputs(*values[:4], A_log=values[4], dt_bias=values[5], **options)
    reference = independent(reference_values, flags)
    generator = torch.Generator().manual_seed(8108)
    sensitivities = [torch.randn(t.shape, generator=generator).to(t.dtype) for t in result]
    actual_outputs = [t for t in result if t.requires_grad]
    expected_outputs = [t for t in reference if t.requires_grad]
    if actual_outputs:
        gs = [g for t, g in zip(result, sensitivities) if t.requires_grad]
        actual = torch.autograd.grad(actual_outputs, [values[i] for i in selected], gs, allow_unused=True)
        expected = torch.autograd.grad(expected_outputs, [reference_values[i] for i in selected], gs, allow_unused=True)
        for index, got, want in zip(selected, actual, expected):
            if want is None:
                assert got is None
            else:
                assert got.dtype == values[index].dtype
                assert torch.isfinite(got).all() and torch.count_nonzero(got)
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
    else:
        assert not expected_outputs  # Disabled parameter flags create no edge.
    for i, (got, want) in enumerate(zip(result, reference)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        if not flags[(0, 0, 1, 2)[i]]:
            assert got is values[i]
    for got, old in zip(values, before):
        torch.testing.assert_close(got, old, rtol=0, atol=0)
    if not any(flags):
        assert preparation_abi == []


@pytest.mark.parametrize('types', list(itertools.product((torch.bfloat16, torch.float32), repeat=3)))
@pytest.mark.parametrize('selected', [(2,), (4,), (5,), (2, 4, 5)])
def test_independent_gate_parameter_dtypes(preparation_abi, types, selected):
    values = list(inputs(torch.float32))
    for i, dtype in zip((2, 4, 5), types):
        values[i] = values[i].to(dtype).requires_grad_(i in selected)
    result = autograd._prepare_training_inputs(*values[:4], A_log=values[4], dt_bias=values[5], use_gate_in_kernel=True)[2]
    assert result.dtype == torch.float32 and result.requires_grad
    sensitivity = torch.randn(result.shape, generator=torch.Generator().manual_seed(8118))
    grads = torch.autograd.grad(result, [values[i] for i in selected], sensitivity)
    for index, grad in zip(selected, grads):
        assert grad.dtype == values[index].dtype
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
    assert preparation_abi.count('gate_backward') == 1


def test_chunk_compile_includes_future_training_vendors(monkeypatch):
    events = []
    runtime = SimpleNamespace(prepare=lambda *a: events.append('prep_forward'),
                              prepare_backward=lambda *a: events.append('prep_backward'))
    monkeypatch.setattr(chunk, '_prep_runtime', lambda: runtime)
    monkeypatch.setattr(chunk, '_layout_runtime', lambda: SimpleNamespace(prepare=lambda *a: events.append('layout')))
    monkeypatch.setattr(chunk, 'kda_fwd_kernels', lambda *a: {})
    chunk._compiled_chain.cache_clear()
    try:
        chunk._compiled_chain('a5', 1, 'stable')
        assert events == ['layout', 'prep_forward', 'prep_backward']
    finally:
        chunk._compiled_chain.cache_clear()
