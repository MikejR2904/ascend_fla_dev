"""Raw-input preparation and chunk-only domain heuristics; no NPU required."""
from __future__ import annotations

import itertools

import pytest
import torch

from ascend_fla.ops.kda.chunk import _QK_NORM_TOL, _check_input_domain, _prepare_inputs


def inputs():
    rng = torch.Generator().manual_seed(244)
    q, k = [torch.randn(1, 64, 1, 128, generator=rng) for _ in range(2)]
    g = torch.randn(1, 64, 2, 128, generator=rng) * .1
    beta = torch.randn(1, 64, 2, generator=rng)
    a = torch.tensor([-.2, .1])
    bias = torch.linspace(-3., -2., 256)
    return q, k, g, beta, a, bias


def independent_prepare(q, k, g, beta, a, bias):
    # Written independently from production preprocessing, evaluated in FP32.
    q = q.float() / (torch.sum(q.float() * q.float(), -1, keepdim=True) + 1e-6).sqrt()
    k = k.float() / (torch.sum(k.float() * k.float(), -1, keepdim=True) + 1e-6).sqrt()
    gate = -torch.exp(a.float())[None, None, :, None] * torch.nn.functional.softplus(
        g.float() + bias.float().reshape(1, 1, 2, 128))
    return q.bfloat16(), k.bfloat16(), gate, torch.sigmoid(beta.float())


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=3)))
def test_raw_flags_compose_and_preserve_disabled_inputs(flags):
    raw = inputs()
    expected = independent_prepare(*raw)
    selected = [raw[0] if flags[0] else expected[0], raw[1] if flags[0] else expected[1],
                raw[2] if flags[1] else expected[2], raw[3] if flags[2] else expected[3]]
    actual = _prepare_inputs(*selected, A_log=raw[4], dt_bias=raw[5],
                             use_qk_l2norm_in_kernel=flags[0], use_gate_in_kernel=flags[1],
                             use_beta_sigmoid_in_kernel=flags[2])
    for i, (got, want) in enumerate(zip(actual, expected)):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        if not flags[(0, 0, 1, 2)[i]]:
            assert got is selected[i]


@pytest.mark.parametrize('missing', ['A_log', 'dt_bias'])
def test_raw_gate_requires_both_parameters(missing):
    q, k, g, beta, a, bias = inputs()
    kw = dict(A_log=a, dt_bias=bias, use_gate_in_kernel=True)
    kw[missing] = None
    with pytest.raises(ValueError, match=missing):
        _prepare_inputs(q, k, g, beta, **kw)


@pytest.mark.parametrize('name,bad', [('A_log', torch.zeros(2, 1)), ('dt_bias', torch.zeros(2, 128)),
                                     ('dt_bias', torch.zeros(128))])
def test_raw_gate_parameters_have_exact_gva_shapes(name, bad):
    q, k, g, beta, a, bias = inputs()
    kw = dict(A_log=a, dt_bias=bias, use_gate_in_kernel=True)
    kw[name] = bad
    with pytest.raises(ValueError, match=name):
        _prepare_inputs(q, k, g, beta, **kw)


@pytest.mark.parametrize('index,flag', [(0, 'use_qk_l2norm_in_kernel'), (1, 'use_qk_l2norm_in_kernel'),
                                      (2, 'use_gate_in_kernel'), (3, 'use_beta_sigmoid_in_kernel')])
def test_preparation_does_not_silently_fix_noncontiguous_raw_inputs(index, flag):
    q, k, g, beta, a, bias = inputs()
    values = [q, k, g, beta]
    values[index] = values[index].transpose(0, 1).contiguous().transpose(0, 1)
    # B=1 transpose can still be contiguous; use a strided last dimension.
    values[index] = torch.stack([values[index], values[index]], -1)[..., 0]
    assert not values[index].is_contiguous()
    with pytest.raises(ValueError, match='contiguous'):
        _prepare_inputs(*values, A_log=a, dt_bias=bias, **{flag: True})


@pytest.mark.parametrize('name,flag', [('q', 'use_qk_l2norm_in_kernel'), ('k', 'use_qk_l2norm_in_kernel'),
                                     ('g', 'use_gate_in_kernel'), ('beta', 'use_beta_sigmoid_in_kernel')])
@pytest.mark.parametrize('bad', [2., float('nan'), float('inf')])
def test_each_bad_prepared_route_names_the_raw_flag(name, flag, bad):
    values = dict(zip(('q', 'k', 'g', 'beta'), independent_prepare(*inputs())))
    values[name] = torch.full_like(values[name], bad)
    with pytest.raises(ValueError, match=rf'{name}.*{flag}'):
        _check_input_domain(**values)


def test_valid_prepared_and_small_raw_vectors_pass_heuristic():
    q, k, g, beta = independent_prepare(*inputs())
    _check_input_domain(q, k, g, beta)
    # Explicit limitation: an unnormalized but small raw vector is indistinguishable.
    _check_input_domain(q.float() * .01, k.float() * .01, g, beta)


def test_bf16_calibrated_margin_accepts_correlated_rounding():
    # Calibration: torch2.10 CPU, K128, seeds244..251, 32768 Gaussian rows/seed
    # at scales1e-4,1e-2,1,30; add equal nonzero component counts1..128.
    # Total1,048,704 rows; max norm1.0027123859343965. The chosen 2^-8
    # BF16 round-to-nearest bound covers that measured maximum, unlike 1e-3.
    x = torch.tril(torch.ones(128, 128)) * 30
    y = x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    norms = y.bfloat16().double().norm(dim=-1)
    assert norms.max().item() == pytest.approx(1.0027123859343965, abs=1e-14)
    assert .0027 < norms.max().item() - 1 < _QK_NORM_TOL == 2**-8
    q = y.bfloat16().reshape(1, 128, 1, 128)
    _check_input_domain(q, q, torch.full_like(q.float(), -.1), torch.full((1, 128, 1), .5))


def test_raw_preparation_keeps_all_six_gradient_paths():
    raw = tuple(x.clone().requires_grad_() for x in inputs())
    q, k, g, beta = _prepare_inputs(*raw[:4], A_log=raw[4], dt_bias=raw[5],
                                   use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                                   use_beta_sigmoid_in_kernel=True)
    # Nonconstant upstream weights avoid a normalization-invariant loss.
    rng = torch.Generator().manual_seed(245)
    gradients = [torch.randn(x.shape, generator=rng) for x in (q, k, g, beta)]
    loss = sum((x.float() * dy).sum() for x, dy in zip((q, k, g, beta), gradients))
    actual = torch.autograd.grad(loss, raw)
    reference_raw = tuple(x.detach().clone().requires_grad_() for x in raw)
    reference = independent_prepare(*reference_raw)
    expected = torch.autograd.grad(sum((x.float() * dy).sum() for x, dy in zip(reference, gradients)), reference_raw)
    for got, want in zip(actual, expected):
        assert torch.isfinite(got).all() and torch.count_nonzero(got)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=3)))
def test_public_decode_flags_reach_the_existing_fp32_abi(monkeypatch, flags):
    from ascend_fla.ops.kda import fused_recurrent as decode
    q, k, g, beta, a, bias = inputs()
    raw = [q[:, :2].contiguous(), k[:, :2].contiguous(), g[:, :2].contiguous(), beta[:, :2].contiguous()]
    expected = independent_prepare(*raw, a, bias)
    selected = [raw[0] if flags[0] else expected[0], raw[1] if flags[0] else expected[1],
                raw[2] if flags[1] else expected[2], raw[3] if flags[2] else expected[3]]
    v = torch.ones(1, 2, 2, 128, dtype=torch.bfloat16)
    captured = {}

    def kernel(ins, scalars, outs):
        captured.update(ins)
        for x in outs.values():
            x.zero_()

    monkeypatch.setattr(decode, '_compiled', lambda *args: kernel)
    decode.fused_recurrent_kda(selected[0], selected[1], v, selected[2], selected[3],
                               A_log=a, dt_bias=bias, use_qk_l2norm_in_kernel=flags[0],
                               use_gate_in_kernel=flags[1], use_beta_sigmoid_in_kernel=flags[2])
    # Existing decode scale multiplication rounds in q's dtype BEFORE FP32 layout conversion.
    expected_q = (expected[0].repeat_interleave(2, dim=2) * (128**-.5)).float().permute(0, 2, 1, 3)
    torch.testing.assert_close(captured['qs'], expected_q, rtol=0, atol=0)
    torch.testing.assert_close(captured['k'], expected[1].repeat_interleave(2, dim=2).float().permute(0, 2, 1, 3), rtol=0, atol=0)
    torch.testing.assert_close(captured['g'], expected[2].permute(0, 2, 1, 3), rtol=0, atol=0)
    torch.testing.assert_close(captured['beta'], expected[3].permute(0, 2, 1).reshape(1, 2, 1, 2), rtol=0, atol=0)


@pytest.mark.parametrize('check_domain', [True, False])
def test_decode_never_runs_chunk_domain_heuristics(monkeypatch, check_domain):
    from ascend_fla.ops.kda import fused_recurrent as decode
    reached = []

    def kernel(ins, scalars, outs):
        reached.append(True)
        for x in outs.values():
            x.zero_()

    monkeypatch.setattr(decode, '_compiled', lambda *a: kernel)
    x = torch.full((1, 1, 1, 128), 2.)
    decode.fused_recurrent_kda(x, x, x, x, torch.full((1, 1, 1), -2.), check_domain=check_domain)
    assert reached == [True]


@pytest.mark.parametrize('flag', ['allow_neg_eigval', 'lower_bound', 'safe_gate'])
def test_unsupported_decode_flags_still_raise(flag):
    from ascend_fla.ops.kda import fused_recurrent_kda
    x = torch.zeros(1, 1, 1, 128)
    with pytest.raises(TypeError, match=flag):
        fused_recurrent_kda(x, x, x, x, torch.ones(1, 1, 1), **{flag: True})


@pytest.mark.parametrize('flags', list(itertools.product((False, True), repeat=3)))
def test_public_chunk_flags_reach_prepared_boundary(monkeypatch, flags):
    from ascend_fla.ops.kda import autograd, chunk_kda
    raw = inputs()
    expected = independent_prepare(*raw)
    selected = [raw[0] if flags[0] else expected[0], raw[1] if flags[0] else expected[1],
                raw[2] if flags[1] else expected[2], raw[3] if flags[2] else expected[3]]
    v = torch.zeros(1, 64, 2, 128, dtype=torch.bfloat16)
    captured = []

    def boundary(q, k, value, g, beta, *args, **kwargs):
        assert value is v
        captured.extend((q, k, g, beta))
        return v, None

    monkeypatch.setattr(autograd, 'chunk_kda_fwd', boundary)
    assert chunk_kda.__module__ == 'ascend_fla.ops.kda.autograd'
    chunk_kda(selected[0], selected[1], v, selected[2], selected[3], A_log=raw[4], dt_bias=raw[5],
              use_qk_l2norm_in_kernel=flags[0], use_gate_in_kernel=flags[1],
              use_beta_sigmoid_in_kernel=flags[2])
    for i, (got, want) in enumerate(zip(captured, expected)):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
        if not flags[(0, 0, 1, 2)[i]]:
            assert got is selected[i]


@pytest.mark.parametrize('parameter', ['A_log', 'dt_bias'])
def test_public_chunk_routes_gate_parameter_only_gradients_to_autograd(monkeypatch, parameter):
    from ascend_fla.ops.kda import autograd, chunk_kda
    q, k, g, beta, a, bias = inputs()
    q, k, _, beta = independent_prepare(q, k, g, beta, a, bias)
    a.requires_grad_(parameter == 'A_log')
    bias.requires_grad_(parameter == 'dt_bias')
    seen = []

    def apply(q, k, v, transformed_g, beta, *args):
        assert transformed_g.requires_grad
        seen.append(True)
        # The native tests exercise the real Function; here the kernel boundary
        # is replaced to inspect the new preparation graph without an NPU.
        return transformed_g.sum(), None

    monkeypatch.setattr(autograd._ChunkKDA, 'apply', apply)
    monkeypatch.setattr(autograd, 'chunk_kda_fwd', lambda *a, **kw: pytest.fail('lost parameter gradient'))
    o, _ = chunk_kda(q, k, torch.zeros_like(g).bfloat16(), g, beta, A_log=a, dt_bias=bias,
                     use_gate_in_kernel=True)
    o.backward()
    assert seen == [True]
    assert torch.count_nonzero((a if parameter == 'A_log' else bias).grad)


@pytest.mark.parametrize('name,index,flag', [('q', 0, 'use_qk_l2norm_in_kernel'),
                                           ('k', 1, 'use_qk_l2norm_in_kernel'),
                                           ('g', 2, 'use_gate_in_kernel'),
                                           ('beta', 3, 'use_beta_sigmoid_in_kernel')])
def test_public_chunk_rejects_each_raw_route_before_launch(name, index, flag):
    from ascend_fla.ops.kda import chunk_kda
    x = list(independent_prepare(*inputs()))
    x[index] = torch.full_like(x[index], 2.)
    with pytest.raises(ValueError, match=rf'{name}.*{flag}'):
        chunk_kda(x[0], x[1], torch.zeros_like(x[2]).bfloat16(), x[2], x[3])


@pytest.mark.parametrize('bad', [0., 1.])
def test_beta_open_interval_boundaries_are_rejected(bad):
    q, k, g, beta = independent_prepare(*inputs())
    with pytest.raises(ValueError, match='beta.*use_beta_sigmoid_in_kernel'):
        _check_input_domain(q, k, g, torch.full_like(beta, bad))


def test_domain_optout_preserves_the_chunk_abi_dtype_guard():
    from ascend_fla.ops.kda import chunk_kda
    q, k, g, beta = independent_prepare(*inputs())
    with pytest.raises(ValueError, match='q.*dtype'):
        chunk_kda(q.float(), k, torch.zeros_like(g).bfloat16(), g, beta, check_domain=False)


@pytest.mark.parametrize('entry_name', ['chunk_kda', 'fused_recurrent_kda'])
@pytest.mark.parametrize('missing', ['A_log', 'dt_bias'])
def test_public_entries_reject_missing_raw_gate_parameters(entry_name, missing):
    from ascend_fla.ops import kda
    q, k, g, beta, a, bias = inputs()
    opts = dict(A_log=a, dt_bias=bias, use_gate_in_kernel=True)
    opts[missing] = None
    t = 1 if entry_name == 'fused_recurrent_kda' else 64
    with pytest.raises(ValueError, match=missing):
        getattr(kda, entry_name)(q[:, :t], k[:, :t], torch.zeros_like(g[:, :t]).bfloat16(),
                                 g[:, :t], beta[:, :t], **opts)


@pytest.mark.parametrize('flag', ['allow_neg_eigval', 'lower_bound', 'safe_gate'])
def test_unsupported_chunk_flags_still_raise(flag):
    from ascend_fla.ops.kda import chunk_kda
    q, k, g, beta = independent_prepare(*inputs())
    with pytest.raises(TypeError, match=flag):
        chunk_kda(q, k, torch.zeros_like(g).bfloat16(), g, beta, **{flag: True})


def calibrate_bf16_norm_domain():
    """Reproduce the full calibration and emit machine-independent raw data.

    Run ``python tests/test_kda_domain_checks.py`` in the CPU test environment.
    This is a diagnostic sweep, separate from the fast pytest regression.
    """
    records = []
    for seed in range(244, 252):
        x = torch.randn(32768, 128, generator=torch.Generator().manual_seed(seed))
        for scale in (1e-4, 1e-2, 1., 30.):
            z = x * scale
            normalized = z / (z.square().sum(-1, keepdim=True) + 1e-6).sqrt()
            records.append(dict(kind='gaussian', seed=seed, scale=scale, rows=32768,
                                max_norm=normalized.bfloat16().double().norm(dim=-1).max().item()))
    for count in range(1, 129):
        x = torch.zeros(128)
        x[:count] = 30.
        normalized = x / (x.square().sum() + 1e-6).sqrt()
        records.append(dict(kind='equal_nonzero', count=count, rows=1,
                            max_norm=normalized.bfloat16().double().norm().item()))
    return dict(torch=torch.__version__,dimension=128,rows=sum(r['rows'] for r in records),
                maximum=max(r['max_norm'] for r in records),tolerance=_QK_NORM_TOL,records=records)


if __name__ == '__main__':
    import json
    print(json.dumps(calibrate_bf16_norm_domain(), indent=2))


@pytest.mark.parametrize('check_domain', [True, False])
def test_malformed_ranks_keep_the_existing_shape_error(check_domain):
    from ascend_fla.ops.kda import chunk_kda
    x = torch.zeros(1)
    # Real entrypoint, no dispatch mocks: beta=0 must not mask malformed ABI.
    with pytest.raises(ValueError, match='q/k/v.*4'):
        chunk_kda(x, x, x, x, x, check_domain=check_domain)
