"""Host ABI and audit regressions; these do not qualify device instructions."""
from __future__ import annotations

import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import _disable_current_modes

from ascend_fla.ops.kda import chunk, chunk_bwd

ROOT = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/kda_layout'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support = load('_fmt02_test_support', ROOT / 'native_support.py')
unit = load('_fmt02_test_unit', ROOT / 'unit.py')
CASES = json.loads((ROOT / 'contract.json').read_text())['cases']


def test_native_leaf_unit_survives_reference_import_collision(monkeypatch, tmp_path):
    # Readonly algorithm references may prepend another directory exporting unit.
    (tmp_path / 'unit.py').write_text("raise RuntimeError('wrong reference unit')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(sys.modules, 'native_support', support)
    monkeypatch.setitem(sys.modules, 'unit', SimpleNamespace(make_inputs=None))
    checks = load('_fmt02_test_native_checks', ROOT / 'native_checks.py')
    assert Path(checks.unit.__file__) == ROOT / 'unit.py'
    inputs = checks.unit.make_inputs(CASES[0])
    expected = checks.unit.reference(inputs)
    assert expected['destination'].numel() == CASES[0]['parameters']['N']


def test_native_decode_audit_preserves_bd3_rejection(monkeypatch):
    monkeypatch.setitem(sys.modules, 'native_support', support)
    checks = load('_fmt02_test_decode_guard', ROOT / 'native_checks.py')
    api = importlib.import_module('ascend_fla.ops.kda.fused_recurrent')
    monkeypatch.setattr(torch.Tensor, 'npu', lambda self: self, raising=False)
    x = dict(q=torch.zeros(1,64,2,128).bfloat16(), k=torch.zeros(1,64,2,128).bfloat16(),
             v=torch.zeros(1,64,4,128).bfloat16(), g=torch.full((1,64,4,128),-.1),
             beta=torch.full((1,64,4),.5), h0=torch.zeros(1,4,128,128))
    real = SimpleNamespace(make_inputs=lambda **kwargs: x, ref_fwd=lambda inputs: None)
    result = checks.decode_audit(api, real, 3)
    assert result['passed'] and len(result['cases']) == 2
    assert all(row['rejected'] and not row['launches'] for row in result['cases'])


@pytest.fixture
def layout(monkeypatch):
    runtime = chunk._layout_runtime()
    calls = []

    def native(key):
        def call(inputs, scalars, outputs):
            # CPU stand-in at the compiled ABI boundary only. Audit still sees
            # every real host-wrapper operation before and after this boundary.
            with _disable_current_modes():
                destination = outputs['destination']
                calls.append((key, dict(scalars)))
                if key.startswith('zero'):
                    destination.zero_()
                else:
                    shape = tuple(scalars[f'D{i}'] for i in range(1, 5))
                    shape = (scalars['N'] // math.prod(shape),) + shape
                    strides = tuple(scalars[f'S{i}'] for i in range(5))
                    source = inputs['source']
                    value = source.as_strided(shape, strides, source.storage_offset())
                    if scalars.get('multiply'):
                        value = value * scalars['factor']
                    destination.copy_(value.contiguous().view_as(destination).to(destination.dtype))
                return outputs
        return call

    keys = ('bf16_bf16', 'bf16_f32', 'f32_bf16', 'f32_f32', 'zero_bf16', 'zero_f32')
    monkeypatch.setattr(runtime, 'prepare', lambda *a, **kw: {key: native(key) for key in keys})
    return runtime, calls


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and support.digest(a) == support.digest(b)


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
def test_portable_case_runtime_and_independent_reference(layout, case):
    runtime, calls = layout
    x = unit.make_inputs(case)
    original = support.digest(x['source'])
    expected = unit.reference(x)['destination']
    key, shape, scalars = unit.launch_description(x)
    audit = support.Audit()
    with audit:
        if key.startswith('zero'):
            actual = runtime.zeros(shape, unit.DTYPES[case['parameters']['destination']], 'cpu')
        else:
            actual = runtime.move(x['source'], shape, tuple(scalars[f'S{i}'] for i in range(5)),
                                  dtype=unit.DTYPES[case['parameters']['destination']],
                                  multiply=bool(scalars.get('multiply', False)),
                                  factor=scalars.get('factor', 1))
    assert byte_equal(actual.view_as(expected), expected)
    assert support.digest(x['source']) == original
    assert len(calls) == 1
    assert not audit.unexpected(), audit.unexpected()


@pytest.mark.parametrize('c', (1, 2, 3))
@pytest.mark.parametrize('heads', (1, 2, 4, 8, 32))
@pytest.mark.parametrize('width', (None, 64, 128))
def test_actual_wrapper_mapping_matches_pinned_predecessor(layout, c, heads, width):
    auto = importlib.import_module('ascend_fla.ops.kda.autograd')
    before, _, _ = support.baseline(chunk, chunk_bwd, auto)
    shape = (2, c*64, heads) + (() if width is None else (width,))
    source = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    want = before._to_bhcld(source, heads, on_cpu=False)
    with support.Audit() as audit:
        got = chunk._to_bhcld(source, heads)
    assert byte_equal(got, want)
    assert not audit.unexpected(), audit.unexpected()
    if width:
        with support.Audit() as audit:
            restored = chunk._from_bhcld(got)
        assert byte_equal(restored, source)
        assert not audit.unexpected(), audit.unexpected()


@pytest.mark.parametrize('form', ('transposed', 'broadcast', 'offset'))
@pytest.mark.parametrize('dtype', (torch.bfloat16, torch.float32))
def test_gradient_views_are_packed_without_host_conversion(layout, form, dtype):
    runtime, calls = layout
    if form == 'transposed':
        source = torch.randn(2, 3, 64, 128).transpose(1, 2)
    elif form == 'broadcast':
        source = torch.tensor(-0.5).expand(2, 64, 3, 128)
    else:
        source = torch.randn(2, 64, 3, 128, 2)[..., 1]
    expected = source.contiguous().to(dtype)
    with support.Audit() as audit:
        got = runtime.cast(source, dtype)
    assert byte_equal(got, expected)
    assert len(calls) == 1 and not audit.unexpected()


@pytest.mark.parametrize('missing_state_gradient', (False, True))
def test_autograd_backward_casts_and_missing_state_remain_exact(layout, monkeypatch, missing_state_gradient):
    auto = importlib.import_module('ascend_fla.ops.kda.autograd')
    q = torch.randn(1, 64, 2, 128).bfloat16()
    v = torch.randn(1, 64, 4, 128).bfloat16()
    beta = torch.rand(1, 64, 4).bfloat16()
    do = torch.tensor(0.25).expand_as(v)
    dht = None if missing_state_gradient else torch.randn(1, 4, 128, 128).transpose(-1, -2)
    expected_do = do.contiguous().bfloat16()
    expected_dht = torch.zeros(1, 4, 128, 128).bfloat16() if dht is None else dht.contiguous().bfloat16()
    gradients = dict(dq=q.clone(), dk=-q, dv=v.clone(), dg=-v,
                     dbeta=beta.clone(), dh0=torch.randn(1, 4, 128, 128).bfloat16())
    ctx = SimpleNamespace(saved_tensors=(q, q, v, beta, *(torch.empty(1) for _ in range(9))),
                          bwd_options=dict(device='a5', block_dim=4, impl='stable'),
                          state_shape=(1, 4, 128, 128), needs_input_grad=(True,)*13)

    def backward(**kwargs):
        with _disable_current_modes():
            assert byte_equal(kwargs['do'], expected_do)
            assert byte_equal(kwargs['dht'], expected_dht)
        return gradients

    monkeypatch.setattr(auto, 'chunk_kda_bwd', backward)
    with support.Audit() as audit:
        result = auto._ChunkKDA.backward(ctx, do, dht)
    expected = (gradients['dq'], gradients['dk'], gradients['dv'], gradients['dg'].float(),
                gradients['dbeta'].float(), None, gradients['dh0'].float(), None, None, None, None, None, None)
    assert len(result) == len(expected)
    assert all(a is b is None or byte_equal(a, b) for a, b in zip(result, expected))
    assert not audit.unexpected(), audit.unexpected()


@pytest.mark.parametrize('mutation', ('sign', 'zero', 'head_order', 'dtype', 'missing_write'))
def test_bitwise_verifier_rejects_corrupt_outputs(mutation):
    reference = torch.randn(2, 64, 4, 128).bfloat16()
    actual = reference.clone()
    if mutation == 'sign': actual.neg_()
    elif mutation == 'zero': actual.zero_()
    elif mutation == 'head_order': actual = actual.flip(2)
    elif mutation == 'dtype': actual = actual.float()
    else: actual[0, 0, 0, 0] = float('nan')
    assert not support.exact({'o': actual}, {'o': reference})['o']['passed']


def test_audit_rejects_host_cast_copy_and_arithmetic():
    source = torch.randn(2, 128)
    with support.Audit() as audit:
        source.bfloat16()
        source.t().contiguous()
        source + 1
    unexpected = {r['operator'] for r in audit.unexpected()}
    assert {'aten._to_copy.default', 'aten.clone.default', 'aten.add.Tensor'} <= unexpected


def test_cpu_layout_alias_is_rejected():
    for name in ('auto', 'npu'):
        assert chunk._resolve_layout(name) is False
    with pytest.raises(ValueError, match='D-PM-37'):
        chunk._resolve_layout('cpu')


@pytest.mark.parametrize('shape', [(2,3,64,3,64), (1,1,1,1,4160),
                                   (1,1,2,64,192), (2,3,3,1,64), (2,2,2,1,128)])
def test_batched_tiles_own_contiguous_complete_output_intervals(shape):
    runtime = chunk._layout_runtime()
    count, tiles = runtime.tile_shape(shape)
    total = math.prod(shape)
    assert count <= 4096 and count % 64 == 0 and total % count == 0
    indices = torch.arange(total).reshape(shape)
    for offset in range(0, total, count):
        rest = offset
        coordinates = []
        for dimension in reversed(shape):
            coordinates.append(rest % dimension)
            rest //= dimension
        coordinates.reverse()
        block = indices[tuple(slice(start, start + size) for start, size in zip(coordinates, tiles))]
        assert block.numel() == count
        assert torch.equal(block.reshape(-1), torch.arange(offset, offset + count))
