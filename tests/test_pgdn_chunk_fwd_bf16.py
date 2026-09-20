"""BF16 PGDN ABI, independent references, clamp boundary and rejection checks."""
import ast
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
from ascend_fla.ops import pgdn_chunk_fwd as public

ROOT = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/pgdn_chunk_fwd_bf16'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load('bf03_test_reference', ROOT / 'ref/reference.py')
oracle = load('bf03_test_oracle', ROOT / 'ref/oracle.py')


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(*, ratio=2, chunks=1, dtype='bfloat16', **parameters):
    return ref.make_inputs(dict(seed=73103, dtype=dtype, block_dim=1,
                               parameters=dict(B=1, T=64*chunks, N=chunks, H=2, HV=2*ratio, **parameters)))


@pytest.mark.parametrize('dtype', ['bfloat16', 'float32'])
@pytest.mark.parametrize('ratio', [1, 2, 4, 8])
@pytest.mark.parametrize('chunks', [1, 3])
def test_independent_grouped_references(dtype, ratio, chunks):
    x = inputs(dtype=dtype, ratio=ratio, chunks=chunks)
    before = {n: v.clone() for n, v in x.items()}
    expected = ref.reference(x)
    assert expected['final_A_state'].shape == (1, 2, 128)
    assert expected['final_state'].shape == (1, 2*ratio, 128, 128)
    if not os.environ.get('FLA_PGDN_NAIVE'):
        pytest.skip('pinned PGDN naive path not configured')
    actual = oracle.reference(x)
    for name in expected:
        assert ref.metric(actual[name], expected[name])['relative_l2'] <= 1e-4
    for name in x:
        assert torch.equal(x[name], before[name])


@pytest.mark.parametrize('norm', [0., 1e-13, 1e-12, 2e-12, 1e-6])
def test_clamp_uses_actual_bf16_values_in_fp32(norm):
    x = inputs(norm=norm)
    assert x['q'].dtype == x['k'].dtype == torch.bfloat16
    expected_value = torch.tensor(norm).bfloat16().float().item()
    assert x['q'][0, 0, 0, 0].item() == expected_value
    stage = ref.reference_stages(x)
    for source, name in [('q', 'q_norm'), ('k', 'k_read')]:
        expected = torch.nn.functional.normalize(x[source].float(), dim=-1)
        torch.testing.assert_close(stage[name], expected, atol=0, rtol=0)
        assert stage[name].dtype == torch.float32
    # A nominal clamp input may round to the other side; this is input quantization.
    if norm == 1e-12:
        assert expected_value > norm
    if norm == 1e-13:
        assert stage['q_norm'][0, 0, 0, 0].item() == pytest.approx(.09992007166, abs=1e-9)


def test_zero_budget_and_corrupt_actual_storage():
    x = inputs(beta=0.)
    expected = ref.reference(x)
    actual = {n: v.to(torch.bfloat16 if n == 'o' else torch.float32).clone()
              for n, v in expected.items()}
    assert ref.acceptable(actual, expected, torch.bfloat16)
    for name in ('o', 'final_state'):
        assert ref.floor(expected[name]) == 0
        assert ref.budget(expected[name], torch.bfloat16, name) == 0
        wrong = {n: v.clone() for n, v in actual.items()}
        wrong[name][(0,) * wrong[name].ndim] = 1e-30
        assert wrong[name][(0,) * wrong[name].ndim] != 0
        assert not ref.acceptable(wrong, expected, torch.bfloat16)
    for mode in ('nan', 'wrong_dtype', 'wrong_shape', 'wrong_ATK'):
        wrong = {n: v.clone() for n, v in actual.items()}
        if mode == 'nan': wrong['o'].fill_(float('nan'))
        if mode == 'wrong_dtype': wrong['o'] = wrong['o'].float()
        if mode == 'wrong_shape': wrong['o'] = wrong['o'][:, :-1]
        if mode == 'wrong_ATK': wrong['final_A_state'].mul_(1.25)
        assert not ref.acceptable(wrong, expected, torch.bfloat16)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_public_passes_original_storage_and_returns_device_buffers(monkeypatch, dtype):
    import ascriptor.runtime
    x = inputs(dtype=str(dtype).removeprefix('torch.'))
    selected = public._bf16_pipeline() if dtype == torch.bfloat16 else public._pipeline()
    expected = ref.reference_stages(x)
    names = [entry.name for entry in selected.entries()]
    seen = []
    returned = {}
    class Device:
        def __init__(self, entry, **options): self.index = names.index(entry.name)
        def __call__(self, *args):
            _, sources, outputs = selected.GRAPH[self.index]
            for name, value in zip(sources, args):
                if name in x and name != 'initial_state': assert value is x[name]
                if name == 'initial_state': assert value.dtype == torch.float32 and not value.any()
            buffers = args[len(sources):len(sources)+len(outputs)]
            result = tuple(expected[name].to(buffer.dtype).clone() for name, buffer in zip(outputs, buffers))
            returned.update(zip(outputs, result));seen.append(self.index)
            return result[0] if len(result) == 1 else result
    monkeypatch.setattr(ascriptor.runtime, 'OpExec', Device)
    actual = public.chunk_pgdn(**{n: v for n, v in x.items() if n != 'initial_state'},
                               launcher='board', output_final_state=True)
    assert seen == list(range(6))
    for name, value in zip(ref.OUTPUTS, actual): assert value is returned[name]
    assert actual[0].dtype == dtype and actual[1].dtype == actual[2].dtype == torch.float32


@pytest.mark.parametrize('name', ['q', 'k', 'v', 'g_atk', 'g', 'beta_atk', 'beta'])
def test_bf16_noncontiguous_rejected_before_dispatch(monkeypatch, name):
    x = inputs();shape = x[name].shape
    x[name] = torch.empty((*shape[:-1], shape[-1]*2), dtype=x[name].dtype)[..., ::2]
    monkeypatch.setattr(public, '_bf16_pipeline', lambda: pytest.fail('invalid input dispatched'))
    with pytest.raises(ValueError, match='contiguous'):
        public.chunk_pgdn(**{n: v for n, v in x.items() if n != 'initial_state'}, launcher='board')


@pytest.mark.parametrize('name', ['g_atk', 'g', 'beta_atk', 'beta'])
def test_gate_storage_remains_fp32(monkeypatch, name):
    x = inputs();x[name] = x[name].bfloat16()
    monkeypatch.setattr(public, '_bf16_pipeline', lambda: pytest.fail('invalid input dispatched'))
    with pytest.raises(ValueError, match='requires float32'):
        public.chunk_pgdn(**{n: v for n, v in x.items() if n != 'initial_state'}, launcher='board')


def test_literal_oracle_hash_checked_before_import(tmp_path, monkeypatch):
    path = tmp_path / 'naive.py';path.write_text("raise AssertionError('must not execute')\n")
    monkeypatch.setenv('FLA_PGDN_NAIVE', str(path));oracle.load.cache_clear()
    try:
        with pytest.raises(ValueError, match='FLA pin'): oracle.load()
    finally:
        oracle.load.cache_clear()


def test_public_has_no_dtype_or_layout_conversion():
    tree = ast.parse(Path(public.__file__).read_text())
    forbidden = {'float', 'bfloat16', 'half', 'to', 'contiguous', 'permute', 'transpose',
                 'repeat', 'repeat_interleave', 'clone', 'cat', 'stack'}
    assert not [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in forbidden]


def test_case_grid_preserved_and_clamp_cases_added():
    new = json.loads((ROOT / 'contract.json').read_text())
    old = json.loads((ROOT.with_name('pgdn_chunk_fwd') / 'contract.json').read_text())
    mapped = {c['id']: c for c in new['cases']}
    assert len(mapped) == 66
    for case in old['cases']:
        expected = dict(case, parameters={n: v for n, v in case['parameters'].items() if n != 'bf16_inputs'})
        assert mapped[case['id']] == expected
    assert {mapped[f'clamp_{i}_bd1']['parameters']['norm'] for i in range(3)} == {1e-13, 1e-12, 2e-12}
