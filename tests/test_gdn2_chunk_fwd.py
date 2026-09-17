"""Chunk math, public rejection gates and model cache continuity on CPU."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from ascend_fla.ops.gdn2_chunk_fwd import chunk_gdn2
from ascend_fla.reference.gdn2 import gdn2_recurrent_reference


def _chunk_reference():
    path = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/gdn2_chunk_fwd/ref/chunk.py'
    spec = importlib.util.spec_from_file_location('gdn2_chunk_reference_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gdn2_chunk_reference


gdn2_chunk_reference = _chunk_reference()


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(time, heads=1, gate_scale=3.):
    rng = torch.Generator().manual_seed(61 + time)
    shape = (1, time, heads, 128)
    q, k, v = (torch.randn(shape, generator=rng) for _ in range(3))
    g = -torch.rand(shape, generator=rng) * gate_scale
    b = torch.rand(shape, generator=rng) * 2
    w = torch.rand(shape, generator=rng)
    state = torch.randn(1, heads, 128, 128, generator=rng) * 0.1
    return q, k, v, g, b, w, state


def assert_close(got, expected, budget=1e-4):
    assert torch.isfinite(got).all()
    diff = got.float() - expected.float()
    rel = float(diff.norm() / expected.float().norm().clamp_min(1e-30))
    assert rel <= budget, (rel, float(diff.abs().max()))


@pytest.mark.parametrize('time', [1,16,63,64,65,127,128,192,320,1024,4096])
@pytest.mark.parametrize('gate_scale', [0.,60.])
def test_chunk_vs_independent_recurrence(time, gate_scale):
    args = inputs(time, heads=16, gate_scale=gate_scale)
    state_before = args[-1].clone()
    got = gdn2_chunk_reference(*args)
    expected = gdn2_recurrent_reference(*args, scale=128**-0.5,
                                       use_qk_l2norm=True, qk_norm_eps=1e-6)
    for a, b in zip(got, expected):
        assert_close(a, b)
    assert torch.equal(args[-1], state_before)
    assert got[1].data_ptr() != args[-1].data_ptr()


def test_strong_decay_split_prefill_and_decode():
    args = inputs(193, gate_scale=60.)
    whole = gdn2_chunk_reference(*args)
    prefix, state = gdn2_chunk_reference(*(x[:, :65].contiguous() for x in args[:6]), args[-1])
    suffix, state = gdn2_chunk_reference(*(x[:, 65:192].contiguous() for x in args[:6]), state)
    step, state = gdn2_recurrent_reference(*(x[:, 192:].contiguous() for x in args[:6]), state,
                                          scale=128**-0.5, use_qk_l2norm=True, qk_norm_eps=1e-6)
    assert_close(torch.cat((prefix, suffix, step), dim=1), whole[0])
    assert_close(state, whole[1])


@pytest.mark.parametrize('erase,write', [(0.,0.),(0.,1.),(2.,0.),(2.,1.)])
def test_gate_endpoints_and_zero_state(erase, write):
    args = list(inputs(65))
    args[4].fill_(erase)
    args[5].fill_(write)
    args[6].zero_()
    expected = gdn2_recurrent_reference(*args, scale=128**-0.5, use_qk_l2norm=True, qk_norm_eps=1e-6)
    for a, b in zip(gdn2_chunk_reference(*args), expected):
        assert_close(a, b)


def test_fla_oracle_when_explicitly_supplied():
    path = os.environ.get('FLA_GDN2_NAIVE')
    if not path:
        pytest.skip('set FLA_GDN2_NAIVE to the pinned FLA gdn2/naive.py')
    spec = importlib.util.spec_from_file_location('fla_gdn2_naive_test', Path(path))
    naive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(naive)
    args = inputs(65, heads=16, gate_scale=60.)
    q, k = (x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6) for x in args[:2])
    expected = naive.naive_recurrent_gdn2(q, k, *args[2:6], initial_state=args[6], output_final_state=True)
    for a, b in zip(gdn2_chunk_reference(*args), expected):
        assert_close(a, b)


@pytest.mark.parametrize('defect,match', [
    ('grad','inference'), ('positive_gate','g<=0'), ('erase','g<=0'),
    ('nan','finite'), ('shape','shape'), ('dtype','float32'),
    ('noncontiguous','contiguous'), ('state','initial_state'),
])
def test_public_rejects_before_compilation(defect, match):
    args = list(inputs(2))
    if defect == 'grad': args[0].requires_grad_()
    if defect == 'positive_gate': args[3].fill_(1.)
    if defect == 'erase': args[4].fill_(3.)
    if defect == 'nan': args[1].fill_(float('nan'))
    if defect == 'shape': args[2] = args[2][..., :64]
    if defect == 'dtype': args[3] = args[3].to(torch.bfloat16)
    if defect == 'noncontiguous': args[0] = args[0].expand(2,-1,-1,-1)[::2]
    if defect == 'noncontiguous':
        storage = torch.empty(1,2,1,256)
        storage[..., ::2] = args[0]
        args[0] = storage[..., ::2]
    if defect == 'state': args[6] = args[6].to(torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError), match=match):
        chunk_gdn2(*args[:6], initial_state=args[6], launcher='aclnn')


@pytest.mark.parametrize('kwargs,match', [
    ({'launcher':'sim'}, 'launcher'), ({'block_dim':0}, 'block_dim'),
    ({'scale':1.}, 'scale'), ({'use_qk_l2norm':False}, 'normalization'),
    ({'qk_norm_eps':0.}, 'normalization'), ({'device':'a2'}, 'device'),
])
def test_public_options(kwargs, match):
    args = inputs(2)
    options = dict(launcher='aclnn') | kwargs
    with pytest.raises(ValueError, match=match):
        chunk_gdn2(*args[:6], initial_state=args[6], **options)






@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_cpu_public_launch_graph_with_real_simulator(monkeypatch, tmp_path, dtype):
    """Exercise the CPU adapter and workspace retirement without claiming CCE run."""
    runtime = pytest.importorskip('ascriptor.runtime')
    original = runtime.OpExec
    launches = []
    def simulator(entry, **options):
        assert options['launcher'] == 'aclnn'
        launches.append(entry.name)
        options['launcher'] = 'sim'
        return original(entry, **options)
    monkeypatch.setattr(runtime, 'OpExec', simulator)
    args = list(inputs(2, gate_scale=60.))
    args[:3] = [x.to(dtype) for x in args[:3]]
    before = args[-1].clone()
    # The simulator starts worker threads; no_grad keeps writable ordinary
    # tensors while disabling autograd (InferenceMode is thread-local).
    with torch.no_grad():
        got = chunk_gdn2(*args[:6], initial_state=args[-1], output_final_state=True,
                         launcher='aclnn', block_dim=1, out_dir=tmp_path)
        expected = gdn2_recurrent_reference(*args, scale=128**-0.5,
                                           use_qk_l2norm=True, qk_norm_eps=1e-6)
    assert len(launches) == 5 and len(set(launches)) == 5
    assert got[0].dtype == dtype and got[1].dtype == torch.float32
    assert torch.equal(args[-1], before)
    for a,b in zip(got, expected):
        assert_close(a,b,1e-4 if dtype == torch.float32 else 1e-2)


