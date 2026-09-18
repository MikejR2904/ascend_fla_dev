"""Independent GDN math, public rejection gates and launch graph contracts."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

from ascend_fla.ops.gdn_chunk_fwd import chunk_gdn

ROOT = Path(__file__).resolve().parents[1] / 'kernels/projects/a5/gdn_chunk_fwd'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load('gdn_test_reference', ROOT / 'ref/reference.py')
stages = load('gdn_test_stages', ROOT / 'ref/stages.py')


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def data(time=64, heads=3, gate=1., normalized=False):
    return ref.make_inputs({'seed':73+time,'parameters':{'B':1,'T':time,'H':heads,'gate_scale':gate,'normalized_keys':normalized}})


def compare(got, expected, budget=1e-5):
    assert torch.isfinite(got).all()
    delta=got.float()-expected.float()
    norm=expected.float().norm()
    assert float(delta.norm()/norm.clamp_min(1e-30)) <= budget
    torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize('time,heads,gate',[(64,1,0.),(64,3,.03),(128,3,1.),(192,3,30.),(320,3,.03),(1024,16,.03),(4096,16,1.)])
def test_stage_equations_match_independent_block_solve(time, heads, gate):
    inp=data(time,heads,gate)
    before={n:x.clone() for n,x in inp.items()}
    expected=ref.reference(inp)
    got=stages.reference_stages(inp)
    for name in expected:
        compare(got[name],expected[name])
    for name in inp:
        assert torch.equal(inp[name],before[name])


@pytest.mark.parametrize('gate',[0.,.03,30.])
def test_fla_oracle(gate):
    path=os.environ.get('FLA_GDN_NAIVE')
    if not path:
        pytest.skip('set FLA_GDN_NAIVE to the pinned gated_delta_rule/naive.py')
    fla=load('fla_gdn_test_naive',Path(path))
    inp=data(192,3,gate,normalized=True)
    got=ref.reference(inp)
    o,s=fla.naive_recurrent_gated_delta_rule(*(inp[n] for n in ('q','k','v','beta','g')),output_final_state=True)
    compare(got['o'],o)
    compare(got['final_state'],s)


@pytest.mark.parametrize('beta',[0.,1.])
def test_gate_endpoints(beta):
    inp=data(128)
    inp['beta'].fill_(beta)
    got=ref.reference(inp)
    expected=stages.reference_stages(inp)
    for n in got:
        compare(got[n],expected[n])
        if beta==0:
            assert not got[n].any()


def call(inp, **kwargs):
    return chunk_gdn(*(inp[n] for n in ('q','k','v','g','beta')),launcher='board',**kwargs)


@pytest.mark.parametrize('option,value,match',[
    ('initial_state',torch.ones(1,3,128,128),'initial_state'),
    ('head_first',True,'head_first'),('scale',1.,'scale'),
    ('block_dim',8,'block_dim'),('block_dim',True,'block_dim'),('device','a2','a5')])
def test_options_reject_before_launch(option,value,match):
    with pytest.raises(ValueError,match=match):
        call(data(),**{option:value})


@pytest.mark.parametrize('mutation,match',[
    ('gqa','no GQA'),('tail','multiple of 64'),('strided','contiguous'),
    ('dtype','matching'),('positive_gate','g<=0'),('beta','beta'),('nan','finite')])
def test_invalid_inputs(mutation,match):
    inp=data()
    if mutation=='gqa': inp['v']=torch.zeros(1,64,6,128)
    if mutation=='tail':
        for n in ('q','k','v','g','beta'): inp[n]=inp[n][:,:63].contiguous()
    if mutation=='strided': inp['q']=torch.zeros(1,64,3,256)[...,::2]
    if mutation=='dtype': inp['k']=inp['k'].half()
    if mutation=='positive_gate': inp['g'].fill_(1.)
    if mutation=='beta': inp['beta'].fill_(2.)
    if mutation=='nan': inp['v'].fill_(float('nan'))
    with pytest.raises(ValueError,match=match): call(inp)


def test_no_backward():
    inp=data(); inp['q'].requires_grad_(True)
    with pytest.raises(RuntimeError,match='inference-only'): call(inp)
