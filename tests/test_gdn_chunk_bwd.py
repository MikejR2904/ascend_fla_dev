"""GDA-03 public domain and mathematical adjoint regression tests."""
import importlib.util
from pathlib import Path
import sys
import pytest
import torch

from ascend_fla.ops import gdn_chunk_bwd as op

ROOT=Path(__file__).resolve().parents[1]/'kernels/projects/a5/gdn_chunk_bwd'


def ref_package():
    name='_gda03_test_ref'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'ref/__init__.py',submodule_search_locations=[str(ROOT/'ref')])
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return importlib.import_module(name+'.calibrate'),importlib.import_module(name+'.reference')


def valid(dtype=torch.float32):
    return dict(q=torch.zeros(1,64,2,128,dtype=dtype),k=torch.zeros(1,64,2,128,dtype=dtype),
                v=torch.zeros(1,64,4,128,dtype=dtype),g=torch.zeros(1,64,4),beta=torch.zeros(1,64,4),
                do=torch.zeros(1,64,4,128,dtype=dtype),dht=torch.zeros(1,4,128,128))


@pytest.mark.parametrize('mode',['do','dht','both'])
def test_valid_modes(mode):
    values=valid()
    if mode=='do':values['dht']=None
    if mode=='dht':values['do']=None
    op._validate(**values,launcher='board')


@pytest.mark.parametrize('launcher', ['inprocess', 'aclnn', 'board'])
@pytest.mark.parametrize('name', ['q', 'k', 'v', 'do', 'all'])
def test_bf16_rejected_before_preparation(name, launcher, monkeypatch):
    values = valid()
    for key in ('q', 'k', 'v', 'do') if name == 'all' else (name,):
        values[key] = values[key].bfloat16()
    monkeypatch.setattr(op, '_pipeline', lambda: pytest.fail('BF16 must reject before dispatch'))
    from torch.utils._python_dispatch import TorchDispatchMode
    class NoTensorOperation(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            pytest.fail(f'BF16 rejection must precede tensor preparation: {func}')
    with NoTensorOperation(), pytest.raises(ValueError, match='BF-02'):
        op.chunk_gdn_bwd(**values, launcher=launcher)


@pytest.mark.parametrize('opts',[
    dict(initial_state=torch.zeros(1,4,128,128)),dict(head_first=True),dict(transpose_state_layout=True),
    dict(cu_seqlens=torch.tensor([0,64])),dict(cp_context=object()),dict(use_qk_l2norm_in_kernel=True),
    dict(scale=1.),dict(scale=float('nan')),dict(block_dim=3),dict(block_dim=True),dict(device='a2'),
    dict(launcher='sim'),
])
def test_options_reject_before_dispatch(opts,monkeypatch):
    monkeypatch.setattr(op,'_pipeline',lambda:pytest.fail('must reject before kernel import'))
    with pytest.raises(ValueError):op.chunk_gdn_bwd(**valid(),**dict({'launcher':'board'},**opts))


@pytest.mark.parametrize('name,value',[
    ('v',torch.zeros(1,64,4,64)),('v',torch.zeros(1,64,0,128)),
    ('q',torch.zeros(1,0,2,128)),('q',torch.zeros(1,64,0,128)),
    ('q',torch.zeros(1,63,2,128)),('q',torch.zeros(1,4160,2,128)),('q',torch.zeros(0,64,2,128)),
    ('q',torch.zeros(1,64,2,64)),('q',torch.zeros(1,64,2,128,dtype=torch.float16)),
    ('k',torch.zeros(1,64,2,128,dtype=torch.bfloat16)),('v',torch.zeros(1,64,3,128)),
    ('g',torch.ones(1,64,4)),('beta',torch.full((1,64,4),-0.1)),('beta',torch.full((1,64,4),1.1)),
    ('g',torch.zeros(1,64,4,dtype=torch.bfloat16)),('do',torch.zeros(1,64,2,128)),
    ('do',torch.zeros(1,64,4,128,dtype=torch.bfloat16)),('dht',torch.zeros(1,4,128,128,dtype=torch.bfloat16)),
    ('dht',torch.zeros(1,4,128,64)),('beta',torch.full((1,64,4),float('nan'))),
    ('dht',torch.full((1,4,128,128),float('inf'))),('do',torch.zeros(1,128,4,64).transpose(1,3)),
])
def test_tensor_domain_rejected(name,value):
    values=valid();values[name]=value
    with pytest.raises(ValueError):op._validate(**values,launcher='board')


def test_absent_both_and_wrong_device():
    values=valid();values.update(do=None,dht=None)
    with pytest.raises(ValueError,match='at least one'):op._validate(**values,launcher='board')
    with pytest.raises(ValueError,match='one npu'):op._validate(**valid())


@pytest.mark.parametrize('ratio',[1,2,4,8])
@pytest.mark.parametrize('mode',['do','dht','both'])
def test_independent_analytical_vs_literal_pin(ratio,mode):
    import os
    if 'FLA_GDN_NAIVE' not in os.environ:pytest.skip('set explicit pinned FLA_GDN_NAIVE')
    cal,ref=ref_package();xs=cal.inputs(1,64,1,ratio)
    q,k,v,g,beta,do,dh=xs
    if mode=='do':dh=None
    if mode=='dht':do=None
    expected=cal.oracle.autograd(q,k,v,g,beta,do,dh)
    actual=ref.analytical(q,k,v,g,beta,do,dh)
    assert ref.acceptable(ref.metrics(actual,expected),1e-5)
    if mode=='dht':assert torch.count_nonzero(actual['dq'])==0


def test_double_gradcheck_and_negative_controls():
    import os
    if 'FLA_GDN_NAIVE' not in os.environ:pytest.skip('set explicit pinned FLA_GDN_NAIVE')
    cal,ref=ref_package()
    assert all(r['A_precision_lift'] and r['B_analytical'] for r in cal.gradcheck())
    xs=cal.inputs(1,64,1,2)
    a=cal.oracle.autograd(*xs);b=ref.analytical(*xs)
    for factor in (0.,-1.,1.25):
        assert all(m['relative_l2']>1e-4 for m in ref.metrics({n:x*factor for n,x in b.items()},a).values())


def test_grouped_sum_and_scale_in_zero_gate_case():
    cal,ref=ref_package();q,k,v,g,beta,do,dh=cal.inputs(1,64,2,8,'zero_gate')
    grouped=ref.analytical(q,k,v,g,beta,do,dh)
    expanded=ref.analytical(q.repeat_interleave(4,2),k.repeat_interleave(4,2),v,g,beta,do,dh)
    for n in ('dq','dk'):
        torch.testing.assert_close(grouped[n],expanded[n].reshape(1,64,2,4,128).sum(3),rtol=0,atol=0)
    unscaled=ref.analytical(q,k,v,g,beta,do,None,scale=1.)
    scaled=ref.analytical(q,k,v,g,beta,do,None)
    torch.testing.assert_close(scaled['dq'],unscaled['dq']*op.SCALE,rtol=1e-6,atol=1e-9)
