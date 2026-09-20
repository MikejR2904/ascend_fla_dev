"""Native BF16 backward contract, lowering and independent-reference regression."""
import importlib.util
import os
from pathlib import Path
import sys
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from ascend_fla.ops import gdn_chunk_bwd as public

ROOT=Path(__file__).resolve().parents[1]/'kernels/projects/a5/gdn_chunk_bwd_bf16'


def references():
    name='_bf02_test_ref'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'ref/__init__.py',submodule_search_locations=[str(ROOT/'ref')])
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return importlib.import_module(name+'.bf16'),importlib.import_module(name+'.oracle')


def generated(ratio=2,mode='both'):
    ref,_=references()
    return ref.make_inputs(dict(seed=17391,parameters=dict(B=1,T=64,H=1,HV=ratio,mode=mode)))


@pytest.mark.parametrize('mode',['both','do','dht'])
@pytest.mark.parametrize('ratio',[1,2,4,8])
def test_bf16_rounded_oracles_and_fixed_budget(ratio,mode):
    if 'FLA_GDN_NAIVE' not in os.environ:pytest.skip('explicit pinned FLA oracle required')
    ref,oracle=references();inputs=generated(ratio,mode)
    b=ref.reference(inputs);wide=ref.fp32_inputs(inputs)
    a=oracle.autograd(*(wide[n] for n in ref.INPUTS))
    actual={n:x.bfloat16() if n in ref.NARROW_OUTPUTS else x for n,x in b.items()}
    numbers=ref.comparison(actual,a,torch.bfloat16)
    assert ref.acceptable(numbers),numbers
    assert all(m['budget']==min(.01,3*m['floor']) for m in numbers.values())
    if mode=='dht':assert numbers['dq']['floor']==0 and numbers['dq']['relative_l2']==0


def test_zero_floor_cannot_admit_a_nonzero_output():
    ref,_=references()
    for target in (torch.zeros(1,2,4).transpose(1,2),torch.ones(1,2,4).transpose(1,2)):
        assert not target.is_contiguous()
        for name in ref.NAMES:
            expected={name:target}
            actual={name:target.to(torch.bfloat16 if name in ref.NARROW_OUTPUTS else torch.float32).clone()}
            assert ref.acceptable(ref.comparison(actual,expected,torch.bfloat16))
            actual[name][0,0,0]=2
            assert torch.count_nonzero(actual[name].float()-target)>0
            numbers=ref.comparison(actual,expected,torch.bfloat16)
            assert numbers[name]['budget']==0
            assert not ref.acceptable(numbers)


@pytest.mark.parametrize('factor',[0.,-1.,1.25])
def test_nonzero_gradient_negative_controls(factor):
    ref,_=references();expected=ref.reference(generated())
    bad={n:(x*factor).to(torch.bfloat16 if n in ref.NARROW_OUTPUTS else torch.float32) for n,x in expected.items()}
    assert all(not m['passed'] for m in ref.comparison(bad,expected,torch.bfloat16).values())


@pytest.mark.parametrize('mode',['both','do','dht'])
def test_real_pipeline_allocates_without_operand_preparation(mode):
    values=generated(mode=mode)
    if mode=='do':values['dht']=None
    if mode=='dht':values['do']=None
    class AllocationsOnly(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            assert str(func) in ('aten.empty.memory_format','aten.empty_strided.default'),str(func)
            return func(*args,**(kwargs or {}))
    seen=[]
    def launch(entry,sources,outputs,scalars):
        if entry.name.endswith('reverse'):
            assert scalars['has_do']==int(mode!='dht')
            assert scalars['has_dht']==int(mode!='do')
            assert sources['dout'].dtype==torch.bfloat16
            assert sources['dht'].dtype==torch.float32
            if mode!='dht':assert sources['dout'] is values['do']
            if mode!='do':assert sources['dht'] is values['dht']
        seen.append(entry.name)
        return outputs
    pipeline=public._bf16_pipeline()
    with AllocationsOnly():got=pipeline.run(values,launch)
    assert len(seen)==3
    assert [got[n].dtype for n in ('dq','dk','dv','dg','dbeta')]==[torch.bfloat16]*3+[torch.float32]*2


def test_all_typed_stages_lower_with_balanced_events():
    # Regression for accidentally narrowing partial dq/dk UB before FP32 DMA.
    from ascriptor.passes import PIPELINE,PassManager
    from ascriptor.passes.autosync import check_balance
    for entry in public._bf16_pipeline().entries():
        assert not check_balance(PassManager(PIPELINE).run(entry.ir()))


def test_prepare_compiles_both_paths_before_first_resolution(monkeypatch):
    calls=[]
    monkeypatch.setattr(public,'_compiled',lambda bd:calls.append(('fp32',bd)))
    monkeypatch.setattr(public,'_bf16_compiled',lambda bd:calls.append(('bf16',bd)))
    public.prepare(block_dim=1)
    assert calls==[('fp32',1),('bf16',1)]
