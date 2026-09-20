"""BF16 calibration, native graph ownership, public gates and fixed budgets."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from ascend_fla.ops import gdn_chunk_fwd as api

ROOT=Path(__file__).resolve().parents[1]/'kernels/projects/a5/gdn_chunk_fwd_bf16'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


ref=load('bf01_test_ref',ROOT/'ref/reference.py')
oracle=load('bf01_test_oracle',ROOT/'ref/oracle.py')
stage_ref=load('bf01_test_stage_ref',ROOT/'ref/stages.py')


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(ratio=2,chunks=1,dtype='bfloat16'):
    return ref.make_inputs(dict(seed=7314,B=1,T=64*chunks,H=2,HV=2*ratio,dtype=dtype))


@pytest.mark.parametrize('dtype',['bfloat16','float32'])
@pytest.mark.parametrize('ratio',[1,2,4,8])
@pytest.mark.parametrize('chunks',[1,3])
def test_independent_block_and_stage_references(dtype,ratio,chunks):
    x=inputs(ratio,chunks,dtype);before={n:v.clone() for n,v in x.items()}
    b=ref.reference(x);stages=stage_ref.reference_stages(x)
    for n in b:assert ref.metric(stages[n],b[n])['relative_l2']<=1e-4
    if os.environ.get('FLA_GDN_NAIVE'):
        a=oracle.reference(x)
        for n in b:assert ref.metric(b[n],a[n])['relative_l2']<=1e-4
    for n in x:assert torch.equal(x[n],before[n])
    ref.validate_reference(x,b)


@pytest.mark.parametrize('dtype',['bfloat16','float32'])
@pytest.mark.parametrize('ratio',[1,2,3,8])
def test_allocation_graph_preserves_original_typed_inputs(monkeypatch,dtype,ratio):
    pipeline=api._native_pipeline();x=inputs(ratio,dtype=dtype)
    fake=tuple(SimpleNamespace(name=name) for name,_,_ in pipeline.GRAPH)
    monkeypatch.setattr(pipeline,'entries',lambda selected:fake)
    seen=[];ops=[]
    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            ops.append(str(func));return func(*args,**(kwargs or {}))
    def launch(entry,sources,outputs,scalars):
        if not seen:
            for n in ('q','k','v'):assert sources[n] is x[n]
            assert scalars['HK']==2 and scalars['H']==2*ratio
        else:assert 'HK' not in scalars
        seen.append(entry.name)
        return outputs
    with Audit():got=pipeline.run(x,launch,retain_stages=False)
    assert len(seen)==5 and set(ops)=={'aten.empty.memory_format'}
    assert got['o'].shape==x['v'].shape and got['o'].dtype==x['v'].dtype
    assert got['final_state'].shape==(1,2*ratio,128,128)
    assert got['final_state'].dtype==torch.float32
    assert all(got['o'].data_ptr()!=value.data_ptr() for value in x.values())


def test_zero_floor_requires_exact_zero_and_negative_controls():
    x=inputs();x['beta'].zero_();expected=ref.reference(x)
    assert all(ref.floor(v)==0. and ref.budget(v,torch.bfloat16)==0. for v in expected.values())
    assert ref.acceptable(expected,expected,torch.bfloat16)
    bad={n:v.clone() for n,v in expected.items()};bad['o'].flatten()[0]=1e-30
    assert not ref.acceptable(bad,expected,torch.bfloat16)
    x=inputs();expected=ref.reference(x)
    for change in (lambda v:torch.zeros_like(v),lambda v:-v,lambda v:v*1.25):
        assert not ref.acceptable({n:change(v) for n,v in expected.items()},expected,torch.bfloat16)


def test_literal_oracle_hash_gate(tmp_path,monkeypatch):
    fake=tmp_path/'naive.py';fake.write_text("raise AssertionError('must not execute')\n")
    monkeypatch.setenv('FLA_GDN_NAIVE',str(fake));oracle.load.cache_clear()
    with pytest.raises(ValueError,match='FLA pin'):oracle.load()
    oracle.load.cache_clear()


@pytest.mark.parametrize('name',['q','k','v','g','beta'])
def test_bf16_public_rejects_noncontiguous_without_dispatch(monkeypatch,name):
    x=inputs();shape=x[name].shape
    x[name]=torch.empty((*shape[:-1],shape[-1]*2),dtype=x[name].dtype)[...,::2]
    monkeypatch.setattr(api,'_native_pipeline',lambda:pytest.fail('invalid input dispatched'))
    with pytest.raises(ValueError,match='contiguous'):
        api.chunk_gdn(**x,launcher='board')


@pytest.mark.parametrize('name',['g','beta'])
def test_gate_storage_requires_fp32(monkeypatch,name):
    x=inputs();x[name]=x[name].bfloat16()
    monkeypatch.setattr(api,'_native_pipeline',lambda:pytest.fail('invalid input dispatched'))
    with pytest.raises(ValueError,match='requires float32'):
        api.chunk_gdn(**x,launcher='board')


@pytest.mark.parametrize('option,value,match',[
    ('initial_state',torch.zeros(1,4,128,128),'initial_state'),
    ('head_first',True,'head_first'),('scale',1.,'scale'),
    ('block_dim',True,'block_dim'),('block_dim',3,'block_dim')])
def test_bf16_option_gates(monkeypatch,option,value,match):
    monkeypatch.setattr(api,'_native_pipeline',lambda:pytest.fail('invalid input dispatched'))
    with pytest.raises(ValueError,match=match):
        api.chunk_gdn(**inputs(),launcher='board',**{option:value})
