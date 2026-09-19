"""PKDA semantics, independent states, public gates and real model outputs."""
import os
import pytest
import torch
from ascend_fla.ops import pkda_chunk_fwd as op

ref = op._module('ref.reference')
stages = op._module('ref.stages')
oracles = op._module('ref.oracles')


def inputs(T=2,H=1,B=1,**parameters):
    return ref.make_inputs(dict(seed=1701,parameters=dict(B=B,T=T,H=H,**parameters)))


def assert_close(got, expected):
    ref.validate_reference({'q':torch.empty(*expected['o'].shape)},got)
    for n, m in oracles.metrics(got,expected).items():
        assert m['relative_l2'] <= 1e-4, (n,m)
        torch.testing.assert_close(got[n],expected[n],atol=1e-4,rtol=1e-4)


@pytest.mark.parametrize('T,H',[(1,1),(16,1),(63,8),(64,8),(65,8),(128,8),(192,8),(1024,8)])
def test_chunk_against_pinned_fla(T,H):
    if 'FLA_PKDA_NAIVE' not in os.environ:
        pytest.skip('set FLA_PKDA_NAIVE to the pinned FLA source')
    data=inputs(T,H)
    assert_close(ref.reference(data),oracles.oracle(data))


@pytest.mark.parametrize('scale_S,scale_A',[(0.,0.),(.1,0.),(0.,.03),(.1,.03)])
def test_independent_state_carry_and_no_mutation(scale_S,scale_A):
    data=inputs(65,3,state_scale=scale_S,A_scale=scale_A)
    originals={n:x.clone() for n,x in data.items()}
    whole=ref.reference(data)
    def segment(start,end):
        return {n:(x[:,start:end].contiguous() if n in ('q','k','v','g','g_atk','beta_atk','beta') else x)
                for n,x in data.items()}
    left=ref.reference(segment(0,17))
    right_inputs=segment(17,65)
    right_inputs.update(initial_state=left['final_state'],initial_A_state=left['final_A_state'])
    right=ref.reference(right_inputs)
    split=dict(o=torch.cat([left['o'],right['o']],dim=1),final_state=right['final_state'],final_A_state=right['final_A_state'])
    assert_close(split,whole)
    for n in data: assert torch.equal(data[n],originals[n]),n


@pytest.mark.parametrize('norm',[1e-2,1e-4,1e-6,0.])
def test_no_hidden_normalization(norm):
    if 'FLA_PKDA_NAIVE' not in os.environ: pytest.skip('pinned FLA required')
    data=inputs(2,row_norm=norm)
    assert_close(ref.reference(data),oracles.oracle(data))
    if norm==0:
        assert torch.count_nonzero(ref.reference(data)['o'])==0


@pytest.mark.parametrize('name,shape',[
 ('initial_state',(1,1,128)),('initial_A_state',(1,1,128,128)),
 ('g_atk',(1,2,1,128)),('beta_atk',(1,2)),('beta',(1,2)),
 ('v',(1,2,1,120)),('log_atk_scale',(1,1))])
def test_public_shape_errors_before_launch(name,shape,monkeypatch):
    data=inputs();data[name]=torch.zeros(shape)
    monkeypatch.setattr(op,'_compiled',lambda *a:pytest.fail('compiled invalid ABI'))
    with pytest.raises(ValueError,match=name): op.chunk_precond_kda(**data,launcher='sim')


@pytest.mark.parametrize('change',[
 'bf16','noncontiguous','positive_g','positive_g_atk','negative_A','beta','beta_atk',
 'key_norm','span','nonfinite','head_dim','empty','training'])
def test_public_domain_errors(change):
    data=inputs()
    if change=='bf16': data['q']=data['q'].bfloat16()
    elif change=='noncontiguous': data['q']=data['q'].expand(2,-1,-1,-1)
    elif change=='positive_g': data['g'].fill_(.1)
    elif change=='positive_g_atk': data['g_atk'].fill_(.1)
    elif change=='negative_A': data['initial_A_state'].fill_(-.1)
    elif change in ('beta','beta_atk'):data[change].fill_(1.01)
    elif change=='key_norm':data['k'].mul_(2)
    elif change=='span':data['g'].fill_(-78)
    elif change=='nonfinite':data['v'][0,0,0,0]=float('nan')
    elif change=='head_dim':data['q']=torch.zeros(1,2,1,120)
    elif change=='empty':data['q']=torch.zeros(1,0,1,128)
    elif change=='training':data['q'].requires_grad_()
    with pytest.raises((ValueError,RuntimeError)):
        op.chunk_precond_kda(**data,launcher='sim')


@pytest.mark.parametrize('extra',[
 dict(use_gate_in_kernel=True),dict(safe_gate=True),dict(lower_bound=-5),
 dict(cu_seqlens=torch.tensor([0,2])),dict(transpose_state_layout=True),
 dict(return_intermediate_states=True),dict(x=2.),dict(eps=1e-5),
 dict(block_dim=8),dict(scale=float('inf')),dict(A_log=torch.zeros(1))])
def test_unsupported_options_explicit(extra):
    with pytest.raises((ValueError,NotImplementedError)):
        op.chunk_precond_kda(**inputs(),launcher='sim',**extra)


def test_comparator_rejects_missing_zero_and_nonfinite():
    import importlib.util
    from pathlib import Path
    path=Path(ref.__file__).parents[1]/'_unit_runner.py'
    spec=importlib.util.spec_from_file_location('_pkda_comparator_test',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    import json
    contract=json.loads((path.parent/'contract.json').read_text())
    expected=ref.reference(inputs())
    for wrong in ({'o':expected['o']},{n:torch.zeros_like(x) for n,x in expected.items()},
                  {n:torch.full_like(x,float('nan')) for n,x in expected.items()}):
        with pytest.raises(runner.ContractError):runner.compare_outputs(wrong,expected,contract)


def test_oracle_rejects_unpinned_source(tmp_path,monkeypatch):
    wrong=tmp_path/'naive.py'
    wrong.write_text('raise AssertionError("unverified source was executed")\n')
    monkeypatch.setenv('FLA_PKDA_NAIVE',str(wrong))
    with pytest.raises(ValueError,match='pinned'):
        oracles.fla_naive()


def test_public_returns_actual_sim_outputs_and_optional_states(tmp_path):
    pytest.importorskip('ascriptor')
    if 'FLA_PKDA_NAIVE' not in os.environ:pytest.skip('pinned FLA required')
    data=inputs(1)
    original={n:x.clone() for n,x in data.items()}
    result=op.chunk_precond_kda(**data,output_final_state=True,launcher='sim',out_dir=tmp_path/'with-state')
    assert_close(dict(zip(ref.OUTPUTS,result)),oracles.oracle(data))
    for n in data:assert torch.equal(data[n],original[n])
    assert result[1].data_ptr()!=data['initial_state'].data_ptr()
    assert result[2].data_ptr()!=data['initial_A_state'].data_ptr()
    result=op.chunk_precond_kda(**data,launcher='sim',out_dir=tmp_path/'without-state')
    assert result[1:] == (None,None)
