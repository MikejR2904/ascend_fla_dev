"""BF16 contracts and CPU precision checks; native acceptance is separate."""
import importlib
import math
import pytest
import torch
from ascend_fla.ops import pkda_chunk_fwd as api


def module(suffix):
    return importlib.import_module(api._bf16_module().__package__+'.'+suffix)


def inputs():
    return module('ref.reference').make_inputs(dict(id='test',seed=4111,parameters=dict(B=1,T=65,H=1)))


@pytest.mark.parametrize('name',('q','k','v'))
def test_actual_bf16_input_dtype(name):
    data=inputs();assert data[name].dtype==torch.bfloat16
    data[name]=data[name].float()
    with pytest.raises(ValueError,match=name):module('kernels.pipeline').validate_metadata(data)


@pytest.mark.parametrize('name',('g','g_atk','beta','beta_atk','initial_state','initial_A_state','log_atk_scale'))
def test_fp32_side_inputs(name):
    data=inputs();data[name]=data[name].bfloat16()
    with pytest.raises(ValueError,match=name):module('kernels.pipeline').validate_metadata(data)


@pytest.mark.parametrize('scale',(0.,-1.,math.inf,math.nan,1e40))
def test_scale_domain(scale):
    data=inputs();data['scale']=scale
    with pytest.raises(ValueError,match='scale'):module('kernels.pipeline').validate_metadata(data)


def test_independent_optional_states_metadata():
    for s in (False,True):
        for a in (False,True):
            data=inputs()
            if not s:data.pop('initial_state')
            if not a:data.pop('initial_A_state')
            data.pop('log_atk_scale')
            module('kernels.pipeline').validate_metadata(data)


def test_autograd_rejected():
    data=inputs();data['q'].requires_grad_(True)
    with pytest.raises(RuntimeError,match='no backward'):module('kernels.pipeline').validate_metadata(data)


def test_strided_input_rejected():
    data=inputs();data['q']=torch.empty(1,65,1,256,dtype=torch.bfloat16)[...,::2]
    with pytest.raises(ValueError,match='contiguous'):module('kernels.pipeline').validate_metadata(data)


@pytest.mark.parametrize('code',range(1,9))
def test_device_status_control_readback(code):
    pipeline=module('kernels.pipeline');status=torch.zeros(1,2,64);status[0,1,37]=code
    with pytest.raises(ValueError,match='PKDA|must|key|span|initial'):pipeline.check_status(status)


def test_status_unwritten_detected():
    with pytest.raises(RuntimeError,match='valid status'):module('kernels.pipeline').check_status(torch.full((1,1,64),float('nan')))


def test_status_readback_issues_no_aten():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Audit(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):raise AssertionError(str(func))
    status=torch.zeros(2,3,64)
    with Audit():module('kernels.pipeline').check_status(status)


@pytest.mark.parametrize('span',(0.,105.,155.))
def test_fixed_output_operand_budget(span):
    ref=module('ref.reference');checks=module('ref.checks')
    data=ref.make_inputs(dict(id='floor',seed=704,parameters=dict(B=1,T=64,H=1,gate_span=span)))
    refs=checks.references(data)
    # Independent BF16 operand simulation, no kernel or simulator execution.
    st=refs['stages'];s=st['states'][:,0];delta=st['delta'][:,0]
    rounded=lambda x:x.bfloat16().float()
    output=rounded(st['qn'][:,0]*st['gc'][:,0].exp())@rounded(s)+rounded(st['score'][:,0])@rounded(delta)
    got=dict(refs['B'],o=output.transpose(1,2).bfloat16())
    result=checks.compare(got,refs)
    assert result['passed'],result
    assert refs['budgets']['final_A_state']==1e-4
    assert all(b<=.01 for n,b in refs['budgets'].items())


def test_public_mixed_dtype_rejected_before_launch():
    data=inputs();data['v']=data['v'].float()
    with pytest.raises(ValueError,match='v must'):api.chunk_precond_kda(**data,launcher='sim')
