"""Host ABI/routing checks; numerical silicon acceptance is a separate stage."""
import itertools

import pytest
import torch

from ascend_fla.ops.kda import autograd, chunk, chunk_bwd, fused_recurrent


def raw_inputs(dtype=torch.float32, t=64):
    rng = torch.Generator().manual_seed(707)
    return dict(q=torch.randn(1,t,1,128,generator=rng).to(dtype),
                k=torch.randn(1,t,1,128,generator=rng).to(dtype),
                v=torch.randn(1,t,2,128,generator=rng).bfloat16(),
                g=(torch.randn(1,t,2,128,generator=rng)*.1).to(dtype),
                beta=torch.randn(1,t,2,generator=rng).to(dtype),
                A_log=torch.tensor([-.2,.1],dtype=dtype),
                dt_bias=torch.linspace(-3.,-2.,256).to(dtype))


@pytest.fixture
def native_abi(monkeypatch):
    """Explicit CPU replacement of the native vendor, never a production path."""
    runtime = chunk._prep_runtime()
    events = []

    def check_source(name, value):
        assert value.device.type == 'cpu' and value.is_contiguous()
        assert value.dtype in (torch.bfloat16,torch.float32)

    class Vendors:
        def __getitem__(self, key):
            def launch(inputs, scalars, outputs):
                assert 'backward_ready' in events or 'decode_ready' in events
                events.append(key)
                value = inputs['source'].float()
                if key.startswith('norm_'):
                    value = value.view(-1,128)
                    result = value / (torch.sum(value*value,-1,keepdim=True)+1e-6).sqrt()
                elif key.startswith('gate_'):
                    hv = scalars['HV']
                    assert scalars['Channels']==hv*128
                    assert scalars['N']==scalars['BT']*hv*128
                    value = value.view(scalars['BT'],hv,128)
                    result = -inputs['alog'].float().view(hv,1).exp() * torch.nn.functional.softplus(
                        value+inputs['bias'].float().view(hv,128))
                else:
                    result = torch.sigmoid(value)
                outputs['destination'].copy_(result.reshape(1,-1))
            return launch

    monkeypatch.setattr(runtime,'_check_source',check_source)
    monkeypatch.setattr(runtime,'prepare',lambda *args: Vendors())
    monkeypatch.setattr(chunk,'_compiled_chain',lambda *args: events.append('forward_ready'))
    monkeypatch.setattr(chunk_bwd,'_compiled_chain',lambda *args: events.append('backward_ready'))
    monkeypatch.setattr(fused_recurrent,'_compiled_pair',lambda *args: events.append('decode_ready'))
    return events


@pytest.mark.parametrize('flags',list(itertools.product((False,True),repeat=3)))
@pytest.mark.parametrize('source_dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('output_dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('namespace',['chunk','decode'])
def test_typed_native_abi_and_disabled_identity(native_abi,flags,source_dtype,output_dtype,namespace):
    x=raw_inputs(source_dtype,t=2 if namespace=='decode' else 64)
    options=dict(use_qk_l2norm_in_kernel=flags[0],use_gate_in_kernel=flags[1],
                 use_beta_sigmoid_in_kernel=flags[2],qk_dtype=output_dtype)
    names=('q','k','g','beta')
    old=chunk._prepare_inputs(*(x[n] for n in names),A_log=x['A_log'],dt_bias=x['dt_bias'],**options)
    before={n:t.clone() for n,t in x.items()}
    got=chunk._prepare_kernel_inputs(*(x[n] for n in names),A_log=x['A_log'],dt_bias=x['dt_bias'],
                                    namespace=namespace,**options)
    for index,(a,b) in enumerate(zip(got,old)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        if not flags[(0,0,1,2)[index]]:
            assert a is x[names[index]]
    for name,value in x.items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    if not any(flags):
        assert not native_abi
    else:
        expected=('forward_ready','backward_ready') if namespace=='chunk' else ('decode_ready',)
        assert tuple(native_abi[:len(expected)])==expected
        assert len(native_abi)==len(expected)+2*flags[0]+flags[1]+flags[2]


@pytest.mark.parametrize('name,flag',[(n,'use_qk_l2norm_in_kernel') for n in ('q','k')]
    +[(n,'use_gate_in_kernel') for n in ('g','A_log','dt_bias')]+[('beta','use_beta_sigmoid_in_kernel')])
@pytest.mark.parametrize('dtype',[torch.float16,torch.float64])
@pytest.mark.parametrize('entry',['chunk','decode','training_helper'])
def test_raw_dtype_rejected_before_compile(monkeypatch,name,flag,dtype,entry):
    x=raw_inputs(t=2 if entry=='decode' else 64)
    x[name]=x[name].to(dtype)
    monkeypatch.setattr(chunk,'_prep_runtime',lambda: pytest.fail('compiled before dtype rejection'),raising=False)
    monkeypatch.setattr(fused_recurrent,'_compiled',lambda *a,**kw: pytest.fail('decode compiled before dtype rejection'))
    with pytest.raises(ValueError,match=rf'{name}.*BF16 or FP32'):
        if entry=='training_helper':
            chunk._prepare_inputs(*(x[n] for n in ('q','k','g','beta')),
                A_log=x['A_log'],dt_bias=x['dt_bias'],**{flag:True})
        else:
            fn=autograd.chunk_kda if entry=='chunk' else fused_recurrent.fused_recurrent_kda
            fn(**x,**{flag:True},check_domain=False)


@pytest.mark.parametrize('training_leaf',['q','k','v','g','beta','A_log','dt_bias','initial_state'])
@pytest.mark.parametrize('enabled',[True,False])
def test_training_graph_selection_and_no_grad(monkeypatch,training_leaf,enabled):
    x=raw_inputs()
    x['initial_state']=torch.randn(1,2,128,128)
    x[training_leaf].requires_grad_()
    seen=[]
    real=chunk._prepare_inputs

    def host(*args,**kwargs):
        seen.append('training_host')
        return real(*args,**kwargs)

    def native(*args,**kwargs):
        seen.append('native')
        for key in ('device','block_dim','namespace','impl'):
            kwargs.pop(key)
        return real(*args,**kwargs)

    monkeypatch.setattr(autograd,'_prepare_inputs',host)
    monkeypatch.setattr(autograd,'_prepare_kernel_inputs',native,raising=False)
    monkeypatch.setattr(autograd._ChunkKDA,'apply',lambda *args: (x['v'],None))
    monkeypatch.setattr(autograd,'chunk_kda_fwd',lambda *args,**kwargs: (x['v'],None))
    with torch.set_grad_enabled(enabled):
        autograd.chunk_kda(**x,use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True)
    assert seen==(['training_host'] if enabled else ['native'])


def test_disabled_gate_parameters_do_not_select_training(monkeypatch):
    x=raw_inputs()
    q,k,g,beta=chunk._prepare_inputs(x['q'],x['k'],x['g'],x['beta'],A_log=x['A_log'],
        dt_bias=x['dt_bias'],use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True)
    x.update(q=q,k=k,g=g,beta=beta)
    x['A_log'].requires_grad_();x['dt_bias'].requires_grad_()
    monkeypatch.setattr(chunk,'_prep_runtime',lambda: pytest.fail('disabled flags launched preparation'),raising=False)
    monkeypatch.setattr(autograd._ChunkKDA,'apply',lambda *a: pytest.fail('unused parameter selected training'))
    monkeypatch.setattr(autograd,'chunk_kda_fwd',lambda *a,**kw: (x['v'],None))
    assert autograd.chunk_kda(**x)[0] is x['v']
