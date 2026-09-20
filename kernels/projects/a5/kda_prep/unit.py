"""Generated inputs, independent precision references, and source-file execution."""
import functools
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DTYPES = {'bf16': torch.bfloat16, 'f32': torch.float32}


@functools.lru_cache(maxsize=None)
def module(name):
    path = ROOT / ('ref/calibrate.py' if name == 'precision' else name + '.py')
    spec = importlib.util.spec_from_file_location('_bf07_unit_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def make_inputs(case):
    p = case['parameters']
    generator = torch.Generator().manual_seed(case['seed'])
    types = p['types'].split('_')
    shape = tuple(p['shape'])
    values = dict(source=torch.randn(shape,generator=generator).to(DTYPES[types[0]]))
    if p['kind']=='gate':
        hv=shape[-2]
        values.update(alog=torch.linspace(-.3,.3,hv).to(DTYPES[types[1]]),
                      bias=torch.linspace(-.2,.2,hv*128).to(DTYPES[types[2]]))
    return dict(values=values,parameters=dict(p))


def reference(inputs):
    """CPU FP64 precision study; end-to-end goldens remain CPU FP32."""
    v=inputs['values'];kind=inputs['parameters']['kind'];precision=module('precision')
    if kind=='norm':
        result=precision.high_precision_norm(v['source'])
    elif kind=='gate':
        result=precision.high_precision_gate(v['source'],v['alog'],v['bias'])
    else:
        result=precision.high_precision_beta(v['source'])
    return {'destination':result.reshape(1,-1)}


def host_reference(inputs):
    v=inputs['values'];p=inputs['parameters'];source=v['source'].float()
    if p['kind']=='norm':
        value=source/(torch.sum(source*source,-1,keepdim=True)+1e-6).sqrt()
        value=value.to(DTYPES[p['types'].split('_')[1]])
    elif p['kind']=='gate':
        value=-v['alog'].float().exp().view(-1,1)*torch.nn.functional.softplus(
            source+v['bias'].float().view(source.shape[-2:]))
    else:
        value=source.sigmoid()
    return {'destination':value.reshape(1,-1)}


def kernel_for(case):
    p=case['parameters']
    return module('runtime').kernels()[p['kind']+'_'+p['types']]


def execute(inputs, options):
    from _unit_runner import launch_kernel

    p=inputs['parameters'];v=inputs['values'];source=v['source'];n=source.numel()
    output_dtype=DTYPES[p['types'].split('_')[1]] if p['kind']=='norm' else torch.float32
    out=torch.full((1,n),float('nan'),dtype=output_dtype)
    if p['kind']=='gate':
        hv=source.shape[-2];bt=source.shape[0]*source.shape[1]
        args=(source.reshape(1,-1),v['alog'].reshape(1,-1),v['bias'].reshape(1,-1),out,n,hv,hv*128,bt)
    else:
        args=(source.reshape(1,-1),out,n)
    entry=module('runtime').kernels()[p['kind']+'_'+p['types']]
    actual=launch_kernel(entry,args,options)
    return {'destination':actual}


def compare(inputs,actual,expected,budgets):
    p=inputs['parameters'];key=p['kind']+':'+p['types'];limit=budgets['limits'][key]
    precision=module('precision')
    a=actual['destination'];high=precision.metrics(a,expected['destination'])
    host=precision.metrics(a,host_reference(inputs)['destination'])
    passed=(high['finite_pairs']==a.numel() and high['relative_l2'] is not None
        and high['relative_l2']<=limit['relative_l2']
        and high['max_relative_nonzero']<=limit['max_relative_nonzero']
        and high['zero_reference_nonzero_actual']==0)
    if a.dtype==torch.bfloat16:
        passed=passed and all(m['rounded_reference_over_one_ulp']==0 for m in (high,host))
    return dict(to_fp64=high,to_predecessor_fp32=host,budget=limit,passed=passed)
