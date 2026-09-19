"""Canonical unit hooks for standalone GDN backward and independent leaves."""
import torch
from ref.calibrate import inputs as generated_inputs
from ref.reference import NAMES,acceptable,analytical,metrics
from ref.stages import reference_stages


def make_inputs(case):
    p=case['parameters']
    values=list(generated_inputs(p['B'],p['T'],p['H'],p['HV'],p.get('kind','random'),seed=case['seed']))
    if p.get('bf16_values',False):
        for i in (0,1,2,5):values[i]=values[i].bfloat16().float()
    if p['mode']=='do':values[6].zero_()
    if p['mode']=='dht':values[5].zero_()
    return dict(zip(('q','k','v','g','beta','do','dht'),values))


def validate_inputs(inputs,case=None):
    if set(inputs)!={'q','k','v','g','beta','do','dht'}:raise ValueError('expected all seven explicit stage inputs')
    # Keep this project independently copyable without the public repository.
    q=inputs['q'];v=inputs['v']
    if q.ndim!=4 or min(q.shape[:3])<1 or q.shape[-1]!=128 or q.shape[1]%64 or q.shape[1]>4096:
        raise ValueError('require positive B/H,T%64=0<=4096,K128')
    B,T,H,_=q.shape
    if v.ndim!=4 or v.shape[:2]!=(B,T) or v.shape[-1]!=128 or v.shape[2]<1 or v.shape[2]%H:raise ValueError('invalid grouped v')
    HV=v.shape[2]
    shapes=dict(q=q.shape,k=q.shape,v=v.shape,g=(B,T,HV),beta=(B,T,HV),do=v.shape,dht=(B,HV,128,128))
    for n,x in inputs.items():
        if x.shape!=shapes[n] or x.dtype!=torch.float32 or x.device.type!='cpu' or not x.is_contiguous() or not bool(torch.isfinite(x).all()):raise ValueError('invalid '+n)
    if bool((inputs['g']>0).any()) or bool(((inputs['beta']<0)|(inputs['beta']>1)).any()):raise ValueError('gate domain')
    if case and case.get('block_dim',1) not in (1,2):raise ValueError('block_dim')


def reference(inputs):
    validate_inputs(inputs)
    return analytical(*(inputs[n] for n in ('q','k','v','g','beta','do','dht')))


def validate_reference(inputs,outputs,case=None):
    for name,x in outputs.items():
        source=dict(dq='q',dk='k',dv='v',dg='g',dbeta='beta')[name]
        if x.shape!=inputs[source].shape or x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()):raise ValueError('reference '+name)
        if x.double().norm()>0:
            for factor in (0.,-1.,1.25):
                error=((x.double()*factor-x.double()).norm()/x.double().norm()).item()
                assert error>1e-4


def _launch(inputs,options):
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if options['device']!='a5' or options['backend']!='cce' or options['block_dim'] not in (1,2):raise ValueError('a5/cce/bd1-or2 required')
    def launch(entry,sources,outputs,scalars):
        result=launch_kernel(entry,tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values()),options)
        return dict(zip(outputs,(result,) if len(outputs)==1 else result))
    return launch


def execute_stages(inputs,options):
    from kernels.pipeline import run
    launch=_launch(inputs,options)
    known={**inputs,**reference_stages(inputs),'dout':inputs['do']}
    def independent(entry,sources,outputs,scalars):
        return launch(entry,{n:known[n] for n in sources},outputs,scalars)
    return run(inputs,independent,retain_stages=True)


def execute(inputs,options):
    from kernels.pipeline import run
    return run(inputs,_launch(inputs,options))
