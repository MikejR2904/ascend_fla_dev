"""Pre-kernel A/B calibration and FP64 finite differences; CPU only."""
import argparse
import json
from pathlib import Path
import time
import torch
from . import oracle
from .reference import NAMES, acceptable, analytical, forward, metrics


def inputs(B,T,H,HV,kind='random',seed=91000,K=128,V=128,dtype=torch.float32):
    gen=torch.Generator().manual_seed(seed+T+H+HV)
    q=torch.randn(B,T,H,K,generator=gen,dtype=dtype)*.05
    k=torch.randn(q.shape,generator=gen,dtype=dtype)*.05
    v=torch.randn(B,T,HV,V,generator=gen,dtype=dtype)*.05
    g=-torch.rand(B,T,HV,generator=gen,dtype=dtype)
    beta=torch.rand(g.shape,generator=gen,dtype=dtype)
    do=torch.randn(v.shape,generator=gen,dtype=dtype)*.1
    dht=torch.randn(B,HV,K,V,generator=gen,dtype=dtype)*.1
    if kind=='weak':g.mul_(.001)
    if kind=='zero_gate':g.zero_()
    if kind=='beta_zero':beta.zero_()
    if kind=='beta_one':beta.fill_(1.)
    if kind=='underflow':g.fill_(-120.)
    if kind=='spike':
        g.fill_(-1e-5);g[:,::64].fill_(-155.)
    if kind=='zero_qk':q.zero_();k.zero_()
    return q,k,v,g,beta,do,dht


def gradcheck():
    rows=[]
    for ratio in (1,2):
        xs=inputs(1,3,1,ratio,K=3,V=2,dtype=torch.float64)
        vals=tuple(x.requires_grad_() for x in xs[:5]);do,dh=xs[5:]
        def A(*args):
            o,h=oracle.outputs(*args,fp64=True)
            return (o*do).sum()+(h*dh).sum()
        class B(torch.autograd.Function):
            @staticmethod
            def forward(ctx,*args):
                ctx.save_for_backward(*args)
                o,h=forward(*args)
                return (o*do).sum()+(h*dh).sum()
            @staticmethod
            def backward(ctx,upstream):
                grads=analytical(*ctx.saved_tensors,do,dh)
                return tuple(upstream*grads[n] for n in NAMES)
        a=torch.autograd.gradcheck(A,vals,eps=1e-6,atol=1e-8,rtol=1e-5)
        b=torch.autograd.gradcheck(B.apply,vals,eps=1e-6,atol=1e-8,rtol=1e-5)
        rows.append(dict(ratio=ratio,A_precision_lift=a,B_analytical=b))
    return rows


def run(full=True):
    torch.set_num_threads(1)
    report=dict(scope='CPU reference calibration before kernel implementation; no native claim',
                torch=torch.__version__,oracle_pin=oracle.PIN,oracle_sha256=oracle.SHA256,
                A='Literal pinned FP32 naive autograd. Only FP64 gradcheck uses two dtype AST substitutions.',
                gradcheck=gradcheck(),cases=[],negative_controls=[])
    grid=[(1,T,1,r,'random') for T in (64,128,192) for r in (1,2,4,8)]
    grid += [(2,192,2,8,'weak')]+[(1,128,1,4,kind) for kind in ('zero_gate','beta_zero','beta_one','underflow','spike','zero_qk')]
    if full:grid += [(1,4096,8//r,8,'random') for r in (1,2,4,8)]
    for B,T,H,HV,kind in grid:
        xs=inputs(B,T,H,HV,kind)
        for mode in ('do','dht','both'):
            q,k,v,g,beta,do,dh=xs
            if mode=='do':dh=None
            if mode=='dht':do=None
            started=time.monotonic()
            a=oracle.autograd(q,k,v,g,beta,do,dh)
            b=analytical(q,k,v,g,beta,do,dh)
            numbers=metrics(b,a)
            row=dict(B=B,T=T,H=H,HV=HV,kind=kind,mode=mode,metrics=numbers,seconds=time.monotonic()-started)
            report['cases'].append(row)
            print(json.dumps(row),flush=True)
            assert acceptable(numbers,1e-5),row
            if mode=='both' and kind=='random' and T==64 and H==HV==1:
                for label,factor in (('zero',0.),('negated',-1.),('scaled_1.25',1.25)):
                    wrong={n:b[n]*factor for n in NAMES}
                    values=metrics(wrong,a)
                    assert all(m['relative_l2']>1e-4 for m in values.values()),values
                    report['negative_controls'].append(dict(label=label,metrics=values,rejected=True))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--quick',action='store_true');args=p.parse_args()
    result=run(not args.quick)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print('CALIBRATION PASS',len(result['cases']),flush=True)
