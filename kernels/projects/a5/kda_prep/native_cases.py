"""Generated native precision, range, and tail probes after the full workload."""
import itertools

import torch

TYPES={'bf16':torch.bfloat16,'f32':torch.float32}


def leaf_cases():
    generator=torch.Generator().manual_seed(7008)
    shape=(2,192,2,128)
    sample=torch.randn(shape,generator=generator)
    for source,target in itertools.product(TYPES,repeat=2):
        for scale in (0.,1e-20,1e-8,1e-4,1.,30.,1e10,1e18,1e20):
            yield dict(id=f'norm_{source}_{target}_scale{scale:g}',kind='norm',
                       types=f'{source}_{target}',numerical=scale<1e18,
                       inputs={'source':(sample*scale).to(TYPES[source])})
        x=torch.zeros(1,4,1,128,dtype=TYPES[source])
        x[0,0,0,0]=float('nan');x[0,1,0,0]=float('inf');x[0,2,0,0]=float('-inf');x[0,3]=-0.
        yield dict(id=f'norm_{source}_{target}_nonfinite',kind='norm',types=f'{source}_{target}',
                   numerical=False,inputs={'source':x})
    for labels in itertools.product(TYPES,repeat=3):
        types='_'.join(labels)
        for kind in ('random','threshold','subnormal','alog_range','nonfinite'):
            shape=(2,65,8,128)
            if kind=='random':
                g=torch.randn(shape,generator=generator)
                a=torch.linspace(-3.,2.7,8);bias=torch.linspace(-.2,.2,8*128)
            else:
                values={'threshold':[-40.,-20.,-1.,0.,1.,19.999998092651367,20.,20.000001907348633,40.,80.],
                        'subnormal':[-120.,-104.,-100.,-90.,-88.,-87.],
                        'alog_range':[-2.,-1.,0.,1.,2.],
                        'nonfinite':[float('nan'),float('inf'),float('-inf'),-0.,0.]}[kind]
                count=2*65*8*128
                g=torch.tensor(values).repeat((count+len(values)-1)//len(values))[:count].reshape(shape)
                a=torch.tensor([-120.,-104.,-100.,80.,88.,88.7,89.,100.]) if kind=='alog_range' else torch.linspace(-.2,.2,8)
                bias=torch.zeros(8*128)
            values={n:x.to(TYPES[t]) for n,x,t in zip(('source','alog','bias'),(g,a,bias),labels)}
            yield dict(id=f'gate_{types}_{kind}',kind='gate',types=types,
                       numerical=kind in ('random','threshold'),inputs=values)
    for dtype in TYPES:
        for n in (1,63,64,65,4095,4096,4097,8193):
            yield dict(id=f'beta_{dtype}_n{n}',kind='beta',types=dtype,numerical=True,
                       inputs={'source':torch.randn(1,n,1,generator=generator).to(TYPES[dtype])})
        for kind,values in (('saturation',[-120.,-104.,-100.,-90.,-88.,-40.,-20.,-1.,0.,1.,16.,20.,40.,100.]),
                            ('nonfinite',[float('nan'),float('inf'),float('-inf'),-0.])):
            yield dict(id=f'beta_{dtype}_{kind}',kind='beta',types=dtype,numerical=False,
                       inputs={'source':torch.tensor(values,dtype=TYPES[dtype]).reshape(1,-1,1)})


def semantic_comparison(actual,expected):
    a,e=actual.reshape(-1),expected.reshape(-1)
    finite=torch.isfinite(a)&torch.isfinite(e)
    same_sign=torch.signbit(a)==torch.signbit(e)
    return dict(nan_mask_equal=bool(torch.equal(torch.isnan(a),torch.isnan(e))),
                positive_infinity_mask_equal=bool(torch.equal(torch.isposinf(a),torch.isposinf(e))),
                negative_infinity_mask_equal=bool(torch.equal(torch.isneginf(a),torch.isneginf(e))),
                finite_count=int(finite.sum()),finite_differing_values=int((a[finite]!=e[finite]).sum()),
                finite_sign_differences=int((finite&~same_sign).sum()),
                expected_subnormal_count=int(((e.abs()<torch.finfo(e.dtype).tiny)&(e!=0)&torch.isfinite(e)).sum()),
                expected_subnormal_flushed_to_zero=int(((e.abs()<torch.finfo(e.dtype).tiny)&(e!=0)&(a==0)).sum()))
