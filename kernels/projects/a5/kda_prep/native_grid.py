"""Cartesian supported raw dtype/flag populations, generated independently on CPU."""
import itertools
import torch

TYPES={'bf16':torch.bfloat16,'f32':torch.float32}
FLAG_NAMES=('use_qk_l2norm_in_kernel','use_gate_in_kernel','use_beta_sigmoid_in_kernel')


def cases(route):
    values=('bf16',) if route=='chunk' else tuple(TYPES)
    for c,group,vd,flags in itertools.product((1,2,3),(1,2,4,8),values,itertools.product((False,True),repeat=3)):
        qks=tuple(itertools.product(TYPES,repeat=2)) if flags[0] else ((vd,vd),)
        gates=tuple(itertools.product(TYPES,repeat=3)) if flags[1] else (('f32','f32','f32'),)
        betas=tuple(TYPES) if flags[2] else ('f32',)
        for (q,k),(g,a,bias),beta in itertools.product(qks,gates,betas):
            labels=dict(q=q,k=k,v=vd,g=g,A_log=a,dt_bias=bias,beta=beta)
            yield dict(id=f'c{c}_g{group}_v{vd}_flags'+''.join(str(int(f)) for f in flags)+'_'+'_'.join((q,k,g,a,bias,beta)),
                       B=1,C=c,H=1,HV=group,flags=dict(zip(FLAG_NAMES,flags)),types=labels,seed=7011)


def inputs(case):
    gen=torch.Generator().manual_seed(case['seed'])
    b,t,h,hv=case['B'],case['C']*64,case['H'],case['HV']
    def randn(*shape):return torch.randn(*shape,generator=gen)
    x={n:randn(b,t,h,128) for n in ('q','k')}
    if not case['flags'][FLAG_NAMES[0]]:
        x.update({n:z/(z.square().sum(-1,keepdim=True)+1e-6).sqrt() for n,z in x.items()})
    x['v']=randn(b,t,hv,128)*.04
    x['g']=randn(b,t,hv,128)*.1 if case['flags'][FLAG_NAMES[1]] else -torch.rand(b,t,hv,128,generator=gen)*.03
    x['beta']=randn(b,t,hv) if case['flags'][FLAG_NAMES[2]] else torch.rand(b,t,hv,generator=gen)*.45+.05
    x['A_log']=torch.linspace(-.2,.2,hv)
    x['dt_bias']=torch.linspace(-.3,-.1,hv*128)
    x={n:z.to(TYPES[case['types'][n]]) for n,z in x.items()}
    x['h0']=randn(b,hv,128,128)*.01
    return x


def boundary_cases(route):
    value_types=('bf16',) if route=='chunk' else tuple(TYPES)
    for c,group,raw,vd,flags,kind in itertools.product((1,2,3),(1,2,4,8),TYPES,value_types,itertools.product((False,True),repeat=3),('zero','nearzero','threshold','beta_saturation')):
        types=dict(q=raw if flags[0] else vd,k=raw if flags[0] else vd,v=vd,
            g=raw if flags[1] else 'f32',A_log=raw if flags[1] else 'f32',dt_bias=raw if flags[1] else 'f32',beta=raw if flags[2] else 'f32')
        yield dict(id=f'{kind}_c{c}_g{group}_r{raw}_v{vd}_flags'+''.join(str(int(f)) for f in flags),
            B=1,C=c,H=1,HV=group,flags=dict(zip(FLAG_NAMES,flags)),types=types,seed=7011,boundary=kind)


def boundary_inputs(case):
    x=inputs(case);kind=case['boundary']
    if kind in ('zero','nearzero'):
        for n in ('q','k'):
            if kind=='zero':x[n].zero_()
            else:x[n]*=1e-20
    elif kind=='threshold' and case['flags'][FLAG_NAMES[1]]:
        values=torch.tensor([-40.,-20.,0.,19.999998092651367,20.,20.000001907348633,40.,80.])
        x['g']=values.repeat((x['g'].numel()+len(values)-1)//len(values))[:x['g'].numel()].reshape(x['g'].shape).to(x['g'].dtype)
        x['A_log'].fill_(-6.);x['dt_bias'].zero_()
    elif kind=='beta_saturation' and case['flags'][FLAG_NAMES[2]]:
        values=torch.tensor([-100.,-20.,-1.,0.,1.,20.,100.])
        x['beta']=values.repeat((x['beta'].numel()+len(values)-1)//len(values))[:x['beta'].numel()].reshape(x['beta'].shape).to(x['beta'].dtype)
    return x
