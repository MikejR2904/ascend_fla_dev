"""Independent CPU FP32 block solve; no device implementation imports."""
import torch

NAMES = ('o', 'final_state')
SCALE = 128**-0.5


def cases():
    result = []
    for ratio in (1, 2, 4, 8):
        for chunks in (64, 1, 2, 3):
            h = 8 // ratio if chunks == 64 else 2
            result.append(dict(id=f'r{ratio}_c{chunks}', seed=171000+ratio*10+chunks,
                               B=1, T=chunks*64, H=h, HV=h*ratio, gate_scale=1.))
    extras = [
        ('batch2_c2', dict(B=2, T=128, H=3, HV=6)),
        ('batch2_c3', dict(B=2, T=192, H=3, HV=12)),
        ('uneven_c1', dict(T=64, H=1, HV=3)),
        ('uneven_c3', dict(T=192, H=1, HV=3)),
        ('gate_zero', dict(gate_scale=0.)),
        ('gate_small', dict(gate_scale=.03)),
        ('gate_large', dict(gate_scale=30., normalized_keys=True)),
        ('gate_underflow', dict(gate_constant=-1000.)),
        ('beta_zero', dict(beta_constant=0.)),
        ('beta_one', dict(beta_constant=1., normalized_keys=True)),
        ('zero_qk', dict(zero_qk=True)),
        ('spike', dict(spike=True)),
    ]
    for index, (name, overrides) in enumerate(extras):
        row=dict(id=name, seed=172000+index, B=1, T=192, H=2, HV=4, gate_scale=1.)
        row.update(overrides); result.append(row)
    return result


def make_inputs(case):
    p = case.get('parameters', case)
    rng = torch.Generator().manual_seed(case['seed'])
    b,t,h,hv=(p[n] for n in ('B','T','H','HV'))
    dtype = getattr(torch, p.get('dtype','bfloat16'))
    data={n:torch.randn(b,t,hv if n=='v' else h,128,generator=rng)*.05
          for n in ('q','k','v')}
    if p.get('normalized_keys'):data['k']=torch.nn.functional.normalize(data['k'],dim=-1)
    if p.get('zero_qk'):
        data['q'].zero_();data['k'].zero_()
    if p.get('spike'):
        data['q'][:,::64,:,7]=1.;data['k'][:,63::64,:,71]=.5
    data={n:x.to(dtype) for n,x in data.items()}
    data['g']=-torch.rand(b,t,hv,generator=rng)*p.get('gate_scale',1.)
    data['beta']=torch.rand(b,t,hv,generator=rng)
    if 'gate_constant' in p:data['g'].fill_(p['gate_constant'])
    if 'beta_constant' in p:data['beta'].fill_(p['beta_constant'])
    return data


def validate_inputs(inputs, case=None):
    if set(inputs)!= {'q','k','v','g','beta'}:raise ValueError('expected q/k/v/g/beta')
    q=inputs['q'];v=inputs['v']
    if q.ndim!=4 or q.shape[-1]!=128 or min(q.shape[:3])<1 or q.shape[1]%64 or q.shape[1]>4096:
        raise ValueError('positive B/H, T multiple64 <=4096, K=128 required')
    b,t,h,_=q.shape
    if v.ndim!=4 or v.shape[:2]!=(b,t) or v.shape[-1]!=128 or v.shape[2]<1 or v.shape[2]%h:
        raise ValueError('v requires [B,T,HV,128], HV positive multiple of H')
    for n,x in inputs.items():
        shape=v.shape[:3] if n in ('g','beta') else (v.shape if n=='v' else q.shape)
        dtype=torch.float32 if n in ('g','beta') else q.dtype
        if x.shape!=shape or x.dtype!=dtype or q.dtype not in (torch.float32,torch.bfloat16):
            raise ValueError(f'{n}: invalid shape/dtype')
        if x.device.type!='cpu' or not x.is_contiguous() or not torch.isfinite(x).all():
            raise ValueError(f'{n}: finite contiguous CPU tensor required by reference')
    if (inputs['g']>0).any() or ((inputs['beta']<0)|(inputs['beta']>1)).any():
        raise ValueError('g<=0 and beta in[0,1] required')


def reference(inputs):
    validate_inputs(inputs)
    q,k,v,beta,g=(inputs[n].float() for n in ('q','k','v','beta','g'))
    h,hv=q.shape[2],v.shape[2]
    # Independently enumerated contiguous groups, not the kernel's indexing code.
    q,k=(torch.stack([x[:,:,j//(hv//h)] for j in range(hv)],dim=2).transpose(1,2)
         for x in (q,k))
    v,beta,g=(x.transpose(1,2) for x in (v,beta,g))
    state=torch.zeros(q.shape[0],hv,128,128)
    output=torch.empty_like(v)
    eye=torch.eye(64);causal=torch.ones(64,64,dtype=torch.bool).tril()
    for start in range(0,q.shape[2],64):
        sl=slice(start,start+64)
        kc,qc,vc,bc=k[:,:,sl],q[:,:,sl],v[:,:,sl],beta[:,:,sl]
        prefix=g[:,:,sl].cumsum(-1)
        decay=(prefix.unsqueeze(-1)-prefix.unsqueeze(-2)).masked_fill(~causal,-torch.inf).exp()
        lower=((kc@kc.transpose(-1,-2))*decay*bc.unsqueeze(-1)).tril(-1)
        rhs=bc.unsqueeze(-1)*(vc-prefix.exp().unsqueeze(-1)*(kc@state))
        updates=torch.linalg.solve_triangular(eye+lower,rhs,upper=False,unitriangular=True)
        output[:,:,sl]=((qc@state)*prefix.exp().unsqueeze(-1)
                       +((qc@kc.transpose(-1,-2))*decay)@updates)*SCALE
        weights=(prefix[...,-1:]-prefix).exp()
        state=state*prefix[...,-1:].exp().unsqueeze(-1)+(kc*weights.unsqueeze(-1)).transpose(-1,-2)@updates
    return dict(o=output.transpose(1,2).contiguous(),final_state=state)


def metric(actual,expected):
    a,e=actual.float().double(),expected.float().double()
    error=(a-e).norm().item();norm=e.norm().item()
    return dict(relative_l2=error/norm if norm else (0. if error==0 else float('inf')),
                max_abs=(a-e).abs().max().item(),finite=bool(torch.isfinite(a).all()))


def floor(expected):
    return metric(expected.bfloat16().float(),expected)['relative_l2']


def budget(expected,dtype):
    return min(.01,3*floor(expected)) if dtype==torch.bfloat16 else 1e-4


def acceptable(actual,expected,dtype):
    return (set(actual)==set(expected)==set(NAMES)
            and all(metric(actual[n],expected[n])['finite']
                    and metric(actual[n],expected[n])['relative_l2']<=budget(expected[n],dtype)
                    for n in NAMES))


def validate_reference(inputs,outputs,case=None):
    b,t,hv,_=inputs['v'].shape
    for n,shape in [('o',(b,t,hv,128)),('final_state',(b,hv,128,128))]:
        x=outputs[n]
        if x.shape!=shape or x.dtype!=torch.float32 or not torch.isfinite(x).all():
            raise ValueError(f'invalid reference {n}')
        if x.norm()>0:
            for wrong in (torch.zeros_like(x),-x,x*1.25):
                assert metric(wrong,x)['relative_l2']>budget(x,inputs['q'].dtype)
