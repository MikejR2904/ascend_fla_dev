"""Five launches; host work consists only of allocation and dispatch."""
GRAPH = (
    ('prepare', ('q','k','v','g','beta'), ('qn','kn','gc','bk','wv')),
    ('scores', ('qn','kn','gc','bk'), ('lower','score')),
    ('wy', ('lower','gc','bk','wv'), ('u','wy')),
    ('scan', ('kn','gc','u','wy'), ('states','delta','final_state')),
    ('output', ('qn','gc','score','states','delta'), ('o',)),
)


def entries(dtype):
    import torch
    from .stages import STAGES_BF16, STAGES_F32
    if dtype==torch.bfloat16:return STAGES_BF16
    if dtype==torch.float32:return STAGES_F32
    raise ValueError('matching BF16 or FP32 q/k/v required')


def all_entries():
    import torch
    return tuple(dict.fromkeys(entries(torch.float32)+entries(torch.bfloat16)))


def run(inputs,launch,*,retain_stages=True):
    import torch
    b,t,hk,_=inputs['q'].shape;h=inputs['v'].shape[2];n=t//64
    scalars=dict(B=b,T=t,H=h,N=n)
    base=(b,n,h)
    shapes={name:(*base,64,128) for name in ('qn','kn','gc','bk','wv','u','wy','delta')}
    shapes.update(lower=(*base,64,64),score=(*base,64,64),states=(*base,128,128),
                  o=(b,t,h,128),final_state=(b,h,128,128))
    values=dict(inputs);checkpoints={}
    for index,(entry,(_,names,outputs)) in enumerate(zip(entries(inputs['q'].dtype),GRAPH)):
        fresh={name:torch.empty(shapes[name],device=inputs['q'].device,
                               dtype=inputs['q'].dtype if name=='o' else torch.float32)
               for name in outputs}
        params=dict(scalars,HK=hk) if index==0 else scalars
        got=launch(entry,{name:values[name] for name in names},fresh,params)
        if set(got)!=set(outputs):raise RuntimeError(f'{entry.name}: incomplete outputs')
        values.update(got)
        if retain_stages:checkpoints.update(got)
        else:
            live={'o','final_state'}
            for _,future_inputs,_ in GRAPH[index+1:]:live.update(future_inputs)
            for name in tuple(values):
                if name not in live:del values[name]
    return checkpoints if retain_stages else {n:values[n] for n in ('o','final_state')}
