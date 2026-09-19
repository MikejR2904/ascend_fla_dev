"""Independent stage boundaries, using CPU matmul for the primal contraction."""
import torch
from .reference import analytical


@torch.no_grad()
def reference_stages(inputs):
    q,k,v,g,beta,do,dht=(inputs[n] for n in ('q','k','v','g','beta','do','dht'))
    B,T,H,K=q.shape;HV=v.shape[2];ratio=HV//H
    keys=k.repeat_interleave(ratio,2)
    state=q.new_zeros(B,HV,128,128)
    checkpoints=q.new_empty(B,T//64,HV,128,128)
    tape=q.new_empty(B,HV,64,128,128)
    for t in range(T):
        if t%64==0:checkpoints[:,t//64]=state
        if t<64:tape[:,:,t]=state
        key=keys[:,t]
        d=g[:,t].exp()[...,None,None]*state
        residual=v[:,t]-torch.matmul(key.unsqueeze(-2),d).squeeze(-2)
        state=d+key.unsqueeze(-1)*(beta[:,t,:,None]*residual).unsqueeze(-2)
    grads=analytical(q.repeat_interleave(ratio,2),keys,v,g,beta,do,dht)
    dq_parts,dk_parts=grads.pop('dq'),grads.pop('dk')
    return dict(checkpoints=checkpoints,final_state=state,tape=tape,dq_parts=dq_parts,dk_parts=dk_parts,
                dq=dq_parts.reshape(B,T,H,ratio,K).sum(3),dk=dk_parts.reshape(B,T,H,ratio,K).sum(3),**grads)
