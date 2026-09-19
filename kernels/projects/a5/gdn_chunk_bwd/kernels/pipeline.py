"""Three native launches; allocation/dispatch only, no host recurrence."""
import torch
from .stages import gdn_bwd_checkpoints, gdn_bwd_reverse, gdn_bwd_group_reduce


def entries():
    return gdn_bwd_checkpoints,gdn_bwd_reverse,gdn_bwd_group_reduce


def run(inputs,launch,*,retain_stages=False):
    B,T,H,K=inputs['q'].shape
    HV=inputs['v'].shape[2]
    scalar=dict(B=B,T=T,H=H,HV=HV,N=T//64)
    def allocate(**shapes):
        return {n:torch.empty(s,dtype=torch.float32,device=inputs['q'].device) for n,s in shapes.items()}
    def select(*names):
        return {n:inputs[n] for n in names}
    saved=launch(gdn_bwd_checkpoints,select('k','v','g','beta'),
                 allocate(checkpoints=(B,T//64,HV,128,128),final_state=(B,HV,128,128)),scalar)
    sources=select('q','k','v','g','beta','do','dht')
    sources = {('dout' if n == 'do' else n): t for n,t in sources.items()}
    sources['checkpoints']=saved['checkpoints']
    part=launch(gdn_bwd_reverse,sources,
                allocate(tape=(B,HV,64,128,128),dq_parts=(B,T,HV,128),dk_parts=(B,T,HV,128),
                         dv=(B,T,HV,128),dg=(B,T,HV),dbeta=(B,T,HV)),scalar)
    reduced=launch(gdn_bwd_group_reduce,{n:part[n] for n in ('dq_parts','dk_parts')},
                   allocate(dq=(B,T,H,128),dk=(B,T,H,128)),{n:scalar[n] for n in ('B','T','H','HV')})
    if retain_stages:
        return {**saved,**part,**reduced}
    return {**reduced,**{n:part[n] for n in ('dv','dg','dbeta')}}
