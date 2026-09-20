"""Three BF16 native launches; host allocation and dispatch only."""
import torch
from .stages import gdn_bf16_bwd_checkpoints, gdn_bf16_bwd_reverse, gdn_bf16_bwd_group_reduce


def entries():
    return gdn_bf16_bwd_checkpoints, gdn_bf16_bwd_reverse, gdn_bf16_bwd_group_reduce


def run(inputs, launch, *, retain_stages=False):
    B, T, H, _ = inputs['q'].shape
    HV = inputs['v'].shape[2]
    scalar = dict(B=B, T=T, H=H, HV=HV, N=T//64)
    device = inputs['q'].device

    def allocate(shapes, dtype=torch.float32):
        return {n: torch.empty(shape, dtype=dtype, device=device) for n, shape in shapes.items()}

    saved = launch(gdn_bf16_bwd_checkpoints,
                   {n: inputs[n] for n in ('k', 'v', 'g', 'beta')},
                   allocate(dict(checkpoints=(B,T//64,HV,128,128), final_state=(B,HV,128,128))), scalar)
    has_do, has_dht = inputs.get('do') is not None, inputs.get('dht') is not None
    # Full-shaped dummy pointers have a valid typed ABI but are never read when
    # absent. Actual zero initialization belongs to the reverse kernel.
    dout = inputs['do'] if has_do else torch.empty((B,T,HV,128), dtype=torch.bfloat16, device=device)
    dht = inputs['dht'] if has_dht else torch.empty((B,HV,128,128), dtype=torch.float32, device=device)
    sources = {n: inputs[n] for n in ('q', 'k', 'v', 'g', 'beta')}
    sources.update(dout=dout, dht=dht, checkpoints=saved['checkpoints'])
    part_outputs = allocate(dict(tape=(B,HV,64,128,128), dq_parts=(B,T,HV,128), dk_parts=(B,T,HV,128)))
    part_outputs.update(allocate(dict(dv=(B,T,HV,128)), torch.bfloat16))
    part_outputs.update(allocate(dict(dg=(B,T,HV), dbeta=(B,T,HV))))
    part = launch(gdn_bf16_bwd_reverse, sources, part_outputs,
                  dict(scalar, has_do=int(has_do), has_dht=int(has_dht)))
    reduced = launch(gdn_bf16_bwd_group_reduce,
                     {n: part[n] for n in ('dq_parts', 'dk_parts')},
                     allocate(dict(dq=(B,T,H,128), dk=(B,T,H,128)), torch.bfloat16),
                     {n: scalar[n] for n in ('B','T','H','HV')})
    if retain_stages:
        return {**saved, **part, **reduced}
    return {**reduced, **{n: part[n] for n in ('dv','dg','dbeta')}}
