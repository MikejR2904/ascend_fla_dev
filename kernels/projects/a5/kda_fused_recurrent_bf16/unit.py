"""Portable BF16 unit; the public FP32 route is covered by verify_native.py."""
import os

import torch

import research


def make_inputs(case):
    return research.make_inputs(case['parameters'], seed=case['seed'])


def validate_inputs(inputs, case=None):
    q = inputs['q']
    b, t, h, k = q.shape
    hv = inputs['v'].shape[2]
    if k != 128 or not 1 <= t <= 16 or min(b, h, hv) < 1 or hv % h:
        raise ValueError('decode requires positive B/H/HV, HV multiple of H, T1..16 and K=V128')
    shapes = dict(q=(b,t,h,128), k=(b,t,h,128), v=(b,t,hv,128),
                  g=(b,t,hv,128), beta=(b,t,hv), initial_state=(b,hv,128,128))
    for name, shape in shapes.items():
        x = inputs[name]
        if name == 'initial_state' and x is None:
            continue
        dtype = torch.bfloat16 if name in ('q','k','v') else torch.float32
        if x.shape != shape or x.dtype != dtype or not x.is_contiguous() or x.device.type != 'cpu':
            raise ValueError(f'{name} requires contiguous CPU {dtype} {shape}')
        if not torch.isfinite(x).all():
            raise ValueError(f'{name} must be finite')


def _references(inputs):
    validate_inputs(inputs)
    return research.references(inputs, research.load_fla(os.environ['BF06_FLA_NAIVE']))


def reference(inputs):
    return dict(zip(('o', 'final_state'), _references(inputs)['A']))


def execute(inputs, options):
    from kernels.step import kda_decode_bf16_kernel
    from _unit_runner import launch_kernel

    refs = _references(inputs)
    if options['device'] != 'a5' or options['backend'] != 'cce':
        raise ValueError('only A5/CCE is declared')
    b, t, h, _ = inputs['q'].shape
    hv = inputs['v'].shape[2]
    state0 = inputs['initial_state']
    final = torch.full((b,hv,128,128), float('nan'), dtype=torch.float32)
    out = torch.full((b,t,hv,128), float('nan'), dtype=torch.bfloat16)
    # Distinct unread poison allocation also exercises the absent-state branch.
    if state0 is None:
        state0 = torch.full_like(final, float('nan'))
    arguments = tuple(inputs[n] for n in ('q','k','v','g','beta')) + (
        state0, out, final, b, t, h, hv, int(inputs['initial_state'] is not None), inputs['scale'])
    got = launch_kernel(kda_decode_bf16_kernel, arguments, options)
    result = research.compare(got, refs)
    options.setdefault('_execution_evidence', []).append(dict(dual_oracle=result))
    if not all(x['passed'] for x in result.values()):
        raise AssertionError(result)
    return dict(zip(('o', 'final_state'), got))
