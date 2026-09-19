"""Generated CPU FP32 inputs and strict input/output contracts; no DSL imports."""
from __future__ import annotations
import math
import torch

D = 128
C = 64
MAX_GATE_SPAN = 155.0
OUTPUTS = ('o', 'final_state', 'final_A_state')


def make_inputs(case):
    p = case['parameters']
    B, T, H = (int(p[n]) for n in ('B', 'T', 'H'))
    rng = torch.Generator().manual_seed(int(case['seed']))
    shape = (B, T, H, D)
    # Input generation only. The public/kernel path never normalizes inputs.
    q, k = (torch.nn.functional.normalize(torch.randn(shape, generator=rng), dim=-1)
            for _ in range(2))
    if 'row_norm' in p:
        q *= p['row_norm']; k *= p['row_norm']
    v = torch.randn(shape, generator=rng) * 0.25
    g = -torch.rand(shape, generator=rng) * float(p.get('gate_scale', 0.1))
    if 'gate_span' in p:
        g.fill_(-float(p['gate_span']) / C)
    g_atk = -torch.rand((B, T, H), generator=rng) * 0.4
    beta_atk = torch.rand((B, T, H), generator=rng)
    beta = torch.rand((B, T, H), generator=rng)
    if p.get('zero_atk', False):
        beta_atk.zero_()
    state = torch.randn((B, H, D, D), generator=rng) * float(p.get('state_scale', 0.05))
    a = torch.rand((B, H, D), generator=rng) * float(p.get('A_scale', 0.03))
    center = torch.linspace(-0.5, 0.1, H)
    return dict(q=q, k=k, v=v, g=g, g_atk=g_atk, beta_atk=beta_atk,
                beta=beta, initial_state=state, initial_A_state=a, log_atk_scale=center)


def validate_inputs(inputs, case=None):
    required = {'q', 'k', 'v', 'g', 'g_atk', 'beta_atk', 'beta',
                'initial_state', 'initial_A_state', 'log_atk_scale'}
    if set(inputs) - required - {'scale'} or required - set(inputs):
        raise ValueError('PKDA input names do not match the declared ABI')
    q = inputs['q']
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError('q must be FP32 [B,T,H,128]')
    B, T, H, K = q.shape
    if B not in (1, 2) or not 1 <= T <= 4096 or not 1 <= H <= 32 or K != D:
        raise ValueError('PKDA fixed domain: B=1/2,T=1..4096,H=1..32,K=V=128')
    shapes = {n: (B,T,H,D) for n in ('q','k','v','g')}
    shapes.update({n: (B,T,H) for n in ('g_atk','beta_atk','beta')})
    shapes.update(initial_state=(B,H,D,D), initial_A_state=(B,H,D), log_atk_scale=(H,))
    for name, shape in shapes.items():
        x = inputs[name]
        if not isinstance(x, torch.Tensor) or tuple(x.shape) != shape or x.dtype != torch.float32:
            raise ValueError(f'{name} must be float32 {shape}')
        if not x.is_contiguous() or x.device != q.device:
            raise ValueError(f'{name} must be contiguous and on the same device as q')
        if not bool(torch.isfinite(x).all()):
            raise ValueError(f'{name} must be finite')
    for n in ('g', 'g_atk'):
        if bool((inputs[n] > 0).any()):
            raise ValueError(f'{n} must be non-positive activated log decay')
    for n in ('beta','beta_atk'):
        if bool(((inputs[n] < 0) | (inputs[n] > 1)).any()):
            raise ValueError(f'{n} must be in [0,1]')
    if bool((inputs['initial_A_state'] < 0).any()):
        raise ValueError('initial_A_state must be non-negative')
    if bool((inputs['k'].square().sum(-1) > (1.0 + 1e-5)**2).any()):
        raise ValueError('key row L2 norm must be <=1+1e-5; normalize explicitly if needed')
    for start in range(0, T, C):
        if bool((-inputs['g'][:, start:start+C].sum(1) > MAX_GATE_SPAN).any()):
            raise ValueError('forward per-chunk gate span must be <=155')
    scale = float(inputs.get('scale', D**-0.5))
    if not math.isfinite(scale) or scale <= 0 or scale > torch.finfo(torch.float32).max:
        raise ValueError('scale must be finite and positive')
    if case is not None and case.get('block_dim', 1) not in (1,2,3,4):
        raise ValueError('block_dim must be 1,2,3 or4')


def reference(inputs):
    from .stages import reference_stages
    validate_inputs(inputs)
    stages = reference_stages(inputs)
    return {name: stages[name] for name in OUTPUTS}


def validate_reference(inputs, outputs, case=None):
    del case
    B,T,H,_ = inputs['q'].shape
    shapes = dict(o=(B,T,H,D), final_state=(B,H,D,D), final_A_state=(B,H,D))
    if set(outputs) != set(shapes):
        raise ValueError('all three PKDA outputs are required')
    for n, shape in shapes.items():
        t = outputs[n]
        if tuple(t.shape) != shape or t.dtype != torch.float32 or not bool(torch.isfinite(t).all()):
            raise ValueError(f'{n}: expected finite FP32 {shape}')
