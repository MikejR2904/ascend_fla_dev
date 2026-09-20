"""Ordered PGDN graph; shared key-head ATK precedes value-head chunk work."""
from __future__ import annotations

GRAPH = (
    ('atk', ('q', 'k', 'g_atk', 'beta_atk'), ('q_norm', 'k_read', 'k_write', 'final_A_state')),
    ('prepare', ('q_norm', 'k_read', 'k_write', 'v', 'g', 'beta'), ('qn', 'kw', 'gc', 'bk', 'wv')),
    ('scores', ('qn', 'kw', 'gc', 'bk'), ('lower', 'score')),
    ('wy', ('lower', 'gc', 'bk', 'wv'), ('u', 'wy')),
    ('scan', ('kw', 'gc', 'u', 'wy', 'initial_state'), ('states', 'delta', 'final_state')),
    ('output', ('qn', 'gc', 'score', 'states', 'delta'), ('o',)),
)
PUBLIC_OUTPUTS = ('o', 'final_state', 'final_A_state')


def entries():
    from .atk import pgdn_bf03_atk
    from .stages import STAGES
    return (pgdn_bf03_atk, *STAGES)


def run(inputs, launch, *, retain_stages=True):
    """Launch device math using fresh outputs and retire buffers after last use."""
    import torch
    batch, time, heads, _ = inputs['q'].shape
    value_heads = inputs['v'].shape[2]
    chunks = time // 64
    all_scalars = dict(B=batch, T=time, H=heads, HV=value_heads, N=chunks)
    base = (batch, chunks, value_heads)
    shapes = {name: (*base, 64, 128) for name in ('qn', 'kw', 'gc', 'bk', 'wv', 'u', 'wy', 'delta')}
    shapes.update({name: tuple(inputs['q'].shape) for name in ('q_norm', 'k_read', 'k_write')})
    shapes.update(lower=(*base, 64, 64), score=(*base, 64, 64), states=(*base, 128, 128),
                  o=(batch, time, value_heads, 128), final_state=(batch, value_heads, 128, 128),
                  final_A_state=(batch, heads, 128))
    values = dict(inputs)
    checkpoints = {}
    for index, (entry, (_, names, outputs)) in enumerate(zip(entries(), GRAPH)):
        scalar_names = ('B', 'T', 'H', 'N') if index == 0 else (
            ('B', 'T', 'H', 'HV', 'N') if index == 1 else ('B', 'T', 'HV', 'N'))
        scalars = {name: all_scalars[name] for name in scalar_names}
        fresh = {name: torch.empty(shapes[name], dtype=torch.bfloat16 if name == 'o' else torch.float32, device=inputs['q'].device)
                 for name in outputs}
        if inputs['q'].device.type == 'cpu':
            for tensor in fresh.values():
                tensor.fill_(float('nan'))
        got = launch(entry, {name: values[name] for name in names}, fresh, scalars)
        if set(got) != set(outputs):
            raise RuntimeError(f'{entry.name}: incomplete stage outputs')
        values.update(got)
        if retain_stages:
            checkpoints.update(got)
        else:
            live = set(PUBLIC_OUTPUTS)
            for _, future_inputs, _ in GRAPH[index + 1:]:
                live.update(future_inputs)
            for name in tuple(values):
                if name not in live:
                    del values[name]
    return checkpoints if retain_stages else {name: values[name] for name in PUBLIC_OUTPUTS}
