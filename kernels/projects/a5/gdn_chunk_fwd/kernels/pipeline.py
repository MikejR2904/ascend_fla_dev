"""Shared ordered launch graph for the standalone harness and public operator."""
from __future__ import annotations

# Ordered inputs/outputs form the complete workspace ABI; no hidden host math.
GRAPH = (
    ('prepare', ('q', 'k', 'v', 'g', 'beta'), ('qn', 'kn', 'gc', 'bk', 'wv')),
    ('scores', ('qn', 'kn', 'gc', 'bk'), ('lower', 'score')),
    ('wy', ('lower', 'gc', 'bk', 'wv'), ('u', 'wy')),
    ('scan', ('kn', 'gc', 'u', 'wy', 'initial_state'), ('states', 'delta', 'final_state')),
    ('output', ('qn', 'gc', 'score', 'states', 'delta'), ('o',)),
)


def entries():
    from .stages import STAGES
    return STAGES


def expand_inputs(inputs):
    """Materialize shared q/k on their current device; equal heads are a no-op.

    The CCE graph retains its equal-head ABI. Its internal H scalar denotes
    value heads after this exact, contiguous replication, with no CPU fallback.
    """
    ratio = inputs['v'].shape[2] // inputs['q'].shape[2]
    if ratio == 1:
        return inputs
    return dict(inputs, **{n: inputs[n].repeat_interleave(ratio, dim=2) for n in ('q', 'k')})


def run(inputs, launch, *, retain_stages=True):
    """Allocate fresh FP32 outputs and invoke each stage via launch(entry,...).

    launch accepts the entry, input/output dictionaries and runtime scalars,
    and returns its output dictionary. Both CPU-tensor harnesses and the
    in-process NPU bridge use this graph; neither computes reference math.
    """
    import torch
    inputs = expand_inputs(inputs)
    batch, time, heads, _ = inputs['q'].shape
    chunks = (time + 63) // 64
    scalars = dict(B=batch, T=time, H=heads, N=chunks)
    base = (batch, chunks, heads)
    shapes = {name: (*base, 64, 128) for name in ('qn', 'kn', 'gc', 'bk', 'wv', 'u', 'wy', 'delta')}
    shapes.update(lower=(*base, 64, 64), score=(*base, 64, 64),
                  states=(*base, 128, 128), o=(batch, time, heads, 128),
                  final_state=(batch, heads, 128, 128))
    values = dict(inputs)
    checkpoints = {}
    for index, (entry, (_, names, outputs)) in enumerate(zip(entries(), GRAPH)):
        fresh = {name: torch.empty(shapes[name], dtype=torch.float32,
                                  device=inputs['q'].device) for name in outputs}
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
            # Retire GM buffers after their last consuming launch. Keep the
            # two public outputs even when no later stage reads them.
            live = {'o', 'final_state'}
            for _, future_inputs, _ in GRAPH[index + 1:]:
                live.update(future_inputs)
            for name in tuple(values):
                if name not in live:
                    del values[name]
    return checkpoints if retain_stages else {name: values[name] for name in ('o', 'final_state')}
