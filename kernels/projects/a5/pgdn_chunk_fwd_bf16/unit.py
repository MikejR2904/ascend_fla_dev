"""Standalone BF16 hooks with independent leaves and strict output budgets."""
import torch
from ref.reference import (make_inputs, reference, reference_stages, validate_inputs,
                           validate_reference, acceptable, budget, metric)


def _launch(inputs, options):
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if inputs['q'].dtype != torch.bfloat16:
        raise ValueError('this standalone unit requires BF16 q/k/v')
    if options['device'] != 'a5' or options['backend'] != 'cce' or options['block_dim'] not in (1, 2):
        raise ValueError('requires a5/cce with block_dim1/2')
    def launch(entry, sources, outputs, scalars):
        for tensor in outputs.values():
            tensor.fill_(float('nan'))
        result = launch_kernel(entry, tuple(sources.values()) + tuple(outputs.values()) + tuple(scalars.values()), options)
        return dict(zip(outputs, (result,) if len(outputs) == 1 else result))
    return launch


def execute_stages(inputs, options):
    from kernels.pipeline import run
    launch = _launch(inputs, options)
    expected = reference_stages(inputs)
    upstream = dict(inputs, **expected)
    def independent(entry, sources, outputs, scalars):
        return launch(entry, {name: upstream[name] for name in sources}, outputs, scalars)
    got = run(inputs, independent)
    if metric(got['o'], expected['o'])['relative_l2'] > budget(expected['o'], torch.bfloat16, 'o'):
        raise AssertionError('independent output stage exceeded fixed min(1e-2,3F)')
    return got


def execute(inputs, options):
    from kernels.pipeline import run
    got = run(inputs, _launch(inputs, options), retain_stages=False)
    if not acceptable(got, reference(inputs), torch.bfloat16):
        raise AssertionError('fixed PGDN per-output comparison failed')
    return got
