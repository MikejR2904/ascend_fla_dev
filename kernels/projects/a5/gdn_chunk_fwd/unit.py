"""Standalone GDN hooks with independent leaf inputs and actual composition."""
from ref.reference import make_inputs, reference, validate_inputs, validate_reference
from ref.stages import reference_stages


def _launch(inputs, options):
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if options['device'] != 'a5' or options['backend'] != 'cce':
        raise ValueError('only a5/cce is declared')
    if options['block_dim'] not in (1, 2):
        raise ValueError('block_dim must be 1 or 2')
    def launch(entry, sources, outputs, scalars):
        result = launch_kernel(entry, tuple(sources.values()) + tuple(outputs.values())
                               + tuple(scalars.values()), options)
        tensors = (result,) if len(outputs) == 1 else result
        return dict(zip(outputs, tensors))
    return launch


def execute_stages(inputs, options):
    from kernels.pipeline import run
    launch = _launch(inputs, options)
    upstream = {name: value.contiguous() for name, value in
                dict(inputs, **reference_stages(inputs)).items()}
    def independent(entry, sources, outputs, scalars):
        return launch(entry, {name: upstream[name] for name in sources}, outputs, scalars)
    return run(inputs, independent)


def execute(inputs, options):
    from kernels.pipeline import run
    stages = run(inputs, _launch(inputs, options))
    return {name: stages[name] for name in ('o', 'final_state')}
