"""Standalone BF16 backward hooks; no sibling forward or backward dependency."""
from ref.bf16 import make_inputs,validate_inputs,reference,reference_stages,validate_reference,comparison,acceptable


def _launch(inputs,options):
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if options['device']!='a5' or options['backend']!='cce' or options['block_dim'] not in (1,2):
        raise ValueError('a5/cce/bd1-or2 required')
    def launch(entry,sources,outputs,scalars):
        for value in outputs.values(): value.fill_(float('nan'))
        result=launch_kernel(entry,tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values()),options)
        return dict(zip(outputs,(result,) if len(outputs)==1 else result))
    return launch


def execute_stages(inputs,options):
    from kernels.pipeline import run
    launch=_launch(inputs,options)
    known={**inputs,**reference_stages(inputs),'dout':inputs['do']}
    def independent(entry,sources,outputs,scalars):
        return launch(entry,{n:known[n].contiguous() for n in sources},outputs,scalars)
    result=run(inputs,independent,retain_stages=True)
    numbers=comparison(result,reference_stages(inputs),inputs['q'].dtype,stages=True)
    if not acceptable(numbers):raise AssertionError(numbers)
    return result


def execute(inputs,options):
    from kernels.pipeline import run
    result=run(inputs,_launch(inputs,options))
    numbers=comparison(result,reference(inputs),inputs['q'].dtype)
    if not acceptable(numbers):raise AssertionError(numbers)
    return result
