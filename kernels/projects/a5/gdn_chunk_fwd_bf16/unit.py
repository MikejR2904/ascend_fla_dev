"""Canonical BF16 unit; FP32 compatibility is additionally checked natively."""
from ref.reference import make_inputs,reference,validate_inputs,validate_reference,acceptable
from ref.stages import reference_stages


def _launch(inputs,options):
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if options['device']!='a5' or options['backend']!='cce' or options['block_dim'] not in (1,2):
        raise ValueError('requires a5/cce block_dim1/2')
    def launch(entry,sources,outputs,scalars):
        # Test-only poison is outside the production pipeline/public host audit.
        for value in outputs.values():value.fill_(float('nan'))
        result=launch_kernel(entry,tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values()),options)
        return dict(zip(outputs,(result,) if len(outputs)==1 else result))
    return launch


def execute_stages(inputs,options):
    from kernels.pipeline import run
    launch=_launch(inputs,options)
    upstream={n:x.contiguous() for n,x in dict(inputs,**reference_stages(inputs)).items()}
    def independent(entry,sources,outputs,scalars):
        return launch(entry,{n:upstream[n] for n in sources},outputs,scalars)
    return run(inputs,independent)


def execute(inputs,options):
    from kernels.pipeline import run
    got=run(inputs,_launch(inputs,options),retain_stages=False)
    if not acceptable(got,reference(inputs),inputs['q'].dtype):
        raise AssertionError('fixed per-output min(1e-2,3F) comparison failed')
    return got
