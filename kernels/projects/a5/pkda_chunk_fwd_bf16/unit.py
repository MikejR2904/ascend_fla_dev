"""Standalone unit; CPU golden stays independent from kernel execution."""
from ref.reference import make_inputs, reference, validate_inputs, validate_reference


def execute_stages(inputs, options):
    from kernels.pipeline import run
    from _unit_runner import launch_kernel
    validate_inputs(inputs)
    if options['device']!='a5' or options['backend']!='cce' or options['block_dim'] not in (1,2,3,4):
        raise ValueError('PKDA BF16 requires a5/cce and block_dim1..4')
    def launch(entry,sources,outputs,scalars):
        for t in outputs.values():
            t.fill_(float('nan'))
        result=launch_kernel(entry,tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values()),options)
        return dict(zip(outputs,(result,) if len(outputs)==1 else result,strict=True))
    got=run(inputs,launch)
    return {n:t for n,t in got.items() if n not in ('state','astate','center','status','status_out')}


def execute(inputs, options):
    from ref.checks import references,compare
    refs=references(inputs)
    stages=execute_stages(inputs,options)
    got={n:stages[n] for n in ('o','final_state','final_A_state')}
    result=compare(got,refs)
    if not result['passed']:
        raise AssertionError(result)
    options.setdefault('_execution_evidence',[]).append(dict(dual_oracle=result))
    return got


def reference_stages(inputs):
    from ref.reference import fp32_inputs
    from ref.stages import reference_stages as stages
    return stages(fp32_inputs(inputs))
