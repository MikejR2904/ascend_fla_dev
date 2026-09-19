"""Fixed dual-oracle, per-output BF16-floor budgets for verifier use only."""
import torch
from .reference import fp32_inputs
from .stages import reference_stages
from .oracles import oracle


def metric(a,b):
    d=a.float()-b.float()
    return dict(relative_l2=float(d.norm()/b.float().norm().clamp_min(1e-30)),max_abs=float(d.abs().max()))


def references(inputs):
    data=fp32_inputs(inputs)
    a=oracle(data);stages=reference_stages(data);b={n:stages[n] for n in a}
    floors={n:{name:metric(ref[n].to(torch.bfloat16),ref[n])['relative_l2'] for name,ref in (('A',a),('B',b))} for n in ('o','final_state')}
    budgets={n:min(.01,3*min(v.values())) for n,v in floors.items()}
    budgets['final_A_state']=1e-4
    return dict(A=a,B=b,stages=stages,floors=floors,budgets=budgets)


def compare(actual,refs):
    errors={name:{n:metric(actual[n],ref[n]) for n in ('o','final_state','final_A_state')} for name,ref in ((n,refs[n]) for n in ('A','B'))}
    finite=all(bool(torch.isfinite(actual[n]).all()) for n in ('o','final_state','final_A_state'))
    passed=finite and all(v['relative_l2']<=refs['budgets'][n] for error in errors.values() for n,v in error.items())
    return dict(passed=passed,finite=finite,errors=errors,floors=refs['floors'],budgets=refs['budgets'])
