"""BF16 storage preparation and fixed FP32 comparison; reference-only code."""
import torch
from .calibrate import inputs as generated_inputs
from .reference import NAMES, analytical
from .stages import reference_stages as fp32_stages

INPUTS = ('q','k','v','g','beta','do','dht')
OPERANDS = ('q','k','v','do')
NARROW_OUTPUTS = ('dq','dk','dv')


def make_inputs(case, dtype=torch.bfloat16):
    p = case['parameters']
    values = generated_inputs(p['B'],p['T'],p['H'],p['HV'],p.get('kind','random'),seed=case['seed'])
    result = {n: x.to(dtype) if n in OPERANDS else x for n,x in zip(INPUTS,values)}
    if p['mode'] == 'do': result['dht'].zero_()
    if p['mode'] == 'dht': result['do'].zero_()
    return result


def fp32_inputs(inputs):
    return {n: x.float() for n,x in inputs.items()}


def reference(inputs):
    values = fp32_inputs(inputs)
    return analytical(*(values[n] for n in INPUTS))


def reference_stages(inputs):
    return fp32_stages(fp32_inputs(inputs))


def metric(actual, expected):
    assert actual.shape == expected.shape
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    if not finite:
        return dict(relative_l2='infinity',max_abs='infinity',finite=False,reference_norm='nonfinite')
    delta, norm = (a-b).norm().item(), b.norm().item()
    return dict(relative_l2=delta/norm if norm else (0. if delta == 0 else 'infinity'),
                max_abs=(a-b).abs().max().item(),finite=True,reference_norm=norm)


def floor(expected):
    return metric(expected.bfloat16().float(),expected)['relative_l2']


def comparison(actual, expected, dtype, *, stages=False):
    result = {}
    for n,target in expected.items():
        value = actual[n]
        wanted = torch.bfloat16 if dtype == torch.bfloat16 and n in NARROW_OUTPUTS else torch.float32
        assert value.dtype == wanted, (n,value.dtype,wanted)
        m = metric(value,target)
        if dtype == torch.bfloat16 and n in NAMES:
            f = floor(target)
            assert isinstance(f,float)
            limit = min(1e-2,3*f)
        else:
            f, limit = None, 1e-4
        m.update(floor=f,budget=limit)
        m['passed'] = m['finite'] and isinstance(m['relative_l2'],float) and m['relative_l2'] <= limit
        result[n] = m
    return result


def acceptable(metrics):
    return all(m['passed'] for m in metrics.values())


def validate_inputs(inputs, case=None):
    if set(inputs) != set(INPUTS): raise ValueError('expected seven explicit stage inputs')
    q,v = inputs['q'],inputs['v']
    if q.ndim != 4 or min(q.shape[:3]) < 1 or q.shape[-1] != 128 or q.shape[1]%64 or q.shape[1] > 4096:
        raise ValueError('require positive B/H, T multiple of64 through4096, K128')
    B,T,H,_ = q.shape
    if v.ndim != 4 or v.shape[:2] != (B,T) or v.shape[-1] != 128 or v.shape[2] < 1 or v.shape[2]%H:
        raise ValueError('invalid grouped v')
    HV = v.shape[2]
    shapes = dict(q=q.shape,k=q.shape,v=v.shape,g=(B,T,HV),beta=(B,T,HV),do=v.shape,dht=(B,HV,128,128))
    if q.dtype not in (torch.float32,torch.bfloat16): raise ValueError('q dtype')
    for n,x in inputs.items():
        dtype = q.dtype if n in OPERANDS else torch.float32
        if x.shape != shapes[n] or x.dtype != dtype or x.device.type != 'cpu' or not x.is_contiguous() or not bool(torch.isfinite(x).all()):
            raise ValueError('invalid '+n)
    if bool((inputs['g']>0).any()) or bool(((inputs['beta']<0)|(inputs['beta']>1)).any()): raise ValueError('gate domain')
    if case and case.get('block_dim',1) not in (1,2): raise ValueError('block_dim')


def validate_reference(inputs, outputs, case=None):
    if set(outputs) != set(NAMES): raise ValueError('gradient names')
    for n,x in outputs.items():
        src = dict(dq='q',dk='k',dv='v',dg='g',dbeta='beta')[n]
        if x.shape != inputs[src].shape or x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
            raise ValueError('reference '+n)
