"""Portable generated inputs and independent Torch CPU layout references."""
import math
import functools
import importlib.util
from pathlib import Path

import torch

DTYPES = {'bf16': torch.bfloat16, 'f32': torch.float32}



@functools.lru_cache(maxsize=1)
def _runtime():
    path = Path(__file__).with_name('runtime.py')
    spec = importlib.util.spec_from_file_location('_fmt02_unit_runtime', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def make_inputs(case):
    p = case['parameters']
    gen = torch.Generator().manual_seed(case['seed'])
    kind = p['kind']
    shapes = dict(to_head=(2,192,3,128), to_token=(2,3,3,64,64),
                  beta=(2,192,3), cast=(2,64), strided=(2,128,2), broadcast=(1,), zero=(1,))
    x = torch.randn(shapes[kind], generator=gen).to(DTYPES[p['source']])
    # Signed zeros and FP32 values precisely halfway between BF16 values.
    if kind == 'cast':
        x = (torch.arange(128, dtype=torch.float32) * (2**-7) + 1 + 2**-8).reshape(2,64)
        x[0,0] = -0.0
        x = x.to(DTYPES[p['source']])
    if kind == 'strided':
        x = x.view(-1)[1:]  # nonzero storage offset; two-element inner stride
    return dict(source=x, parameters=dict(p))


def reference(inputs):
    x, p = inputs['source'], inputs['parameters']
    kind, dtype = p['kind'], DTYPES[p['destination']]
    if kind == 'to_head':
        y = x.reshape(2,3,64,3,128).permute(0,3,1,2,4).contiguous()
    elif kind == 'to_token':
        y = x.permute(0,2,3,1,4).contiguous()
    elif kind == 'beta':
        y = x.reshape(2,3,64,3).permute(0,3,1,2).contiguous().view(2,3,3,1,64)
    elif kind == 'strided':
        y = x.as_strided((2,128), (256,2), x.storage_offset()).contiguous()
    elif kind == 'broadcast':
        y = x.expand(2,2,2,1,128).contiguous()
    elif kind == 'zero':
        y = torch.zeros(2,128, dtype=dtype)
    else:
        y = x
    if p.get('multiply',False):
        y = y * (1.0 / math.log(2.0))
    return {'destination': y.to(dtype).reshape(1,-1)}


def launch_description(inputs):
    p = inputs['parameters'];kind = p['kind'];x = inputs['source']
    options = {
        'to_head': ((2,3,3,64,128), (192*3*128,128,64*3*128,3*128,1)),
        'to_token': ((2,3,64,3,64), (3*3*64*64,64*64,64,3*64*64,1)),
        'beta': ((2,3,3,1,64), (192*3,1,64*3,0,3)),
        'cast': ((1,1,1,2,64), (0,0,0,64,1)),
        'strided': ((1,1,1,2,128), (0,0,0,256,2)),
        'broadcast': ((2,2,2,1,128), (0,0,0,0,0)),
        'zero': ((1,1,1,2,128), (0,0,0,0,0)),
    }
    shape, strides = options[kind];n = math.prod(shape)
    key = 'zero_'+p['destination'] if kind=='zero' else p['source']+'_'+p['destination']
    scalars = dict(N=n)
    if kind != 'zero':
        scalars = dict(Storage=x.numel(), N=n, **{f'D{i}':shape[i] for i in range(1,5)},
                       **{f'S{i}':strides[i] for i in range(5)})
        scalars.update(_runtime().tile_scalars(shape))
        if key=='f32_bf16':scalars.update(multiply=int(p.get('multiply',False)),factor=1.0/math.log(2.0))
    return key, shape, scalars


def kernel_for(case):
    runtime = _runtime()
    p=case['parameters']
    key='zero_'+p['destination'] if p['kind']=='zero' else p['source']+'_'+p['destination']
    return runtime.kernels()[key]


def execute(inputs, options):
    runtime = _runtime()
    from _unit_runner import launch_kernel
    if options['device']!='a5' or options['backend']!='cce' or options['block_dim'] not in (1,2,3,4):
        raise ValueError('layout unit requires A5/CCE with bd1..4')
    key, shape, scalars = launch_description(inputs)
    out = torch.full((1,math.prod(shape)), float('nan'), dtype=DTYPES[inputs['parameters']['destination']])
    args = (() if key.startswith('zero') else (inputs['source'].reshape(1,-1),)) + (out, *scalars.values())
    return {'destination': launch_kernel(runtime.kernels()[key], args, options)}


def validate_reference(inputs, expected, case):
    value = expected["destination"]
    assert value.dtype == DTYPES[case["parameters"]["destination"]]
    assert tuple(value.shape) == (1, case["parameters"]["N"])
