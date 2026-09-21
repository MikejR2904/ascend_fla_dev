"""Typed preparation ABI: metadata, output allocation, and native launches."""
from __future__ import annotations

import functools
import importlib.util
import itertools
from pathlib import Path

import torch

_DTYPES = {torch.bfloat16: 'bf16', torch.float32: 'f32'}
_BLOCK_DIMS = {'chunk': (1, 2, 3, 4), 'decode': (1, 2, 4, 8, 16, 28)}


@functools.lru_cache(maxsize=2)
def kernels(namespace='chunk'):
    if namespace not in _BLOCK_DIMS:
        raise ValueError('KDA prep namespace must be chunk or decode')
    from ascriptor.a5 import bf16, f32

    path = Path(__file__).parent / 'kernels' / 'forward.py'
    spec = importlib.util.spec_from_file_location('_afla_kda_prep_forward', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    types = {'bf16': bf16, 'f32': f32}
    result = {}
    for kind, factory, count in (('norm', module.make_norm, 2),
                                 ('gate', module.make_gate, 3),
                                 ('beta', module.make_beta, 1)):
        for labels in itertools.product(types, repeat=count):
            key = kind + '_' + '_'.join(labels)
            name = f'kda_prep_{namespace}_{key}_kernel'
            result[key] = factory(name, *(types[label] for label in labels))
    return result


@functools.lru_cache(maxsize=None)
def prepare(device='a5', block_dim=1, namespace='chunk'):
    """Register every dtype vendor before the process executes a custom op."""
    from ascend_fla.runtime.compile import compile_kernel

    if device != 'a5' or namespace not in _BLOCK_DIMS or block_dim not in _BLOCK_DIMS[namespace]:
        raise ValueError('KDA prep requires a5 and a supported chunk/decode block_dim')
    return {name: compile_kernel(fn, device=device, block_dim=block_dim)
            for name, fn in kernels(namespace).items()}


def _check_source(name, source):
    if not isinstance(source, torch.Tensor) or source.dtype not in _DTYPES:
        raise ValueError(f'{name} raw input must be BF16 or FP32; FP16/FP64 are unsupported')
    if source.device.type != 'npu' or not source.is_contiguous():
        raise ValueError(f'{name} raw input must be a contiguous NPU tensor')
    if source.numel() <= 0 or source.numel() >= 2**31:
        raise ValueError(f'{name} raw input size must be positive and fit signed int32')


def norm(source, dtype, *, device='a5', block_dim=1, namespace='chunk'):
    _check_source('q/k', source)
    if dtype not in _DTYPES or source.shape[-1] != 128:
        raise ValueError('KDA norm requires K=128 and BF16/FP32 output')
    destination = torch.empty(source.shape, dtype=dtype, device=source.device)
    key = f'norm_{_DTYPES[source.dtype]}_{_DTYPES[dtype]}'
    prepare(device, block_dim, namespace)[key](
        {'source': source.view(1, -1)}, {'N': source.numel()},
        {'destination': destination.view(1, -1)})
    return destination


def gate(source, alog, bias, *, device='a5', block_dim=1, namespace='chunk'):
    for name, value in (('g', source), ('A_log', alog), ('dt_bias', bias)):
        _check_source(name, value)
    if source.dim() != 4 or source.shape[-1] != 128:
        raise ValueError('KDA gate requires [B,T,HV,128]')
    hv = source.shape[-2]
    if alog.shape != (hv,) or bias.shape != (hv * 128,):
        raise ValueError('KDA gate requires A_log[HV] and dt_bias[HV*128]')
    if alog.device != source.device or bias.device != source.device:
        raise ValueError('KDA gate parameters must share the input device')
    destination = torch.empty(source.shape, dtype=torch.float32, device=source.device)
    key = f'gate_{_DTYPES[source.dtype]}_{_DTYPES[alog.dtype]}_{_DTYPES[bias.dtype]}'
    prepare(device, block_dim, namespace)[key](
        {'source': source.view(1, -1), 'alog': alog.view(1, -1), 'bias': bias.view(1, -1)},
        dict(N=source.numel(), HV=hv, Channels=hv*128, BT=source.shape[0]*source.shape[1]),
        {'destination': destination.view(1, -1)})
    return destination


def beta(source, *, device='a5', block_dim=1, namespace='chunk'):
    _check_source('beta', source)
    destination = torch.empty(source.shape, dtype=torch.float32, device=source.device)
    prepare(device, block_dim, namespace)[f'beta_{_DTYPES[source.dtype]}'](
        {'source': source.view(1, -1)}, {'N': source.numel()},
        {'destination': destination.view(1, -1)})
    return destination
