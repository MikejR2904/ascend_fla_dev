"""Non-GQA GDN chunk forward through repository-owned FP32 CCE stages."""
from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path
import sys

import torch

SCALE = 128**-0.5


@functools.lru_cache(maxsize=1)
def _pipeline():
    path = Path(__file__).resolve().parents[2] / 'kernels/projects/a5/gdn_chunk_fwd/kernels'
    name = '_afla_gdn_chunk_kernels'
    spec = importlib.util.spec_from_file_location(name, path / '__init__.py', submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name + '.pipeline')


@functools.lru_cache(maxsize=2)
def _compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(entry, device='a5', block_dim=block_dim, backend='cce')
                 for entry in _pipeline().entries())


def _options(device, block_dim):
    if device != 'a5' or type(block_dim) is not int or block_dim not in (1, 2):
        raise ValueError(f'requires a5 and block_dim in (1,2); got {device}/{block_dim}')


def prepare(*, device='a5', block_dim=2):
    """Compile all stages before the first CANN operator resolution."""
    _options(device, block_dim)
    _compiled(block_dim)


def _validate(q, k, v, g, beta, initial_state, scale, head_first, device, block_dim, launcher):
    _options(device, block_dim)
    if launcher not in ('inprocess', 'aclnn', 'board'):
        raise ValueError(f'unsupported launcher {launcher}')
    if head_first:
        raise ValueError('head_first is unsupported; require token-major [B,T,H,128]')
    if initial_state is not None:
        raise ValueError('initial_state tensors are unsupported; use None for zero state')
    if type(scale) not in (int, float) or not math.isclose(scale, SCALE, rel_tol=0., abs_tol=1e-12):
        raise ValueError(f'scale must be 128**-0.5; got {scale}')
    if q.ndim != 4 or min(q.shape[:3]) < 1 or q.shape[-1] != 128 or q.shape[1] % 64 or q.shape[1] > 4096:
        raise ValueError(f'requires positive B/H, T multiple of 64 up to 4096, D=128; got {tuple(q.shape)}')
    expected_device = 'npu' if launcher == 'inprocess' else 'cpu'
    for name, x in dict(q=q, k=k, v=v, g=g, beta=beta).items():
        shape = q.shape[:3] if name in ('g', 'beta') else q.shape
        if x.shape != shape:
            raise ValueError(f'{name} requires shape {tuple(shape)} (no GQA); got {tuple(x.shape)}')
        if name in ('g', 'beta'):
            if x.dtype != torch.float32:
                raise ValueError(f'{name} requires float32; got {x.dtype}')
        elif x.dtype != q.dtype or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('q/k/v require matching float32 or bfloat16 dtype')
        if x.device != q.device or x.device.type != expected_device:
            raise ValueError(f'{launcher} requires all tensors on one {expected_device} device')
        if not x.is_contiguous():
            raise ValueError(f'{name} must be contiguous token-major')
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('GDN chunk is inference-only; backward is unavailable')
        if expected_device == 'cpu' and not bool(torch.isfinite(x).all()):
            raise ValueError(f'{name} must be finite')
    if expected_device == 'cpu' and (bool((g > 0).any()) or bool(((beta < 0) | (beta > 1)).any())):
        raise ValueError('g<=0 and beta in [0,1] required')


def chunk_gdn(q, k, v, g, beta, *, initial_state=None, output_final_state=False,
              scale=SCALE, head_first=False, device='a5', block_dim=2,
              launcher='inprocess', board=None, out_dir=None, timeout=600):
    """Return token-major output and optional fresh FP32 K-major state.

    No q/k normalization or reference fallback. Finite g<=0, beta in [0,1]
    and finite q/k/v are caller preconditions on NPU, checked on CPU. CPU
    launchers explicitly transfer inputs and are not in-process timing paths.
    """
    _validate(q, k, v, g, beta, initial_state, scale, head_first, device, block_dim, launcher)
    inputs = dict(q=q.float(), k=k.float(), v=v.float(), g=g, beta=beta,
                  initial_state=torch.zeros(q.shape[0], q.shape[2], 128, 128, dtype=torch.float32).to(q.device))
    if launcher == 'inprocess':
        compiled = dict(zip((e.name for e in _pipeline().entries()), _compiled(block_dim)))
        def launch(entry, sources, outputs, scalars):
            op = compiled[entry.name]
            op(sources, {name: scalars[name] for name in op.scalar_names}, outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry, sources, outputs, scalars):
            root = None if out_dir is None else Path(out_dir) / entry.name
            op = OpExec(entry, launcher=launcher, device=device, backend='cce',
                        block_dim=block_dim, board=board, out_dir=root, timeout=timeout)
            result = op(*(tuple(sources.values()) + tuple(outputs.values()) + tuple(scalars.values())))
            return dict(zip(outputs, (result,) if len(outputs) == 1 else result))
    outputs = _pipeline().run(inputs, launch, retain_stages=False)
    return outputs['o'].to(q.dtype), outputs['final_state'] if output_final_state else None
