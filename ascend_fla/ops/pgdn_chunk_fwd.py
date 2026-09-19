"""Grouped PGDN forward with naive FP32 normalization and separate ATK state."""
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
    path = Path(__file__).resolve().parents[2] / 'kernels/projects/a5/pgdn_chunk_fwd/kernels'
    name = '_afla_pgdn_chunk_kernels'
    spec = importlib.util.spec_from_file_location(name, path/'__init__.py', submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from importlib import import_module
    return import_module(name+'.pipeline')


def _options(device, block_dim):
    if device != 'a5' or type(block_dim) is not int or block_dim not in (1,2):
        raise ValueError(f'requires a5 and block_dim in (1,2); got {device}/{block_dim}')


@functools.lru_cache(maxsize=2)
def _compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(entry,device='a5',block_dim=block_dim,backend='cce') for entry in _pipeline().entries())


def prepare(*, device='a5', block_dim=2):
    """Compile all six stages before the first CANN operator resolution."""
    _options(device,block_dim)
    _compiled(block_dim)


def _constant(name, value, expected):
    if type(value) not in (float,int) or not math.isfinite(value) or value != expected:
        raise ValueError(f'{name} must be {expected}; got {value}')


def _validate(q,k,v,g_atk,g,beta_atk,beta,*,scale,initial_state,initial_A_state,
              use_qk_l2norm_in_kernel,x,eps,log_atk_scale,head_first,cu_seqlens,
              cu_seqlens_cpu,cp_context,transpose_state_layout,device,block_dim,launcher):
    _options(device,block_dim)
    if launcher not in ('inprocess','aclnn','board'):
        raise ValueError(f'unsupported launcher {launcher}')
    if head_first or transpose_state_layout:
        raise ValueError('requires token-major input and K-major state; head_first/transpose_state_layout unsupported')
    if any(value is not None for value in (cu_seqlens,cu_seqlens_cpu,cp_context)):
        raise ValueError('varlen and context parallelism are unsupported')
    if use_qk_l2norm_in_kernel is not True:
        raise ValueError('use_qk_l2norm_in_kernel must be True; FP32 naive normalization is required')
    _constant('scale',SCALE if scale is None else scale,SCALE)
    _constant('x',x,1.5)
    _constant('eps',eps,1e-6)
    _constant('log_atk_scale',-0.2 if log_atk_scale is None else log_atk_scale,-0.2)
    if q.ndim != 4 or min(q.shape[:3]) < 1 or q.shape[-1] != 128 or q.shape[1] % 64 or q.shape[1] > 4096:
        raise ValueError(f'requires positive B/H, T multiple of64 up to4096, K=128; got {tuple(q.shape)}')
    b,t,h,_ = q.shape
    if v.ndim != 4 or v.shape[:2] != (b,t) or v.shape[-1] != 128:
        raise ValueError(f'v requires [B,T,HV,128] with B/T={(b,t)}; got {tuple(v.shape)}')
    hv = v.shape[2]
    if hv < 1 or hv % h:
        raise ValueError(f'HV must be a positive multiple of H; got H={h}, HV={hv}')
    for name,value,shape in (('initial_state',initial_state,(b,hv,128,128)),('initial_A_state',initial_A_state,(b,h,128))):
        if value is not None:
            if not isinstance(value,torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f'{name} state layout is {shape}; got {getattr(value,"shape",type(value))}; only None supported')
            raise ValueError(f'{name} tensors are unsupported; use None for zero state')
    expected_device = 'npu' if launcher == 'inprocess' else 'cpu'
    for name,value in dict(q=q,k=k,v=v,g_atk=g_atk,g=g,beta_atk=beta_atk,beta=beta).items():
        shape = (b,t,h) if name in ('g_atk','beta_atk') else (b,t,hv) if name in ('g','beta') else v.shape if name=='v' else q.shape
        if value.shape != shape:
            raise ValueError(f'{name} requires {tuple(shape)} for H={h},HV={hv}; got {tuple(value.shape)}')
        if name in ('g','g_atk','beta','beta_atk'):
            if value.dtype != torch.float32:
                raise ValueError(f'{name} requires float32; got {value.dtype}')
        elif value.dtype != q.dtype or value.dtype not in (torch.float32,torch.bfloat16):
            raise ValueError('q/k/v require matching float32 or bfloat16 dtype')
        if value.device != q.device or value.device.type != expected_device:
            raise ValueError(f'{launcher} requires all tensors on one {expected_device} device')
        if not value.is_contiguous():
            raise ValueError(f'{name} must be contiguous token-major')
        if torch.is_grad_enabled() and value.requires_grad:
            raise RuntimeError('PGDN chunk is inference-only; backward is unavailable')
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'{name} must be finite')
    for name,value in (('g',g),('g_atk',g_atk)):
        if bool((value>0).any()):
            raise ValueError(f'{name} must be nonpositive')
        if not bool(torch.isfinite(value.reshape(b,t//64,64,-1).sum(2)).all()):
            raise ValueError(f'{name} chunk prefix must remain finite in FP32')
    for name,value in (('beta',beta),('beta_atk',beta_atk)):
        if bool(((value<0)|(value>1)).any()):
            raise ValueError(f'{name} must be in [0,1]')


def chunk_pgdn(q,k,v,g_atk,g,beta_atk,beta,*,scale=None,initial_state=None,initial_A_state=None,
               output_final_state=False,use_qk_l2norm_in_kernel=True,x=1.5,eps=1e-6,log_atk_scale=None,
               head_first=False,cu_seqlens=None,cu_seqlens_cpu=None,cp_context=None,
               transpose_state_layout=False,device='a5',block_dim=2,launcher='inprocess',
               board=None,out_dir=None,timeout=600):
    """Return (o, optional FP32 main state, optional FP32 ATK state).

    Consecutive HV/H value heads share one q/k head and one ATK state. Normalize
    in FP32 as x/max(norm(x),1e-12), intentionally following pinned FLA naive;
    the Triton chunk path has different near-zero/BF16 materialization semantics.
    Only zero initial states and default constants are supported. Value validation
    synchronizes NPU predicates before launch and belongs to public-call timing.
    CPU launchers explicitly transfer inputs; no reference or CPU math fallback.
    """
    _validate(q,k,v,g_atk,g,beta_atk,beta,scale=scale,initial_state=initial_state,initial_A_state=initial_A_state,
              use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,x=x,eps=eps,log_atk_scale=log_atk_scale,
              head_first=head_first,cu_seqlens=cu_seqlens,cu_seqlens_cpu=cu_seqlens_cpu,cp_context=cp_context,
              transpose_state_layout=transpose_state_layout,device=device,block_dim=block_dim,launcher=launcher)
    inputs = dict(q=q.float(),k=k.float(),v=v.float(),g_atk=g_atk,g=g,beta_atk=beta_atk,beta=beta,
                  initial_state=torch.zeros(q.shape[0],v.shape[2],128,128,dtype=torch.float32,device=q.device))
    if launcher=='inprocess':
        compiled = dict(zip((e.name for e in _pipeline().entries()),_compiled(block_dim)))
        def launch(entry,sources,outputs,scalars):
            op=compiled[entry.name]
            op(sources,{name:scalars[name] for name in op.scalar_names},outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry,sources,outputs,scalars):
            root=None if out_dir is None else Path(out_dir)/entry.name
            op=OpExec(entry,launcher=launcher,device=device,backend='cce',block_dim=block_dim,
                      board=board,out_dir=root,timeout=timeout)
            result=op(*(tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values())))
            return dict(zip(outputs,(result,) if len(outputs)==1 else result))
    outputs=_pipeline().run(inputs,launch,retain_stages=False)
    return (outputs['o'].to(q.dtype),outputs['final_state'] if output_final_state else None,
            outputs['final_A_state'] if output_final_state else None)
