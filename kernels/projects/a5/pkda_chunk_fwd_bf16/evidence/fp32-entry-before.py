"""Experimental FP32 PKDA forward using five Ascriptor stages.

This entry follows the pinned FLA naive semantics: q/k are consumed as
supplied. It does not normalize them, activate gates, cast inputs or provide a
CPU-reference fallback. Host/model qualification is separate from native use.
"""
from __future__ import annotations
import functools
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import torch


@functools.lru_cache(maxsize=None)
def _module(suffix):
    name = '_afla_pkda_forward_unit'
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[2] / 'kernels/projects/a5/pkda_chunk_fwd'
        spec = importlib.util.spec_from_file_location(name, path / '__init__.py',
                                                     submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(name + '.' + suffix)


def _options(device, block_dim):
    if device != 'a5' or type(block_dim) is not int or block_dim not in (1,2,3,4):
        raise ValueError('PKDA requires device=a5 and block_dim in (1,2,3,4)')


@functools.lru_cache(maxsize=4)
def _compiled(block_dim):
    from ..runtime.compile import compile_kernel
    return tuple(compile_kernel(entry, device='a5', block_dim=block_dim, backend='cce')
                 for entry in _module('kernels.pipeline').entries())


def prepare(*, device='a5', block_dim=1):
    """Compile all five native vendors before the first custom-kernel call."""
    _options(device, block_dim)
    _compiled(block_dim)


def chunk_precond_kda(
    q, k, v, g, g_atk, beta_atk, beta, scale=None, initial_state=None,
    initial_A_state=None, output_final_state=False, use_gate_in_kernel=False,
    safe_gate=False, lower_bound=None, cu_seqlens=None, cu_seqlens_cpu=None,
    chunk_indices=None, cp_context=None, transpose_state_layout=False,
    x=1.5, eps=1e-6, log_atk_scale=None, solve_tril_precision=None,
    disable_recompute=False, return_intermediate_states=False, *,
    device='a5', block_dim=1, launcher='inprocess', board=None, out_dir=None,
    timeout=300, **kwargs,
):
    """Return (o, optional main state, optional ATK state), all FP32.

    Fixed domain: B=1/2, T=1..4096, H=1..32, K=V=128; contiguous tensors;
    key row norm <=1+1e-5, activated non-positive gates, beta/beta_atk in[0,1],
    non-negative A and per-chunk gate span<=155. See the unit contract.
    Use launcher='sim'/'pipesim' for explicit CPU model execution. No native
    qualification is implied by accepting a board/aclnn/inprocess launcher.
    """
    _options(device, block_dim)
    if launcher not in ('inprocess','sim','pipesim','board','aclnn'):
        raise ValueError('unsupported PKDA launcher')
    if (use_gate_in_kernel or safe_gate or lower_bound is not None or
        cu_seqlens is not None or cu_seqlens_cpu is not None or chunk_indices is not None or
        cp_context is not None or transpose_state_layout or solve_tril_precision is not None or
        disable_recompute or return_intermediate_states or kwargs):
        raise NotImplementedError('PKDA supports activated fixed-length forward only; unsupported FLA options')
    if x != 1.5 or eps != 1e-6:
        raise ValueError('PKDA ATK constants are fixed at x=1.5, eps=1e-6')
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError('q must be FP32 [B,T,H,128]')
    B,T,H,K = q.shape
    if B not in (1,2) or not 1 <= T <= 4096 or not 1 <= H <= 32 or K != 128:
        raise ValueError('PKDA fixed domain: B=1/2,T=1..4096,H=1..32,K=V=128')
    if q.dtype != torch.float32:
        raise ValueError('PKDA accepts FP32 only; BF16 is not silently cast')
    expected_device = 'npu' if launcher == 'inprocess' else 'cpu'
    if q.device.type != expected_device:
        raise ValueError(f'{launcher} requires {expected_device} tensors')
    scale = 128**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0 or scale > torch.finfo(torch.float32).max:
        raise ValueError('scale must be finite and positive')
    # Default-value initialization is the only tensor preparation done here.
    if initial_state is None:
        initial_state = torch.zeros((B,H,128,128), dtype=torch.float32, device=q.device)
    if initial_A_state is None:
        initial_A_state = torch.zeros((B,H,128), dtype=torch.float32, device=q.device)
    if log_atk_scale is None:
        log_atk_scale = torch.full((H,), -0.2, dtype=torch.float32, device=q.device)
    inputs = dict(q=q,k=k,v=v,g=g,g_atk=g_atk,beta_atk=beta_atk,beta=beta,
                  initial_state=initial_state,initial_A_state=initial_A_state,
                  log_atk_scale=log_atk_scale,scale=scale)
    _module('ref.reference').validate_inputs(inputs)
    if torch.is_grad_enabled() and any(t.requires_grad for t in inputs.values() if isinstance(t,torch.Tensor)):
        raise RuntimeError('PKDA forward has no backward/autograd implementation')
    pipeline = _module('kernels.pipeline')
    if launcher == 'inprocess':
        compiled = dict(zip((entry.name for entry in pipeline.entries()), _compiled(block_dim), strict=True))
        def launch(entry, sources, outputs, scalars):
            op = compiled[entry.name]
            op(sources, {n: scalars[n] for n in op.scalar_names}, outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry, sources, outputs, scalars):
            root = None if out_dir is None else Path(out_dir) / entry.name
            ex = OpExec(entry, launcher=launcher, device=device, backend='cce',
                        block_dim=block_dim, board=board, out_dir=root, timeout=timeout)
            result = ex(*(tuple(sources.values()) + tuple(outputs.values()) + tuple(scalars.values())))
            return dict(zip(outputs, (result,) if len(outputs)==1 else result, strict=True))
    got = pipeline.run(inputs, launch, retain_stages=False)
    return got['o'], (got['final_state'] if output_final_state else None), (got['final_A_state'] if output_final_state else None)
