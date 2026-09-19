"""Native BF16 launch graph: allocation/layout only on the host."""
from __future__ import annotations
import ctypes
import functools
import math
import torch

OUTPUTS = ('o', 'final_state', 'final_A_state')
ERRORS = {
    1: 'all PKDA inputs must be finite',
    2: 'g must be finite and non-positive activated log decay',
    3: 'g_atk must be finite and non-positive activated log decay',
    4: 'beta must be finite and in [0,1]',
    5: 'beta_atk must be finite and in [0,1]',
    6: 'initial_A_state must be finite and non-negative',
    7: 'key row L2 norm must be <=1+1e-5; normalize explicitly if needed',
    8: 'forward per-chunk gate span must be <=155',
}


def entries():
    from .guard import pkda_bf16_init
    from .prepare import pkda_bf16_prepare
    from .stages import STAGES
    from .output import pkda_bf16_output
    return (pkda_bf16_init, pkda_bf16_prepare, *STAGES, pkda_bf16_output)


def validate_metadata(inputs, *, expected_device=None, input_dtype=torch.bfloat16):
    q = inputs['q']
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError('q must be BF16 [B,T,H,128]')
    B,T,H,K = q.shape
    if B not in (1,2) or not 1 <= T <= 4096 or not 1 <= H <= 32 or K != 128:
        raise ValueError('PKDA fixed domain: B=1/2,T=1..4096,H=1..32,K=V=128')
    if expected_device is not None and q.device.type != expected_device:
        raise ValueError(f'BF16 PKDA requires {expected_device} tensors')
    shapes = {n: (B,T,H,128) for n in ('q','k','v','g')}
    shapes.update({n: (B,T,H) for n in ('g_atk','beta_atk','beta')})
    shapes.update(initial_state=(B,H,128,128),initial_A_state=(B,H,128),log_atk_scale=(H,))
    for n,shape in shapes.items():
        x = inputs.get(n)
        if x is None and n in ('initial_state','initial_A_state','log_atk_scale'):
            continue
        dtype = input_dtype if n in ('q','k','v') else torch.float32
        if not isinstance(x,torch.Tensor) or tuple(x.shape) != shape or x.dtype != dtype:
            raise ValueError(f'{n} must be {dtype} {shape}')
        if x.device != q.device or not x.is_contiguous():
            raise ValueError(f'{n} must be contiguous and on the same device as q')
        if torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError('PKDA forward has no backward/autograd implementation')
    scale = float(inputs.get('scale',128**-.5))
    if not math.isfinite(scale) or scale <= 0 or scale > torch.finfo(torch.float32).max:
        raise ValueError('scale must be finite and positive')


@functools.lru_cache(maxsize=1)
def _acl_copy():
    # CANN acl_rt.h: aclError aclrtMemcpy(void*,size_t,const void*,size_t,
    # aclrtMemcpyKind); ACL_MEMCPY_DEVICE_TO_HOST is enum value2.
    lib = ctypes.CDLL('libascendcl.so')
    fn = lib.aclrtMemcpy
    fn.argtypes = [ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
    fn.restype = ctypes.c_int
    return lib,fn


def check_status(status):
    """Copy only device-produced control codes; no host tensor arithmetic/cast."""
    count = status.numel()
    codes = (ctypes.c_float * count)()
    size = ctypes.sizeof(codes)
    if status.device.type == 'npu':
        torch.npu.current_stream(status.device).synchronize()
        _,copy = _acl_copy()
        rc = copy(ctypes.addressof(codes),size,status.data_ptr(),size,2)
        if rc:
            raise RuntimeError(f'PKDA numeric guard status readback failed rc={rc}')
    elif status.device.type == 'cpu':
        ctypes.memmove(ctypes.addressof(codes),status.data_ptr(),size)
    else:
        raise ValueError('unsupported status device')
    for value in codes:
        if not math.isfinite(value) or value != int(value) or not 0 <= value <= 8:
            raise RuntimeError('PKDA numeric guard did not write a valid status')
    code = int(max(codes))
    if code:
        raise ValueError(ERRORS[code])


def run(inputs, launch, *, retain_stages=True):
    validate_metadata(inputs)
    q = inputs['q']; B,T,H,_ = q.shape; N = (T+63)//64
    scalars = dict(B=B,T=T,H=H,N=N)
    def alloc(shape,dtype=torch.float32):
        return torch.empty(shape,dtype=dtype,device=q.device)
    init,prep,scores,wy,scan,out = entries()
    initial = {}
    for name,shape in (('initial_state',(B,H,128,128)),('initial_A_state',(B,H,128)),('log_atk_scale',(H,))):
        value = inputs.get(name)
        initial[name] = alloc(shape) if value is None else value
    initial['log_atk_scale'] = initial['log_atk_scale'].view(1,H)
    values = dict(inputs)
    checkpoints = {}
    def call(entry,sources,shapes,attrs):
        fresh = {n:alloc(shape,torch.bfloat16 if n=='o' else torch.float32) for n,shape in shapes.items()}
        got = launch(entry,sources,fresh,attrs)
        if set(got) != set(fresh):
            raise RuntimeError(f'{entry.name}: incomplete outputs')
        if retain_stages:
            checkpoints.update(got)
        return got
    defaults = call(init,initial,dict(state=(B,H,128,128),astate=(B,H,128),center=(B,H,8),status=(B,H,64)),
                    dict(B=B,H=H,has_state=int(inputs.get('initial_state') is not None),
                         has_A=int(inputs.get('initial_A_state') is not None),has_center=int(inputs.get('log_atk_scale') is not None)))
    base = (B,N,H)
    source = {n:inputs[n] for n in ('q','k','v','g','g_atk','beta_atk','beta')}
    source.update(initial_A_state=defaults['astate'],log_atk_scale=defaults['center'],status_in=defaults['status'])
    shapes = {n:(*base,64,128) for n in ('qn','kn','gc','bk','wv')}
    shapes.update(final_A_state=(B,H,128),status_out=(B,H,64))
    values.update(call(prep,source,shapes,dict(scalars,scale=float(inputs.get('scale',128**-.5)))))
    check_status(values.pop('status_out'))
    values.update(call(scores,{n:values[n] for n in ('qn','kn','gc','bk')},
                       {n:(*base,64,64) for n in ('lower','score')},scalars))
    values.update(call(wy,{n:values[n] for n in ('lower','gc','bk','wv')},
                       {n:(*base,64,128) for n in ('u','wy')},scalars))
    for n in ('lower','bk','wv'):
        values.pop(n)
    source = {n:values[n] for n in ('kn','gc','u','wy')}
    source['initial_state'] = defaults['state']
    values.update(call(scan,source,dict(states=(*base,128,128),delta=(*base,64,128),final_state=(B,H,128,128)),scalars))
    for n in ('kn','u','wy'):
        values.pop(n)
    values.update(call(out,{n:values[n] for n in ('qn','gc','score','states','delta')},dict(o=(B,T,H,128)),scalars))
    return checkpoints if retain_stages else {n:values[n] for n in OUTPUTS}
