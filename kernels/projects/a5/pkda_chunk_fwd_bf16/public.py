"""BF16-specific dispatch. Tensor math and conversions belong to the kernels."""
from __future__ import annotations
import functools
from pathlib import Path
from .kernels import pipeline


@functools.lru_cache(maxsize=4)
def compiled(block_dim):
    from ascend_fla.runtime.compile import compile_kernel
    return tuple(compile_kernel(entry,device='a5',block_dim=block_dim,backend='cce')
                 for entry in pipeline.entries())


def execute(inputs, *, output_final_state, block_dim, launcher, board, out_dir, timeout):
    pipeline.validate_metadata(inputs,expected_device='npu' if launcher=='inprocess' else 'cpu')
    if launcher == 'inprocess':
        vendors = dict(zip((e.name for e in pipeline.entries()),compiled(block_dim),strict=True))
        def launch(entry,sources,outputs,scalars):
            op = vendors[entry.name]
            op(sources,{n:scalars[n] for n in op.scalar_names},outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry,sources,outputs,scalars):
            root = None if out_dir is None else Path(out_dir)/entry.name
            ex = OpExec(entry,launcher=launcher,device='a5',backend='cce',block_dim=block_dim,
                        board=board,out_dir=root,timeout=timeout)
            result = ex(*(tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values())))
            return dict(zip(outputs,(result,) if len(outputs)==1 else result,strict=True))
    got = pipeline.run(inputs,launch,retain_stages=False)
    return got['o'],(got['final_state'] if output_final_state else None),(got['final_A_state'] if output_final_state else None)


@functools.lru_cache(maxsize=4)
def compiled_fp32_guard(block_dim):
    from ascend_fla.runtime.compile import compile_kernel
    from .kernels.fp32_guard import pkda_fp32_guard
    return compile_kernel(pkda_fp32_guard,device='a5',block_dim=block_dim,backend='cce')


def execute_fp32(inputs, *, original_pipeline, original_compiled, output_final_state,
                 block_dim, launcher, board, out_dir, timeout):
    import torch
    from .kernels.fp32_guard import pkda_fp32_guard
    pipeline.validate_metadata(inputs,expected_device='npu' if launcher=='inprocess' else 'cpu',input_dtype=torch.float32)
    if launcher=='inprocess':
        guard=compiled_fp32_guard(block_dim)
        vendors=dict(zip((e.name for e in original_pipeline.entries()),original_compiled(block_dim),strict=True))
        vendors[pkda_fp32_guard.name]=guard
        def launch(entry,sources,outputs,scalars):
            op=vendors[entry.name];op(sources,{n:scalars[n] for n in op.scalar_names},outputs)
            return outputs
    else:
        from ascriptor.runtime import OpExec
        def launch(entry,sources,outputs,scalars):
            root=None if out_dir is None else Path(out_dir)/entry.name
            ex=OpExec(entry,launcher=launcher,device='a5',backend='cce',block_dim=block_dim,
                      board=board,out_dir=root,timeout=timeout)
            result=ex(*(tuple(sources.values())+tuple(outputs.values())+tuple(scalars.values())))
            return dict(zip(outputs,(result,) if len(outputs)==1 else result,strict=True))
    q=inputs['q'];B,T,H,_=q.shape
    def alloc(shape):return torch.empty(shape,dtype=torch.float32,device=q.device)
    sources={n:inputs[n] for n in ('q','k','v','g','g_atk','beta_atk','beta')}
    for n,shape in (('initial_state',(B,H,128,128)),('initial_A_state',(B,H,128)),('log_atk_scale',(H,))):
        x=inputs.get(n);sources[n]=alloc(shape) if x is None else x
    sources['log_atk_scale']=sources['log_atk_scale'].view(1,H)
    outputs={n:alloc(shape) for n,shape in dict(state=(B,H,128,128),astate=(B,H,128),center=(1,H),status=(B,H,64)).items()}
    scalars=dict(B=B,T=T,H=H,N=(T+63)//64,has_state=int(inputs.get('initial_state') is not None),
                 has_A=int(inputs.get('initial_A_state') is not None),has_center=int(inputs.get('log_atk_scale') is not None))
    defaults=launch(pkda_fp32_guard,sources,outputs,scalars);pipeline.check_status(defaults['status'])
    prepared=dict(inputs,initial_state=defaults['state'],initial_A_state=defaults['astate'],log_atk_scale=defaults['center'])
    got=original_pipeline.run(prepared,launch,retain_stages=False)
    return got['o'],(got['final_state'] if output_final_state else None),(got['final_A_state'] if output_final_state else None)
