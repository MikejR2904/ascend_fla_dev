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
