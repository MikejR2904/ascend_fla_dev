"""Native canaries for a4096-element batch boundary and its64-element tail."""
from pathlib import Path
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--block-dim', type=int, choices=(1,2,3,4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'retain prior evidence'
    sys.argv = ['verify_native.py', '--block-dim', str(args.block_dim), '--mode', 'compile',
                '--output', str(args.output)]
    import verify_native
    verify_native.main()
    import torch
    import native_support as check
    from ascend_fla.ops.kda import chunk
    torch.npu.set_device(0)
    runtime = chunk._layout_runtime()
    values = torch.randn(4160, generator=torch.Generator().manual_seed(206036))
    values[:8] = torch.tensor([0.,-0.,1+2**-8,1+3*2**-8,-(1+2**-8),2**-133,-2**-133,2**-126])
    rows = []
    for src, dst in [(a,b) for a in (torch.bfloat16,torch.float32) for b in (torch.bfloat16,torch.float32)] + [(None,torch.bfloat16),(None,torch.float32)]:
        key = ('zero' if src is None else ('bf16' if src == torch.bfloat16 else 'f32')) + '_' + ('bf16' if dst == torch.bfloat16 else 'f32')
        storage = torch.full((1,4288),7.5,dtype=dst).npu()
        actual = storage[:,64:4224]
        if src is None:
            inputs = {}
            scalars = dict(N=4160)
            expected = torch.zeros(1,4160,dtype=dst)
        else:
            source = values.to(src).npu().view(1,4160)
            inputs = dict(source=source)
            scalars = dict(Storage=4160,N=4160,D1=1,D2=1,D3=1,D4=4160,
                           S0=0,S1=0,S2=0,S3=0,S4=1,**runtime.tile_scalars((1,1,1,1,4160)))
            if key == 'f32_bf16': scalars.update(multiply=0,factor=1.)
            expected = values.to(src).to(dst).view(1,4160)
        before = {n: check.digest(t) for n,t in inputs.items()}
        with check.instrument(audit=False):
            runtime.prepare('a5',args.block_dim)[key](inputs,scalars,dict(destination=actual))
        torch.npu.synchronize()
        host = check.cpu(storage)
        comparison = check.exact(dict(destination=host[:,64:4224]),dict(destination=expected))
        guarded = bool((host[:,:64] == 7.5).all() and (host[:,4224:] == 7.5).all())
        unchanged = before == {n:check.digest(t) for n,t in inputs.items()}
        row = dict(operator=key,block_dim=args.block_dim,scalars=scalars,comparison=comparison,
                   canaries_intact=guarded,input_unchanged=unchanged,
                   passed=guarded and unchanged and comparison['destination']['passed'])
        rows.append(row)
        (args.output/'tails.json').write_text(json.dumps(dict(cases=rows,complete=False),indent=2)+'\n')
        assert row['passed'], row
    (args.output/'tails.json').write_text(json.dumps(dict(cases=rows,complete=True,passed=True),indent=2)+'\n')
    print(json.dumps(dict(block_dim=args.block_dim,cases=len(rows),passed=True)),flush=True)


if __name__ == '__main__':
    main()
