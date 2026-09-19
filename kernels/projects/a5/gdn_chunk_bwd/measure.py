"""Same-device saved-checkpoint cost baseline / complete backward sandwich.

The baseline is the same task-owned adjoint with its boundary checkpoints
already supplied. It is not another full backward implementation and does not
include generating those checkpoints. Candidate is the complete public API.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from ascend_fla.ops.gdn_chunk_bwd import prepare,chunk_gdn_bwd,_compiled,_pipeline,_validate
from ref.calibrate import inputs
from ref.reference import NAMES,analytical,acceptable,metrics


def sample(fn,warmup,repeat):
    for _ in range(warmup):fn()
    torch.npu.synchronize()
    raw=[]
    for _ in range(repeat):
        torch.npu.synchronize()
        start=time.perf_counter_ns()
        result=fn()
        torch.npu.synchronize()
        raw.append((time.perf_counter_ns()-start)*1e-6)
    return dict(samples_ms=raw,mean_ms=statistics.mean(raw),median_ms=statistics.median(raw))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--block-dim',type=int,choices=(1,2),default=2)
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--repeat',type=int,default=50);args=p.parse_args()
    if args.warmup<0 or args.repeat<1:raise ValueError('positive repeat and nonnegative warmup required')
    torch.set_num_threads(1)
    import torch_npu
    assert torch.npu.device_count()==1
    torch.npu.set_device(0);prepare(block_dim=args.block_dim)
    ops=dict(zip((e.name for e in _pipeline().entries()),_compiled(args.block_dim)))
    def launch(entry,sources,outputs,scalars):
        op=ops[entry.name]
        op(sources,{n:scalars[n] for n in op.scalar_names},outputs)
        return outputs
    report=dict(stage='native_same_card_sandwich',baseline='Task-owned saved-boundary-checkpoint adjoint cost baseline; excludes boundary generation, not a separate complete backward implementation',
                candidate='Complete public chunk_gdn_bwd including promotions, allocation, checkpoint recomputation, all3launches and storage casts',
                block_dim=args.block_dim,warmup=args.warmup,repeat=args.repeat,synchronize='before and after every timed call',rounds=3,cases=[])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    for T in (1024,4096):
        for dtype in (torch.float32,torch.bfloat16):
            values=list(inputs(1,T,8,8,seed=91900))
            for i in (0,1,2,5):values[i]=values[i].to(dtype)
            cpu=dict(zip(('q','k','v','g','beta','do','dht'),values))
            public={n:t.npu() for n,t in cpu.items()}
            math_inputs={n:t.float() for n,t in public.items()}
            known=_pipeline().run(math_inputs,launch,retain_stages=True)
            cached={n:known[n] for n in ('checkpoints','final_state')}
            del known
            def cached_launch(entry,sources,outputs,scalars):
                if entry.name=='gdn_bwd_checkpoints':return cached
                return launch(entry,sources,outputs,scalars)
            @torch.no_grad()
            def baseline():
                _validate(**public,block_dim=args.block_dim)
                data={n:t.float() for n,t in public.items()}
                result=_pipeline().run(data,cached_launch)
                return tuple(result[n].to(dtype) if n in ('dq','dk','dv') else result[n] for n in NAMES)
            def candidate():return chunk_gdn_bwd(**public,block_dim=args.block_dim)
            expected=analytical(*(x.float() for x in values))
            def check():
                a,b=baseline(),candidate();torch.npu.synchronize()
                aa={n:t.cpu() for n,t in zip(NAMES,a)};bb={n:t.cpu() for n,t in zip(NAMES,b)}
                exact=all(torch.equal(aa[n].view(torch.uint8),bb[n].view(torch.uint8)) for n in NAMES)
                assert exact,'baseline/candidate should be byte-identical for fixed checkpoint inputs'
                numbers=metrics(bb,expected);assert acceptable(numbers,1e-4 if dtype==torch.float32 else 5e-3)
                return dict(byte_identical=True,independent_B=numbers)
            row=dict(B=1,T=T,H=8,HV=8,dtype=str(dtype),before=check(),rounds=[])
            for round_index in range(3):
                triplet=dict(index=round_index+1,baseline_before=sample(baseline,args.warmup,args.repeat),
                             candidate=sample(candidate,args.warmup,args.repeat),baseline_after=sample(baseline,args.warmup,args.repeat))
                row['rounds'].append(triplet)
                print('ROUND',T,str(dtype),json.dumps(triplet),flush=True)
            row['after']=check();report['cases'].append(row)
            out.write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=True;out.write_text(json.dumps(report,indent=2)+'\n');print('MEASURE_PASS',flush=True)

if __name__=='__main__':main()
