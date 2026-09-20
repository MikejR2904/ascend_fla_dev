"""Same-card complete original-FP32 / native-BF16 / original-FP32 backward."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from benchmark import load_baseline,digest
from ref.bf16 import make_inputs,reference,comparison,acceptable,NAMES
from ascend_fla.ops import gdn_chunk_bwd as candidate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--block-dim',type=int,choices=(1,2),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--baseline-sha',required=True)
    args=p.parse_args()
    import torch_npu
    torch.set_num_threads(1)
    assert torch.npu.device_count()==1
    torch.npu.set_device(0)
    baseline,identity=load_baseline(args.baseline_root)
    assert identity['wrapper_sha256']==args.baseline_sha
    candidate.prepare(block_dim=args.block_dim)
    report=dict(stage='native_same_card_timing',block_dim=args.block_dim,warmup=10,repeat=50,rounds=3,
                baseline='Unchanged complete original FP32 public chunk_gdn_bwd including checkpoint creation, replay, reverse, group reduction, allocations and dispatch; BF16-rounded FP32 operands',
                candidate='Complete new public chunk_gdn_bwd on actual BF16 q/k/v/do storage; both cotangents present',
                synchronization='Before and after every invocation; perf_counter_ns; milliseconds',
                baseline_identity=identity,candidate_source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for length in (1024,4096):
        case=dict(id=f'T{length}',seed=173000+length,parameters=dict(B=1,T=length,H=8,HV=8,mode='both'))
        cpu=make_inputs(case);target=reference(cpu)
        bf16={n:x.npu() for n,x in cpu.items()}
        fp32={n:x.float().npu() for n,x in cpu.items()}
        before={label:{n:digest(x) for n,x in values.items()} for label,values in [('baseline',fp32),('candidate',bf16)]}
        def call(which):
            api,values=(baseline,fp32) if which=='baseline' else (candidate,bf16)
            return api.chunk_gdn_bwd(**values,block_dim=args.block_dim)
        def verify():
            left=call('baseline');right=call('candidate');torch.npu.synchronize()
            numbers={}
            for label,values,dtype in [('baseline',left,torch.float32),('candidate',right,torch.bfloat16)]:
                numbers[label]=comparison({n:x.cpu() for n,x in zip(NAMES,values)},target,dtype)
                assert acceptable(numbers[label]),numbers[label]
            return dict(independent_B=numbers)
        row=dict(case=case,checks_before=verify(),sandwiches=[])
        for iteration in range(3):
            segments=[]
            for name in ('baseline','candidate','baseline'):
                for _ in range(10):
                    torch.npu.synchronize();call(name);torch.npu.synchronize()
                samples=[]
                for _ in range(50):
                    torch.npu.synchronize();started=time.perf_counter_ns();call(name);torch.npu.synchronize()
                    samples.append((time.perf_counter_ns()-started)/1e6)
                segments.append(dict(name=name,samples_ms=samples,median_ms=statistics.median(samples)))
            row['sandwiches'].append(dict(round=iteration,segments=segments))
            print('SANDWICH',case['id'],iteration,json.dumps(segments),flush=True)
        row['checks_after']=verify()
        for label,values in [('baseline',fp32),('candidate',bf16)]:
            assert all(digest(x)==before[label][n] for n,x in values.items())
        row['inputs_unchanged']=True;report['cases'].append(row)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['passed']=True
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('TIMING_PASS',flush=True)


if __name__=='__main__':main()
