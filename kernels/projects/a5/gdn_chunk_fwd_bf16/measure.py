"""Same-card original-FP32 / native-BF16 / original-FP32 sandwiches."""
import argparse,hashlib,importlib.util,json,statistics,time
from pathlib import Path
import torch
from ref.reference import make_inputs,reference,metric,budget
from ascend_fla.ops import gdn_chunk_fwd as candidate


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--block-dim',type=int,choices=(1,2),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--baseline-sha',required=True);args=p.parse_args()
    import torch_npu
    torch.set_num_threads(1);assert torch.npu.device_count()==1;torch.npu.set_device(0)
    path=args.baseline_root/'ascend_fla/ops/gdn_chunk_fwd.py'
    actual_sha=hashlib.sha256(path.read_bytes()).hexdigest();assert actual_sha==args.baseline_sha
    spec=importlib.util.spec_from_file_location('ascend_fla.ops._bf01_original_public',path)
    baseline=importlib.util.module_from_spec(spec);spec.loader.exec_module(baseline)
    # Both named implementations are compiled before the first custom-op resolution.
    baseline.prepare(block_dim=args.block_dim);candidate.prepare(block_dim=args.block_dim)
    report=dict(stage='native_same_card_timing',block_dim=args.block_dim,warmup=10,repeat=50,rounds=3,
                baseline='Unchanged original FP32 public chunk_gdn, including grouped input preparation/allocations, on BF16-rounded FP32 input values',
                candidate='New complete public chunk_gdn on actual BF16 q/k/v storage; FP32 state',
                synchronization='Before and after every invocation; perf_counter_ns; milliseconds',
                baseline_source_sha256=actual_sha,candidate_source_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),cases=[])
    for length in (1024,4096):
        case=dict(id=f'T{length}',seed=173000+length,B=1,T=length,H=8,HV=8,dtype='bfloat16')
        cpu=make_inputs(case);target=reference(cpu)
        bf16={n:x.npu() for n,x in cpu.items()}
        fp32={n:x.float().npu() for n,x in cpu.items()}
        before={label:{n:digest(x) for n,x in values.items()} for label,values in [('baseline',fp32),('candidate',bf16)]}
        def call(which):
            api,values=(baseline,fp32) if which=='baseline' else (candidate,bf16)
            return api.chunk_gdn(**values,output_final_state=True,block_dim=args.block_dim)
        def verify():
            left=call('baseline');right=call('candidate');torch.npu.synchronize()
            left={n:x.cpu() for n,x in zip(('o','final_state'),left)}
            right={n:x.cpu() for n,x in zip(('o','final_state'),right)}
            numbers={}
            for label,values,dtype in [('baseline',left,torch.float32),('candidate',right,torch.bfloat16)]:
                numbers[label]={n:metric(x,target[n]) for n,x in values.items()}
                assert all(x['finite'] and x['relative_l2']<=budget(target[n],dtype) for n,x in numbers[label].items())
            assert digest(left['o'].bfloat16())==digest(right['o'])
            assert digest(left['final_state'])==digest(right['final_state'])
            return dict(independent_B=numbers,rounded_output_byte_equal=True,state_byte_equal=True)
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
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['passed']=True;args.output.write_text(json.dumps(report,indent=2)+'\n');print('TIMING_PASS',flush=True)


if __name__=='__main__':main()
