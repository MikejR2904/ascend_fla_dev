"""Same-card synchronized three-round sandwiches including raw preparation."""
import argparse
import statistics
import time
from pathlib import Path
import torch
from ascend_fla.ops.kda import chunk,autograd
from ascend_fla.reference.kda import kda_chunk_vectorized
from ascend_fla.runtime.compile import CompiledKernel
from native_context import Context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();ctx=Context(args.output,4,4,__file__);check=ctx.check
    flags=dict(use_qk_l2norm_in_kernel=True,use_gate_in_kernel=True,use_beta_sigmoid_in_kernel=True)
    options=dict(block_dim=4,layout_device='npu');rows=[]
    prep_signatures={op.signature for op in chunk._prep_runtime().prepare('a5',4).values()}
    layout_signatures={op.signature for op in chunk._layout_runtime().prepare('a5',4).values()}
    def timed(fn):
        torch.npu.synchronize();start=time.perf_counter();result=fn();torch.npu.synchronize()
        return dict(milliseconds=(time.perf_counter()-start)*1000),result
    def sandwich(old,new):
        old();new();torch.npu.synchronize();rounds=[]
        for index in range(3):
            samples=[]
            for name,fn in (('baseline_before',old),('candidate',new),('baseline_after',old)):
                sample,value=timed(fn);sample['phase']=name;samples.append(sample);del value
            rounds.append(dict(round=index+1,samples=samples))
        old_ms=[v['milliseconds'] for r in rounds for v in (r['samples'][0],r['samples'][2])]
        new_ms=[r['samples'][1]['milliseconds'] for r in rounds]
        return dict(rounds=rounds,baseline_median_ms=statistics.median(old_ms),candidate_median_ms=statistics.median(new_ms),candidate_over_baseline=statistics.median(new_ms)/statistics.median(old_ms))
    def profile(fn):
        events=[];original=CompiledKernel.__call__
        def call(op,inputs,scalars,outputs):
            begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            begin.record();clock=time.perf_counter();result=original(op,inputs,scalars,outputs);dispatch=(time.perf_counter()-clock)*1000;end.record()
            events.append((begin,end,dict(operator=op.op.op_name,signature=op.signature,host_dispatch_ms=dispatch,
                family='prep' if op.signature in prep_signatures else 'layout' if op.signature in layout_signatures else 'attention',
                input_bytes=sum(t.numel()*t.element_size() for t in inputs.values()),output_bytes=sum(t.numel()*t.element_size() for t in outputs.values()))))
            return result
        CompiledKernel.__call__=call
        try:
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            torch.npu.synchronize();clock=time.perf_counter();start.record();value=fn();end.record();torch.npu.synchronize();wall=(time.perf_counter()-clock)*1000
        finally:CompiledKernel.__call__=original
        costs=[]
        for a,b,row in events:row['device_event_ms']=a.elapsed_time(b);costs.append(row)
        return dict(custom_launches=len(costs),events=costs,instrumented_wall_ms=wall,whole_stream_event_ms=start.elapsed_time(end),
            custom_event_sum_ms=sum(r['device_event_ms'] for r in costs),custom_host_dispatch_sum_ms=sum(r['host_dispatch_ms'] for r in costs),
            interpretation='Separate event instrumentation; event intervals include stream idle gaps while host dispatches. These are not clean kernel-only timings, and sums are not an additive decomposition of clean wall time.')
    with torch.no_grad():
        for tokens in (1024,4096):
            base=ctx.real.make_inputs(B=1,H=32,HV=32,C=tokens//64,span=46,want_grads=True)
            gen=torch.Generator().manual_seed(7007)
            x={n:torch.randn(1,tokens,32,128,generator=gen).bfloat16() for n in ('q','k')}
            x.update(g=(torch.randn(1,tokens,32,128,generator=gen)*.1).bfloat16(),beta=torch.randn(1,tokens,32,generator=gen).bfloat16(),
                A_log=torch.linspace(-.2,.2,32),dt_bias=torch.linspace(-.3,-.1,32*128),v=base['v'],h0=base['h0'])
            dev={n:t.npu() for n,t in x.items()}
            prep_kwargs=dict(flags,A_log=dev['A_log'],dt_bias=dev['dt_bias'])
            data={n:dev[n] for n in ('q','k','v','g','beta')};data.update(initial_state=dev['h0'],A_log=dev['A_log'],dt_bias=dev['dt_bias'],**flags)
            def old_plain():return ctx.old_auto.chunk_kda(**data,output_final_state=True,**options)
            def new_plain():return autograd.chunk_kda(**data,output_final_state=True,**options)
            def old_cached():
                q,k,g,beta=ctx.old._prepare_inputs(*(dev[n] for n in ('q','k','g','beta')),**prep_kwargs)
                return ctx.old.chunk_kda_fwd_with_caches(q,k,dev['v'],g,beta,initial_state=dev['h0'],**options)
            def new_cached():
                q,k,g,beta=chunk._prepare_kernel_inputs(*(dev[n] for n in ('q','k','g','beta')),**prep_kwargs,block_dim=4)
                return chunk.chunk_kda_fwd_with_caches(q,k,dev['v'],g,beta,initial_state=dev['h0'],**options)
            def torch_path(function):
                q,k,g,beta=ctx.old._prepare_inputs(*(dev[n] for n in ('q','k','g','beta')),**prep_kwargs)
                return function(q.float(),k.float(),dev['v'].float(),g,beta,initial_state=dev['h0'],output_final_state=True)
            paths=(('public_forward',old_plain,new_plain),('cached_forward',old_cached,new_cached),
                   ('torch_npu_vectorized',lambda:torch_path(kda_chunk_vectorized),new_plain),
                   ('torch_npu_recurrent',lambda:torch_path(ctx.real.kda_recurrent_ref),new_plain))
            for scope,old,new in paths:
                print('PERF_START',tokens,scope,flush=True)
                measured=sandwich(old,new)
                before=check.cpu(old());after=check.cpu(new())
                comparison={n:check.metrics(after[i],before[i],.05) for i,n in enumerate(('o','final_state'))}
                row=dict(tokens=tokens,scope=scope,shape=dict(B=1,H=32,HV=32,K=128),raw_dtype='bf16',parameter_dtype='f32',flags=flags,
                    timing='Clean synchronized NPU public-call wall milliseconds; raw preparation included, input generation/transfers and compilation excluded.',
                    warmup_per_path=1,repetitions_per_phase=1,rounds_count=3,speed_gate=None,**measured,
                    measured_output_comparison=comparison,input_sha256={n:check.digest(t) for n,t in x.items()},input_unchanged={n:check.digest(dev[n])==check.digest(t) for n,t in x.items()})
                if not scope.startswith('torch_'):
                    row['attribution']={label:profile(fn) for label,fn in (('baseline',old),('candidate',new))}
                    row['extra_custom_launches']=row['attribution']['candidate']['custom_launches']-row['attribution']['baseline']['custom_launches']
                row['finite_comparison_passed']=all(m['passed'] for m in comparison.values())
                rows.append(row);ctx.write('measurements',dict(complete=False,rows=rows))
                if not row['finite_comparison_passed'] or not all(row['input_unchanged'].values()):raise AssertionError('measured path failed comparison: '+scope)
                print('PERF_DONE',tokens,scope,measured['candidate_over_baseline'],flush=True)
    ctx.write('measurements',dict(complete=True,rows=rows));ctx.write('summary',dict(complete=True,passed=True,measurements=len(rows),speed_gate=None))

if __name__=='__main__':main()
