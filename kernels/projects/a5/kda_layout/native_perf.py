"""Unfiltered synchronized three-round measurements of the actual public paths."""
import statistics
import time

import torch

import native_support as check


def measure(chunk, before, backward, before_backward, real, bd):
    from ascend_fla.runtime.compile import CompiledKernel
    layout_signatures = {op.signature for op in chunk._layout_runtime().prepare('a5',bd).values()}
    rows = []
    for tokens in (1024,4096):
        x = real.make_inputs(B=1,H=32,HV=32,C=tokens//64,span=46,want_grads=True)
        dev = {n:t.npu() for n,t in x.items()}
        data = {n:dev[n] for n in ('q','k','v','g','beta')}
        data['initial_state']=dev['h0']
        options=dict(block_dim=bd,layout_device='npu')
        caches=chunk.chunk_kda_fwd_with_caches(**data,**options)[2]
        bwd_data={n:dev[n] for n in ('q','k','v','do','dht')}
        bwd_data.update(beta=dev['beta'].bfloat16(),caches=caches)
        fp32={n:dev[n].float() for n in ('q','k','v','g','beta','h0')}
        paths={
            'plain_forward': (lambda:before.chunk_kda_fwd(**data,output_final_state=True,**options),
                              lambda:chunk.chunk_kda_fwd(**data,output_final_state=True,**options)),
            'cached_forward': (lambda:before.chunk_kda_fwd_with_caches(**data,**options),
                               lambda:chunk.chunk_kda_fwd_with_caches(**data,**options)),
            'backward_existing_caches': (lambda:before_backward.chunk_kda_bwd(**bwd_data,**options),
                                         lambda:backward.chunk_kda_bwd(**bwd_data,**options)),
        }
        def timed(fn):
            torch.npu.synchronize()
            start=time.perf_counter()
            value=fn()
            torch.npu.synchronize()
            return (time.perf_counter()-start)*1000, value
        for scope,(old,new) in paths.items():
            old();new();torch.npu.synchronize()
            counts=[]
            for fn in (old,new):
                with check.instrument(poison=False,audit=False) as (_,launches):fn()
                torch.npu.synchronize()
                counts.append(dict(total=len(launches),layout=sum(r['signature'] in layout_signatures for r in launches),
                                   operators=[r['operator'] for r in launches]))
            # A separate serialized diagnostic quantifies the layout launches;
            # synchronization added here is excluded from the sandwiches below.
            layout_cost=[]
            original=CompiledKernel.__call__
            def profile_layout(op,inputs,scalars,outputs):
                if op.signature not in layout_signatures:
                    return original(op,inputs,scalars,outputs)
                elapsed,value=timed(lambda:original(op,inputs,scalars,outputs))
                layout_cost.append(dict(operator=op.op.op_name,milliseconds=elapsed,
                                        output_elements=sum(t.numel() for t in outputs.values())))
                return value
            CompiledKernel.__call__=profile_layout
            try:new()
            finally:CompiledKernel.__call__=original
            rounds=[]
            for index in range(3):
                samples=[]
                for phase,fn in (('baseline_before',old),('candidate',new),('baseline_after',old)):
                    elapsed,value=timed(fn)
                    samples.append(dict(phase=phase,milliseconds=elapsed))
                    del value
                rounds.append(dict(round=index+1,samples=samples))
            candidates=[r['samples'][1]['milliseconds'] for r in rounds]
            baselines=[s['milliseconds'] for r in rounds for s in (r['samples'][0],r['samples'][2])]
            rows.append(dict(tokens=tokens,scope=scope,rounds=rounds,launches=dict(baseline=counts[0],candidate=counts[1]),
                             serialized_layout_cost_diagnostic=layout_cost,
                             extra_launches=counts[1]['total']-counts[0]['total'],
                             candidate_median_ms=statistics.median(candidates),baseline_median_ms=statistics.median(baselines),
                             candidate_over_baseline=statistics.median(candidates)/statistics.median(baselines)))
        # CPU reference timing is never used as a performance baseline.
        def torch_npu():
            return real.kda_recurrent_ref(*(fp32[n] for n in ('q','k','v','g','beta')),
                                         initial_state=fp32['h0'],output_final_state=True)
        torch_npu();torch.npu.synchronize()
        rounds=[]
        for index in range(3):
            samples=[]
            for phase,fn in (('torch_npu_before',torch_npu),('candidate',paths['plain_forward'][1]),('torch_npu_after',torch_npu)):
                elapsed,value=timed(fn)
                samples.append(dict(phase=phase,milliseconds=elapsed))
                del value
            rounds.append(dict(round=index+1,samples=samples))
        rows.append(dict(tokens=tokens,scope='actual_torch_npu_fp32_reference',rounds=rounds))
    return dict(passed=True,timing='synchronized NPU public-call wall milliseconds; input preparation excluded',
                warmup=1,rounds=3,speed_gate=None,measurements=rows)
