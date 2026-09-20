"""Synchronized decode API timings, including the actual Torch NPU baseline."""
import importlib
import statistics
import time
from unittest.mock import patch

import torch

from ascend_fla.reference.kda import kda_recurrent_ref


def measure(api, research, bd, original_fp32, to_device, fla):
    rows = []
    warmup, repeat = 5, 20

    def timed(fn, count):
        samples, enqueue = [], []
        for _ in range(count):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            fn()
            returned = time.perf_counter_ns()
            torch.npu.synchronize()
            end = time.perf_counter_ns()
            samples.append((end-start)/1000)
            enqueue.append((returned-start)/1000)
        return dict(synchronized_us=samples, median_us=statistics.median(samples),
                    enqueue_wall_us=enqueue, median_enqueue_us=statistics.median(enqueue))

    shapes = [dict(B=1,T=t,H=32,G=1,state=True) for t in (1,2,16)]
    shapes += [dict(B=2,T=16,H=4,G=8,state=True)]
    for p in shapes:
        data = research.make_inputs(p, seed=9606)
        fp = {n:(x.float() if n in ('q','k','v') else x) for n,x in data.items()}
        bfdev, fpdev = to_device(data), to_device(fp)
        candidate = lambda: api.fused_recurrent_kda(**bfdev, block_dim=bd)
        fp32_new = lambda: api.fused_recurrent_kda(**fpdev, block_dim=bd)
        baseline = lambda: original_fp32(fpdev, return_cpu=False)
        torch_baseline = lambda: kda_recurrent_ref(**bfdev)
        for fn in (baseline, candidate, fp32_new, torch_baseline):
            for _ in range(warmup):
                fn()
        torch.npu.synchronize()
        refs = research.references(data, fla)
        results = {}
        for label, fn in (('candidate',candidate),('torch_npu',torch_baseline)):
            got = tuple(x.detach().cpu().contiguous() for x in fn())
            results[label] = research.compare(got, refs)
            assert all(r['passed'] for r in results[label].values()), results

        # A warmed default public call must not reread source or compute hashes.
        compiler = importlib.import_module('ascend_fla.runtime.compile')
        with patch.object(compiler, '_signature', side_effect=AssertionError('warm call rebuilt signature')):
            candidate(); fp32_new()
        torch.npu.synchronize()
        rounds = []
        for index in range(3):
            row = {label:timed(fn,repeat) for label,fn in
                   (('original_fp32_before',baseline),('native_bf16',candidate),('original_fp32_after',baseline))}
            average = (row['original_fp32_before']['median_us']+row['original_fp32_after']['median_us'])/2
            row['bf16_speedup_vs_original_fp32'] = average/row['native_bf16']['median_us']
            row['native_fp32'] = timed(fp32_new, repeat)
            rounds.append(row)
            print('ORIGINAL_FP32_SANDWICH',p,index+1,row,flush=True)
        torch_rounds = []
        for index in range(3):
            row = {label:timed(fn,repeat) for label,fn in
                   (('torch_npu_before',torch_baseline),('native_bf16',candidate),('torch_npu_after',torch_baseline))}
            average = (row['torch_npu_before']['median_us']+row['torch_npu_after']['median_us'])/2
            row['bf16_speedup_vs_torch_npu'] = average/row['native_bf16']['median_us']
            torch_rounds.append(row)
            print('TORCH_NPU_SANDWICH',p,index+1,row,flush=True)
        rows.append(dict(shape=p, block_dim=bd, correctness=results, original_fp32_rounds=rounds,
                         torch_npu_rounds=torch_rounds, warm_signature_cache_check=True))

    first, last = rows[0], rows[2]
    fixed = []
    for index in range(3):
        result = {}
        for label in ('original_fp32_before','native_bf16','original_fp32_after','native_fp32'):
            one = first['original_fp32_rounds'][index][label]
            sixteen = last['original_fp32_rounds'][index][label]
            slope = (sixteen['median_us']-one['median_us'])/15
            result[label] = dict(one_token_us=one['median_us'], one_token_enqueue_us=one['median_enqueue_us'],
                                 marginal_us_per_token=slope, estimated_fixed_us=one['median_us']-slope)
        fixed.append(result)
    return dict(passed=True, warmup=warmup, repeat=repeat, rounds=3, cases=rows,
                baseline='Original FP32 wrapper preparation and unchanged FP32 custom kernel',
                additional_baseline='kda_recurrent_ref on actual Torch NPU, FP32 recurrence with BF16 GM inputs/output',
                timing='Synchronized API wall time in microseconds, preallocated inputs, output allocation included',
                fixed_cost=dict(method='T1/T16 linear intercept; estimate includes bridge and device fixed costs',
                                enqueue_note='CPU wall time until API returns; may include runtime waits, not pure CPU instruction cost',
                                rounds=fixed))
