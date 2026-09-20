"""Three same-card original-FP32/native-BF16/original-FP32 sandwiches."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from ascend_fla.ops import pgdn_chunk_fwd as candidate
from benchmark import digest, load_baseline, check
from ref import oracle
from ref.reference import OUTPUTS, make_inputs, reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--block-dim', required=True, type=int, choices=(1, 2))
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--baseline-sha', required=True)
    args = parser.parse_args()
    import torch_npu
    torch.set_num_threads(1)
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    baseline, identity = load_baseline(args.baseline_root)
    assert identity['wrapper_sha256'] == args.baseline_sha
    candidate.prepare(block_dim=args.block_dim)
    report = dict(stage='native_same_card_timing', block_dim=args.block_dim, warmup=10, repeat=50,
                  rounds=3, baseline=identity,
                  baseline_description='Pristine main FP32 public entry on BF16-rounded FP32 input values',
                  candidate_description='Complete native BF16 public PGDN entry; FP32 main/ATK states',
                  scope='All six launches, allocations and existing read-only validation/synchronizations',
                  synchronization='NPU synchronize before and after every call, perf_counter_ns, milliseconds',
                  candidate_sha256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(), cases=[])
    for length in (1024, 4096):
        case = dict(id=f'T{length}', seed=173000+length, dtype='bfloat16',
                    parameters=dict(B=1, T=length, H=8, HV=8, N=length//64))
        cpu = make_inputs(case)
        a, b = oracle.reference(cpu), reference(cpu)
        bf16 = {n: x.npu() for n, x in cpu.items() if n != 'initial_state'}
        fp32 = {n: x.float().npu() for n, x in cpu.items() if n != 'initial_state'}
        values = dict(baseline=fp32, candidate=bf16)
        before = {label: {n: digest(x) for n, x in inp.items()} for label, inp in values.items()}
        def call(which):
            api = baseline if which == 'baseline' else candidate
            return api.chunk_pgdn(**values[which], output_final_state=True, block_dim=args.block_dim)
        def verify():
            left, right = call('baseline'), call('candidate')
            torch.npu.synchronize()
            left = {n: x.cpu() for n, x in zip(OUTPUTS, left)}
            right = {n: x.cpu() for n, x in zip(OUTPUTS, right)}
            results = {}
            for label, result, dtype in [('baseline', left, torch.float32), ('candidate', right, torch.bfloat16)]:
                results[label] = dict(A=check(result, a, dtype, public_outputs=True),
                                      B=check(result, b, dtype, public_outputs=True))
            assert digest(left['o'].bfloat16()) == digest(right['o'])
            for name in ('final_state', 'final_A_state'):
                assert digest(left[name]) == digest(right[name])
            return dict(oracles=results, rounded_output_byte_equal=True, states_byte_equal=True)
        row = dict(case=case, checks_before=verify(), sandwiches=[])
        for iteration in range(3):
            segments = []
            for name in ('baseline', 'candidate', 'baseline'):
                for _ in range(10):
                    torch.npu.synchronize();call(name);torch.npu.synchronize()
                samples = []
                for _ in range(50):
                    torch.npu.synchronize();started = time.perf_counter_ns()
                    call(name);torch.npu.synchronize()
                    samples.append((time.perf_counter_ns()-started)/1e6)
                segments.append(dict(name=name, samples_ms=samples, median_ms=statistics.median(samples)))
            row['sandwiches'].append(dict(round=iteration, segments=segments))
            print('SANDWICH', case['id'], iteration, json.dumps(segments), flush=True)
        row['checks_after'] = verify()
        assert all(digest(x) == before[label][n] for label, inp in values.items() for n, x in inp.items())
        row['inputs_unchanged'] = True
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    report['passed'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('TIMING_PASS', flush=True)


if __name__ == '__main__':
    main()
