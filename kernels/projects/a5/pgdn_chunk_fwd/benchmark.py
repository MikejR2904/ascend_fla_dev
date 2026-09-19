"""Native PGDN verification; run each block dimension in its own process.

Inputs and CPU references are generated at run time. Hardware selection and the
shared lock belong to ignored external configuration, never this source file.
Timing includes Python dispatch, validation predicates and synchronization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
NAMES = ('q', 'k', 'v', 'g_atk', 'g', 'beta_atk', 'beta')
NAIVE_SHA256 = '3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metric(got, expected):
    actual = got.detach().cpu().float()
    delta = actual - expected.detach().cpu().float()
    norm = expected.float().norm().item()
    return dict(max_abs=delta.abs().max().item(), relative_l2=delta.norm().item()/norm if norm else (0. if not delta.any() else float('inf')),
                finite=bool(torch.isfinite(actual).all()))


def check(got, expected, *, bf16=False):
    stats = metric(got, expected)
    assert stats['finite'] and stats['relative_l2'] <= (5e-3 if bf16 else 1e-4), stats
    torch.testing.assert_close(got.detach().cpu().float(), expected.detach().cpu().float(), atol=2e-5, rtol=1e-2 if bf16 else 2e-4)
    return stats


def timing(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns()-start)/1000)
    return dict(median_us=statistics.median(samples), min_us=min(samples), max_us=max(samples), samples_us=samples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block-dim', type=int, choices=(1,2), default=2)
    p.add_argument('--case', action='append', help='Repeat to select cases; default: all cases for this block dimension.')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeat', type=int, default=50)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--torch-oracle', action='store_true',
                   help='Verify the pinned recurrence on the selected NPU without timing it.')
    p.add_argument('--profile-torch-oracle', action='store_true')
    p.add_argument('--fla-naive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError('nonnegative warmup and positive repeat required')
    assert hashlib.sha256(args.fla_naive.read_bytes()).hexdigest() == NAIVE_SHA256, 'FLA naive pin mismatch'
    import torch_npu
    import ascriptor
    # The optional native composition oracle uses FP32 matmul, without HF32.
    # These are process-local torch_npu settings; no machine configuration.
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    from ascend_fla.ops.pgdn_chunk_fwd import _compiled, _pipeline, chunk_pgdn, prepare
    refs = load('pgdn_native_ref', ROOT/'ref/reference.py')
    oracle = load('pgdn_native_naive', args.fla_naive).naive_recurrent_precond_gated_delta_rule
    contract = json.loads((ROOT/'contract.json').read_text())
    selected = set(args.case or ('all',))
    eligible = [c for c in contract['cases'] if c['block_dim'] == args.block_dim]
    unknown = selected - {'all'} - {c['id'] for c in eligible}
    if unknown:
        raise ValueError(f'cases do not match the selected block dimension: {sorted(unknown)}')
    cases = [c for c in eligible if 'all' in selected or c['id'] in selected]
    if not cases:
        raise ValueError('no matching case for selected block dimension')
    torch.set_num_threads(1)
    prepare(block_dim=args.block_dim)
    compiled = dict(zip((e.name for e in _pipeline().entries()), _compiled(args.block_dim)))
    report = dict(schema='pgdn-forward-validation/1', scope='A5 native in-process CCE, FP32 internals, no CUDA/Triton or checkpoint execution',
                  block_dim=args.block_dim, versions=dict(torch=torch.__version__, torch_npu=torch_npu.__version__, ascriptor=ascriptor.__version__),
                  warmup=args.warmup, repeat=args.repeat,
                  precision=dict(matmul_allow_hf32=torch.npu.matmul.allow_hf32,
                                 conv_allow_hf32=torch.npu.conv.allow_hf32),
                  source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ROOT.rglob('*.py'))},
                  wrapper_sha256=hashlib.sha256((ROOT.parents[3]/'ascend_fla/ops/pgdn_chunk_fwd.py').read_bytes()).hexdigest(),
                  oracle_sha256=NAIVE_SHA256, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    for case in cases:
        cpu = refs.make_inputs(case)
        independent = refs.reference_stages(cpu)
        authority = dict(zip(refs.OUTPUTS, oracle(*(cpu[n] for n in NAMES), output_final_state=True)))
        npu = {n: x.to('npu') for n, x in cpu.items()}
        row = dict(case=case['id'], seed=case['seed'], parameters=case['parameters'], stages={}, independent_leaves={}, oracles={},
                   input_sha256={n:digest(x) for n,x in cpu.items()})
        report['cases'].append(row)
        calls = []
        def launch(entry, sources, outputs, scalars):
            print(json.dumps(dict(event='launch', case=case['id'], kernel=entry.name)), flush=True)
            op = compiled[entry.name]
            scalars = {n:scalars[n] for n in op.scalar_names}
            for x in outputs.values():
                x.fill_(float('nan'))
            op(sources, scalars, outputs)
            calls.append((entry.name, op, sources, outputs, scalars))
            return outputs
        try:
            got = _pipeline().run(npu, launch)
            torch.npu.synchronize()
            for name, expected in independent.items():
                row['stages'][name] = metric(got[name], expected); save()
                check(got[name], expected)
            upstream = dict(cpu, **independent)
            for _, op, sources, outputs, scalars in calls:
                leaf_inputs = {n:upstream[n].contiguous().to('npu') for n in sources}
                leaf_outputs = {n:torch.full_like(x, float('nan')) for n,x in outputs.items()}
                op(leaf_inputs, scalars, leaf_outputs)
                torch.npu.synchronize()
                for name, actual in leaf_outputs.items():
                    row['independent_leaves'][name] = metric(actual, independent[name]); save()
                    check(actual, independent[name])
            for label, expected in (('A_pinned_cpu_recurrence', authority), ('B_independent_cpu_block_solve', independent)):
                row['oracles'][label] = {n:check(got[n], expected[n]) for n in refs.OUTPUTS}
            row['stage_sha256'] = {n:digest(x) for n,x in got.items()}
            for dtype in (torch.float32, torch.bfloat16):
                public_inputs = {n:npu[n].to(dtype) if n in ('q','k','v') else npu[n] for n in NAMES}
                cast_cpu = {n:cpu[n].to(dtype).float() if n in ('q','k','v') else cpu[n] for n in cpu}
                a = dict(zip(refs.OUTPUTS, oracle(*(cast_cpu[n] for n in NAMES), output_final_state=True)))
                b = refs.reference(cast_cpu)
                result = chunk_pgdn(*(public_inputs[n] for n in NAMES), block_dim=args.block_dim, output_final_state=True)
                row[str(dtype)] = {label:{n:check(x, expected[n], bf16=(n=='o' and dtype==torch.bfloat16)) for n,x in zip(refs.OUTPUTS,result)}
                                   for label,expected in (('A',a),('B',b))}
                row[str(dtype)+'_sha256'] = {n:digest(x) for n,x in zip(refs.OUTPUTS,result)}
            for name in cpu:
                assert digest(npu[name]) == row['input_sha256'][name], f'input mutated: {name}'
            if args.profile:
                row['stage_timing'] = {name:timing(lambda op=op,s=sources,o=outputs,a=scalars:op(s,a,o),args.warmup,args.repeat)
                                       for name,op,sources,outputs,scalars in calls}
                row['public_timing'] = timing(lambda:chunk_pgdn(*(npu[n] for n in NAMES),block_dim=args.block_dim,output_final_state=True),args.warmup,args.repeat)
            if args.torch_oracle or args.profile_torch_oracle:
                # Same tensor inputs and selected device, actual PyTorch NPU
                # composition; this is not CUDA/Triton or a fused baseline.
                npu_oracle = oracle(*(npu[n] for n in NAMES), output_final_state=True)
                row['torch_npu_composition'] = {n:check(x,authority[n]) for n,x in zip(refs.OUTPUTS,npu_oracle)}
                if args.profile_torch_oracle:
                    row['torch_npu_composition_timing'] = timing(lambda:oracle(*(npu[n] for n in NAMES),output_final_state=True),args.warmup,args.repeat)
            row['passed'] = True
            save()
            print(json.dumps(dict(case=case['id'],oracles=row['oracles'],public_median_us=row.get('public_timing',{}).get('median_us'))), flush=True)
        except BaseException as error:
            row['passed'] = False
            row['failure'] = str(error)
            save()
            raise
    report['passed'] = True
    save()


if __name__ == '__main__':
    main()
