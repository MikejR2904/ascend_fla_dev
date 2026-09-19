"""Three synchronized GDN/PGDN/GDN rounds on one externally locked NPU.

Run after full native correctness acceptance. The GDN baseline uses the same
five chunk kernels with equal read/write keys, without ATK. It is not an
equivalent PGDN algorithm. NPU normalization and output casting are timed.
GDN returns its final main state; public PGDN returns both final states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from benchmark import ROOT, NAMES, NAIVE_SHA256, check, digest, load, timing

GDN_NAIVE_SHA256 = 'd1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb'


def gdn_baseline(inputs, pgdn, compiled, dtype):
    """GDN equations through PGDN-owned stages; no ATK or public API change."""
    q = torch.nn.functional.normalize(inputs['q'].float(), dim=-1)
    k = torch.nn.functional.normalize(inputs['k'].float(), dim=-1)
    b, t, h, _ = q.shape
    hv = inputs['v'].shape[2]
    n = t//64
    scalars = dict(B=b, T=t, H=h, HV=hv, N=n)
    values = dict(q_norm=q, k_read=k, k_write=k, v=inputs['v'].float(),
                  g=inputs['g'], beta=inputs['beta'],
                  initial_state=torch.zeros(b, hv, 128, 128, device=q.device))
    shapes = {name: (b, n, hv, 64, 128) for name in ('qn', 'kw', 'gc', 'bk', 'wv', 'u', 'wy', 'delta')}
    shapes.update(lower=(b,n,hv,64,64), score=(b,n,hv,64,64), states=(b,n,hv,128,128),
                  o=(b,t,hv,128), final_state=(b,hv,128,128))
    graph = pgdn._pipeline().GRAPH[1:]
    for index, (entry, (_, names, outputs)) in enumerate(zip(pgdn._pipeline().entries()[1:], graph)):
        op = compiled[entry.name]
        fresh = {name: torch.empty(shapes[name], dtype=torch.float32, device=q.device) for name in outputs}
        op({name: values[name] for name in names}, {name: scalars[name] for name in op.scalar_names}, fresh)
        values.update(fresh)
        live = {'o', 'final_state'}
        for _, future_inputs, _ in graph[index+1:]:
            live.update(future_inputs)
        for name in tuple(values):
            if name not in live:
                del values[name]
    return values['o'].to(dtype), values['final_state']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--block-dim', type=int, choices=(1, 2), default=2)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeat', type=int, default=50)
    p.add_argument('--fla-naive', type=Path, required=True)
    p.add_argument('--fla-gdn-naive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError('nonnegative warmup and positive repeat required')
    for path, expected in ((args.fla_naive, NAIVE_SHA256), (args.fla_gdn_naive, GDN_NAIVE_SHA256)):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'oracle pin mismatch'
    import ascriptor
    import torch_npu
    from ascend_fla.ops import pgdn_chunk_fwd as pgdn
    torch.set_num_threads(1)
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    # CANN resolves the complete vendor search path on the first operator call.
    pgdn.prepare(block_dim=args.block_dim)
    compiled = dict(zip((entry.name for entry in pgdn._pipeline().entries()), pgdn._compiled(args.block_dim)))
    refs = load('pgdn_measure_reference', ROOT/'ref/reference.py')
    pgdn_naive = load('pgdn_measure_naive', args.fla_naive).naive_recurrent_precond_gated_delta_rule
    gdn_naive = load('gdn_measure_naive', args.fla_gdn_naive).naive_recurrent_gated_delta_rule
    repo = ROOT.parents[3]
    sources = list(ROOT.rglob('*.py')) + [repo/'ascend_fla/ops/pgdn_chunk_fwd.py']
    report = dict(schema='pgdn-forward-sandwich/1', stage='native_inprocess',
                  baseline='GDN equations through the five PGDN-owned chunk kernels with equal read/write keys; timed NPU FP32 normalization/output cast; no ATK',
                  candidate='Native public PGDN with ATK, both final states and validation predicates',
                  timing='Synchronized wall microseconds, including Python dispatch; no speed threshold',
                  block_dim=args.block_dim, warmup=args.warmup, repeat=args.repeat, rounds=3,
                  versions=dict(torch=torch.__version__, torch_npu=torch_npu.__version__, ascriptor=ascriptor.__version__),
                  precision=dict(matmul_allow_hf32=torch.npu.matmul.allow_hf32, conv_allow_hf32=torch.npu.conv.allow_hf32),
                  source_sha256={str(f.relative_to(repo)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(sources)},
                  oracle_sha256=dict(pgdn=NAIVE_SHA256, gdn=GDN_NAIVE_SHA256), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    try:
        for length in (4096, 1024):
            parameters = dict(B=1, T=length, H=8, HV=8, N=length//64, gate_scale=1., atk_gate_scale=.03)
            generated = refs.make_inputs(dict(seed=69017, parameters=parameters))
            for dtype in (torch.float32, torch.bfloat16):
                cpu = {n: x.to(dtype).float() if n in ('q', 'k', 'v') else x for n, x in generated.items()}
                npu = {n: cpu[n].to(dtype if n in ('q', 'k', 'v') else torch.float32).to('npu') for n in NAMES}
                hashes = {n: digest(x) for n, x in npu.items()}
                a = dict(zip(refs.OUTPUTS, pgdn_naive(*(cpu[n] for n in NAMES), output_final_state=True)))
                b = refs.reference(cpu)
                gdn_reference = gdn_naive(torch.nn.functional.normalize(cpu['q'], dim=-1),
                                          torch.nn.functional.normalize(cpu['k'], dim=-1), cpu['v'],
                                          cpu['beta'], cpu['g'], output_final_state=True)
                def baseline():
                    return gdn_baseline(npu, pgdn, compiled, dtype)
                def candidate():
                    return pgdn.chunk_pgdn(*(npu[n] for n in NAMES), block_dim=args.block_dim, output_final_state=True)
                def verify():
                    result = candidate()
                    metrics = {label: {n: check(x, expected[n], bf16=n=='o' and dtype==torch.bfloat16)
                                       for n, x in zip(refs.OUTPUTS, result)} for label, expected in (('A', a), ('B', b))}
                    metrics['GDN_pinned'] = {n: check(x, expected, bf16=n=='o' and dtype==torch.bfloat16)
                                             for n, x, expected in zip(('o', 'final_state'), baseline(), gdn_reference)}
                    return metrics
                row = dict(parameters=parameters, seed=69017, dtype=str(dtype), input_sha256=hashes, rounds=[])
                report['cases'].append(row)
                row['before'] = verify()
                save()
                for index in range(3):
                    trial = dict(round=index+1, phases={})
                    row['rounds'].append(trial)
                    for label, fn in (('baseline_before', baseline), ('candidate', candidate), ('baseline_after', baseline)):
                        print(json.dumps(dict(event='measure', T=length, dtype=str(dtype), round=index+1, phase=label)), flush=True)
                        trial['phases'][label] = dict(started_at_unix=time.time(), **timing(fn, args.warmup, args.repeat))
                        save()
                    before, after = (trial['phases'][n]['median_us'] for n in ('baseline_before', 'baseline_after'))
                    trial['baseline_drift_ratio'] = after/before
                    trial['candidate_over_mean_baseline'] = trial['phases']['candidate']['median_us']/((before+after)/2)
                row['after'] = verify()
                assert {n: digest(x) for n, x in npu.items()} == hashes, 'input mutated'
                row['passed'] = True
                save()
        report['passed'] = True
    except BaseException as error:
        report.update(passed=False, failure=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
