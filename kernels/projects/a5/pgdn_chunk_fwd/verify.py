"""Report both pinned FLA A and independent CPU B against actual model outputs.

The canonical run.py owns contract and leaf checks. This companion adds the
second oracle and per-output numbers without changing that portable helper.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform

import torch

import unit
from kernels.pipeline import run

ROOT = Path(__file__).resolve().parent
NAIVE_SHA256 = '3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8'
NAMES = ('q', 'k', 'v', 'g_atk', 'g', 'beta_atk', 'beta')


def metric(got, expected):
    delta = got.float()-expected.float()
    norm = expected.float().norm().item()
    return dict(relative_l2=delta.norm().item()/norm if norm else (0. if not delta.any() else float('inf')),
                max_abs=delta.abs().max().item(), finite=bool(torch.isfinite(got).all()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launcher', choices=('reference','sim','pipesim'), required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--block-dim', type=int, choices=(1,2))
    p.add_argument('--fla-naive', type=Path, required=True)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert hashlib.sha256(args.fla_naive.read_bytes()).hexdigest() == NAIVE_SHA256, 'FLA naive pin mismatch'
    spec = importlib.util.spec_from_file_location('pgdn_verification_naive', args.fla_naive)
    fla = importlib.util.module_from_spec(spec); spec.loader.exec_module(fla)
    contract = json.loads((ROOT/'contract.json').read_text())
    cases = [c for c in contract['cases'] if c['id'] == args.case or (args.case == 'all' and c['block_dim'] == 1)]
    if not cases:
        raise ValueError('no matching case')
    if args.case == 'all' and args.launcher != 'reference':
        raise ValueError('model execution must name a bounded case')
    torch.set_num_threads(1)
    report = dict(schema='pgdn-dual-reference/1', stage=args.launcher, versions=dict(python=platform.python_version(),torch=torch.__version__),
                  oracle_sha256=NAIVE_SHA256, source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ROOT.rglob('*.py'))},cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        inputs = unit.make_inputs(case)
        before = {n:x.clone() for n,x in inputs.items()}
        b = unit.reference_stages(inputs)
        a = dict(zip(('o','final_state','final_A_state'),fla.naive_recurrent_precond_gated_delta_rule(*(inputs[n] for n in NAMES),output_final_state=True)))
        options = dict(device='a5', backend='cce', launcher=args.launcher, block_dim=args.block_dim or case['block_dim'], timeout=args.timeout, sim_processes='fork')
        if args.launcher == 'reference':
            got = b
        else:
            actual_launch = unit._launch(inputs, options)
            def launch(entry, sources, outputs, scalars):
                print(json.dumps(dict(event='stage_start', kernel=entry.name, stage=args.launcher)), flush=True)
                result = actual_launch(entry, sources, outputs, scalars)
                print(json.dumps(dict(event='stage_done', kernel=entry.name, stage=args.launcher)), flush=True)
                return result
            got = run(inputs, launch)
        row = dict(case=case['id'], parameters=case['parameters'], seed=case['seed'], block_dim=options['block_dim'], oracles={}, stages={})
        report['cases'].append(row)
        for label, expected in (('A_pinned_cpu_recurrence',a),('B_independent_cpu_block_solve',b)):
            row['oracles'][label] = {n:metric(got[n],expected[n]) for n in a}
            for n in a:
                torch.testing.assert_close(got[n],expected[n],atol=2e-5,rtol=2e-4)
                assert row['oracles'][label][n]['relative_l2'] <= 1e-4, row
        for n in b:
            row['stages'][n] = metric(got[n],b[n])
            torch.testing.assert_close(got[n],b[n],atol=2e-5,rtol=2e-4)
            assert row['stages'][n]['relative_l2'] <= 1e-4, (n,row['stages'][n])
        # Storage rounding quality only; this does not execute a BF16 kernel.
        row['bf16_output_storage_quality'] = {label:metric(got['o'].bfloat16().float(),expected['o']) for label,expected in (('A',a),('B',b))}
        for name,expected in before.items():
            assert torch.equal(inputs[name],expected), name
        row['stage_sha256'] = {n:hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for n,x in got.items()}
        row['execution'] = options.get('_execution_evidence',[])
        row['passed'] = True
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(case=case['id'],stage=args.launcher,oracles=row['oracles'])),flush=True)
    report['passed'] = True
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__ == '__main__':
    main()
