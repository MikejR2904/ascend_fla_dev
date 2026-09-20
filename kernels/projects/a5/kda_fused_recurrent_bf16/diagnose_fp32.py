"""Bounded FP32 diagnostic, to run only after the complete NPU workload."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from ascend_fla.ops.kda.fused_recurrent import _native_kernel
from _unit_runner import launch_kernel
from native_checks import compare, per_head
import research


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launcher', choices=('sim', 'pipesim'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = research.make_inputs(dict(B=1, T=2, H=1, G=4, state=False))
    for name in ('q', 'k', 'v'):
        data[name] = data[name].float()
    refs = research.references(data, research.load_fla(os.environ['BF06_FLA_NAIVE']))
    state = torch.full((1, 4, 128, 128), float('nan'))
    output = torch.full((1, 2, 4, 128), float('nan'))
    final = torch.full_like(state, float('nan'))
    inputs = tuple(data[n] for n in ('q', 'k', 'v', 'g', 'beta'))
    inputs += (state, output, final, 1, 2, 1, 4, 0, data['scale'])
    options = dict(launcher=args.launcher, device='a5', backend='cce', block_dim=1,
                   timeout=150, sim_processes='threads', out_dir=str(args.output), board=None)
    got = launch_kernel(_native_kernel(torch.float32), inputs, options)
    comparison = compare(got, refs, torch.float32, research)
    heads = per_head(got, refs, torch.float32, research)
    assert all(row['passed'] for row in heads)
    result = dict(stage=args.launcher, case=dict(B=1, T=2, H=1, HV=4, has_initial=0),
                  dtype='float32', passed=True, comparison=comparison, per_head=heads,
                  execution=options['_execution_evidence'],
                  source_sha256=hashlib.sha256(
                      Path(__file__).with_name('kernels').joinpath('step.py').read_bytes()).hexdigest())
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
