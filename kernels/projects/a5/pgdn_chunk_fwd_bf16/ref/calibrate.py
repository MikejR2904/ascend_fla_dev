"""Freeze A/B rounding floors before authoring the BF16 device kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from . import oracle
from .reference import cases, make_inputs, reference, metric, floor, budget, validate_reference


def run(output):
    torch.set_num_threads(1)
    report = dict(stage='cpu_pre_kernel_calibration', python=sys.version, torch=torch.__version__,
                  fla_pin=oracle.PIN, oracle_sha256=oracle.SHA256,
                  rule='o/main state: min(1e-2,3F) per case/output/oracle; F=0 exact; ATK and FP32 1e-4',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob('*.py')}, cases=[])
    for case in cases():
        for dtype in ('bfloat16', 'float32'):
            started = time.monotonic()
            inputs = make_inputs(dict(case, dtype=dtype))
            a, b = oracle.reference(inputs), reference(inputs)
            validate_reference(inputs, a)
            validate_reference(inputs, b)
            numbers = {n: metric(b[n], a[n]) for n in a}
            assert all(x['finite'] and x['relative_l2'] <= 1e-4 for x in numbers.values()), (case, numbers)
            row = dict(case=case, dtype=dtype, A_B=numbers,
                       floors={label: {n: floor(x) for n, x in values.items()}
                               for label, values in [('A', a), ('B', b)]},
                       budgets={label: {n: budget(x, inputs['q'].dtype, n) for n, x in values.items()}
                                for label, values in [('A', a), ('B', b)]},
                       seconds=time.monotonic() - started)
            if 'norm' in case['parameters']:
                row['actual_q_first_component'] = inputs['q'][0, 0, 0, 0].item()
            report['cases'].append(row)
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print('CALIBRATION', case['id'], dtype, json.dumps(numbers, allow_nan=False), flush=True)
    report['passed'] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('CALIBRATION_PASS', len(report['cases']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
