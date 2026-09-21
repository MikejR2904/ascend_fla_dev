"""CPU-only pre-implementation precision study; never imports candidate kernels.

FP64 is a precision-study reference. End-to-end KDA acceptance retains its two
CPU FP32 goldens. Inputs are rounded to their declared dtype before either path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}


def digest(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def predecessor():
    receipt = json.loads((ROOT / "baseline/source.json").read_text())
    for name, item in receipt["files"].items():
        assert hashlib.sha256((ROOT / "baseline" / name).read_bytes()).hexdigest() == item["sha256"]
    spec = importlib.util.spec_from_file_location("_bf07_predecessor_chunk", ROOT / "baseline/chunk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(actual, expected):
    a, b = actual.detach().double().reshape(-1), expected.detach().double().reshape(-1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    aa, bb = a[finite], b[finite]
    delta = (aa - bb).abs()
    norm = torch.linalg.vector_norm(bb).item()
    residual = torch.linalg.vector_norm(delta).item()
    nonzero = bb != 0
    relative = delta[nonzero] / bb[nonzero].abs()
    rounded = expected.to(actual.dtype)
    good = torch.isfinite(actual) & torch.isfinite(rounded)
    # Same-sign ordered IEEE encodings have the same ULP distance as integers.
    # Sign-crossing values are separately counted (including signed zero).
    code_dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    a_code, b_code = actual.contiguous().view(code_dtype).long(), rounded.contiguous().view(code_dtype).long()
    same_sign = torch.signbit(actual) == torch.signbit(rounded)
    ulps = (a_code - b_code).abs()[good & same_sign]
    return dict(elements=a.numel(), finite_pairs=int(finite.sum()),
                actual_nan=int(torch.isnan(a).sum()), reference_nan=int(torch.isnan(b).sum()),
                actual_posinf=int(torch.isposinf(a).sum()), reference_posinf=int(torch.isposinf(b).sum()),
                actual_neginf=int(torch.isneginf(a).sum()), reference_neginf=int(torch.isneginf(b).sum()),
                reference_norm=norm, residual_norm=residual,
                relative_l2=(residual / norm if norm else (0.0 if residual == 0 else None)) if bool(finite.all()) else None,
                relative_l2_finite_pairs=residual / norm if norm else (0.0 if residual == 0 else None),
                max_abs=delta.max().item() if delta.numel() else None,
                max_relative_nonzero=relative.max().item() if relative.numel() else 0.0,
                relative_quantiles=torch.quantile(relative, torch.tensor([.5, .9, .99, 1.], dtype=torch.float64)).tolist() if relative.numel() else [0.] * 4,
                zero_reference_nonzero_actual=int((aa[~nonzero] != 0).sum()),
                rounded_reference_max_ulp=int(ulps.max()) if ulps.numel() else 0,
                rounded_reference_over_one_ulp=int((ulps > 1).sum()),
                rounded_reference_sign_differences=int((good & ~same_sign).sum()),
                actual_sha256=digest(actual), reference_fp64_sha256=digest(expected))


def high_precision_norm(x):
    z = x.double()
    return z / (torch.sum(z * z, dim=-1, keepdim=True) + 1e-6).sqrt()


def high_precision_gate(g, a, bias):
    u = g.double() + bias.double().view(g.shape[-2:])
    # Independent stable log(1+exp(u)), with the predecessor's strict threshold.
    softplus = torch.where(u > 20., u, torch.logaddexp(u, torch.zeros_like(u)))
    return -torch.exp(a.double()).view(-1, 1) * softplus


def high_precision_beta(x):
    z = x.double()
    e = torch.exp(-z.abs())
    return torch.where(z >= 0., 1. / (1. + e), e / (1. + e))


def run(output):
    torch.set_num_threads(1)
    old = predecessor()
    rows = []
    negative_controls = []
    generator = torch.Generator().manual_seed(7007)

    def record(op, types, kind, inputs, result, reference, numerical=True):
        row = dict(op=op, types=types, kind=kind, shape=list(inputs[0].shape),
                   input_sha256=[digest(x) for x in inputs],
                   numerical_budget_population=numerical, metrics=metrics(result, reference))
        rows.append(row)
        if numerical and kind in ('gaussian', 'random'):
            for label, wrong in [('zero', torch.zeros_like(result)), ('negated', -result),
                                 ('scaled_1p25', result * 1.25)]:
                bad = metrics(wrong, reference)
                negative_controls.append(dict(op=op, types=types, kind=kind, shape=list(result.shape),
                    corruption=label, relative_l2=bad['relative_l2'],
                    calibration_floor=row['metrics']['relative_l2'],
                    rejected=bad['relative_l2'] > 3 * row['metrics']['relative_l2']))
                assert negative_controls[-1]['rejected'], negative_controls[-1]
        print(json.dumps(dict(index=len(rows), op=op, types=types, kind=kind,
                              relative_l2=row['metrics']['relative_l2']), allow_nan=False), flush=True)

    # Mixed parity, multi-batch/head and full Kimi shape; full-sized values are
    # generated at run time, never committed as tensor payloads.
    for shape in ((1, 64, 1, 128), (2, 192, 2, 128), (1, 4096, 32, 128)):
        source = torch.randn(shape, generator=generator)
        for itype, otype in itertools.product(DTYPES, repeat=2):
            x = source.to(DTYPES[itype])
            got = old._prepare_inputs(x, x, None, None, use_qk_l2norm_in_kernel=True,
                                      qk_dtype=DTYPES[otype])[0]
            record('norm', f'{itype}_{otype}', 'gaussian', [x], got, high_precision_norm(x))
    source = torch.randn((1, 64, 2, 128), generator=generator)
    for scale in (0., 1e-20, 1e-8, 1e-4, 30., 1e10, 1e18, 1e20):
        for itype, otype in itertools.product(DTYPES, repeat=2):
            x = (source * scale).to(DTYPES[itype])
            got = old._prepare_inputs(x, x, None, None, use_qk_l2norm_in_kernel=True,
                                      qk_dtype=DTYPES[otype])[0]
            record('norm', f'{itype}_{otype}', f'scale_{scale:g}', [x], got,
                   high_precision_norm(x), numerical=scale < 1e18)

    gate_kinds = {
        'random': None,
        'threshold': [-40., -20., -1., 0., 1., 19.999998092651367, 20., 20.000001907348633, 40., 80.],
        'subnormal': [-120., -104., -100., -90., -88., -87.],
    }
    for types in itertools.product(DTYPES, repeat=3):
        for kind, values in gate_kinds.items():
            shape = (2, 192, 8, 128)
            g = torch.randn(shape, generator=generator) if values is None else torch.tensor(values).repeat((torch.tensor(shape).prod().item() + len(values)-1)//len(values))[:torch.tensor(shape).prod().item()].reshape(shape)
            a = torch.linspace(-3., 2.7, 8)
            bias = torch.linspace(-.2, .2, 8*128) if values is None else torch.zeros(8*128)
            g, a, bias = [x.to(DTYPES[t]) for x, t in zip((g, a, bias), types)]
            got = old._prepare_inputs(None, None, g, None, A_log=a, dt_bias=bias, use_gate_in_kernel=True)[2]
            record('gate', '_'.join(types), kind, [g, a, bias], got,
                   high_precision_gate(g, a, bias), numerical=kind != 'subnormal')
    for dtype in DTYPES:
        for alog in (-120., -104., -100., 80., 88., 88.7, 89., 100.):
            g = torch.linspace(-2., 2., 128).reshape(1, 1, 1, 128).to(DTYPES[dtype])
            a = torch.tensor([alog], dtype=DTYPES[dtype]); bias = torch.zeros(128, dtype=DTYPES[dtype])
            got = old._prepare_inputs(None, None, g, None, A_log=a, dt_bias=bias, use_gate_in_kernel=True)[2]
            record('gate', '_'.join([dtype]*3), f'alog_{alog:g}', [g, a, bias], got,
                   high_precision_gate(g, a, bias), numerical=False)

    for dtype in DTYPES:
        for kind, x in (
            ('random', torch.randn((2, 192, 8), generator=generator)*3),
            ('full_kimi', torch.randn((1, 4096, 32), generator=generator)),
            ('saturation', torch.tensor([-120., -104., -100., -90., -88., -40., -20., -1., 0., 1., 16., 20., 40., 100.]).reshape(1, 14, 1)),
            ('zero', torch.zeros(1, 1, 1)),
        ):
            x = x.to(DTYPES[dtype])
            got = old._prepare_inputs(None, None, None, x, use_beta_sigmoid_in_kernel=True)[3]
            record('beta', dtype, kind, [x], got, high_precision_beta(x), numerical=kind != 'saturation')

    # Nonfinite propagation and signed-zero are semantic endpoints, not a
    # population from which to inflate ordinary finite numerical budgets.
    for dtype in DTYPES:
        x = torch.zeros(1, 4, 1, 128, dtype=DTYPES[dtype])
        x[0, 0, 0, 0] = float('nan'); x[0, 1, 0, 0] = float('inf')
        x[0, 2, 0, 0] = float('-inf'); x[0, 3] = -0.
        for otype in DTYPES:
            got = old._prepare_inputs(x, x, None, None, use_qk_l2norm_in_kernel=True,
                                      qk_dtype=DTYPES[otype])[0]
            record('norm', f'{dtype}_{otype}', 'nonfinite_signedzero', [x], got,
                   high_precision_norm(x), numerical=False)
        a = torch.zeros(1, dtype=DTYPES[dtype]); bias = torch.zeros(128, dtype=DTYPES[dtype])
        got = old._prepare_inputs(None, None, x, None, A_log=a, dt_bias=bias, use_gate_in_kernel=True)[2]
        record('gate', '_'.join([dtype]*3), 'nonfinite_signedzero', [x,a,bias], got,
               high_precision_gate(x,a,bias), numerical=False)
        beta = torch.tensor([float('nan'),float('inf'),float('-inf'),-0.], dtype=DTYPES[dtype]).reshape(1,4,1)
        got = old._prepare_inputs(None,None,None,beta,use_beta_sigmoid_in_kernel=True)[3]
        record('beta', dtype, 'nonfinite_signedzero', [beta], got, high_precision_beta(beta), numerical=False)

    groups = {}
    for row in rows:
        if not row['numerical_budget_population']:
            continue
        key = row['op'] + ':' + row['types']
        group = groups.setdefault(key, dict(cases=0, floor_relative_l2=0., floor_max_relative=0., max_ulp=0))
        m = row['metrics']; group['cases'] += 1
        group['floor_relative_l2'] = max(group['floor_relative_l2'], m['relative_l2'])
        group['floor_max_relative'] = max(group['floor_max_relative'], m['max_relative_nonzero'])
        group['max_ulp'] = max(group['max_ulp'], m['rounded_reference_max_ulp'])
    result = dict(scope='CPU FP64 precision study, predecessor FP32 evaluation; no candidate kernel or native execution',
                  python=platform.python_version(), torch=torch.__version__, seed=7007,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  predecessor=json.loads((ROOT/'baseline/source.json').read_text()),
                  groups=groups, cases=rows, negative_controls=negative_controls)
    output.mkdir(parents=True, exist_ok=True)
    (output/'pre-kernel.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(cases=len(rows), groups=groups), indent=2)+'\n')
    print('CALIBRATION COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
