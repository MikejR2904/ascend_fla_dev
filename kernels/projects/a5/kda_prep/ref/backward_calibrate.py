"""BF-08 pre-implementation CPU precision study, independent of candidate kernels.

The pinned predecessor is differentiated by Torch autograd. Independent FP64
analytic derivatives are precision references only; full KDA goldens stay FP32.
This script records observations. It does not authorize a new endpoint class.
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


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def predecessors():
    receipt = json.loads((ROOT / "baseline_backward/source.json").read_text())
    for name, item in receipt["files"].items():
        assert hashlib.sha256((ROOT / "baseline_backward" / name).read_bytes()).hexdigest() == item["sha256"]
    return load("_bf08_predecessor_chunk", ROOT / "baseline_backward/chunk.py"), receipt


def norm_reference(x, gy):
    z, d = x.detach().double(), gy.double()
    square = (z * z).sum(-1, keepdim=True) + 1e-6
    denominator = square.sqrt()
    return d / denominator - z * ((z * d).sum(-1, keepdim=True) / (square * denominator))


def gate_reference(g, a, bias, gy):
    u = g.detach().double() + bias.detach().double().view(g.shape[-2:])
    decay = -a.detach().double().exp().view(-1, 1)
    e = (-u.abs()).exp()
    sigmoid = torch.where(u >= 0, 1 / (1 + e), e / (1 + e))
    derivative = torch.where(u > 20, torch.ones_like(u), sigmoid)
    softplus = torch.where(u > 20, u, torch.logaddexp(u, torch.zeros_like(u)))
    sensitivity = gy.double()
    dg = sensitivity * decay * derivative
    da = (sensitivity * softplus).sum((0, 1, 3)) * decay.reshape(-1)
    db = dg.sum((0, 1)).reshape(-1)
    return dict(dg=dg, dA_log=da, ddt_bias=db)


def beta_reference(beta, gy):
    # Avoid subtracting an already rounded sigmoid from one in the FP64 oracle.
    e = (-beta.detach().double().abs()).exp()
    return gy.double() * e / ((1 + e) * (1 + e))


def run(output, section):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    old, predecessor_source = predecessors()
    precision = load("_bf08_precision_metrics", ROOT / "ref/calibrate.py")
    rows, controls = [], []
    environment = dict(python=platform.python_version(), torch=torch.__version__,
        device="CPU", precision_reference="independent analytic FP64; not the KDA end-to-end golden",
        predecessor=predecessor_source,
        calibration_driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric_driver_sha256=hashlib.sha256((ROOT / "ref/calibrate.py").read_bytes()).hexdigest())

    def save():
        payload = dict(environment=environment, section=section, complete=False,
            scope="Pre-implementation observations; endpoint classifications and BF16 ULP policy require owner disposition.",
            rows=rows, negative_controls=controls)
        (output / "pre-kernel.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    def record(route, types, kind, inputs, sensitivity, results, references, ordinary):
        for name, result in results.items():
            metric = precision.metrics(result, references[name])
            key = route + ":" + types + ":" + name
            row = dict(id=f"{key}:{kind}:{len(rows)}", key=key, route=route, types=types, output=name,
                kind=kind, input_shape=list(inputs[0].shape), output_shape=list(result.shape),
                output_dtype=str(result.dtype), sensitivity_dtype=str(sensitivity.dtype),
                input_sha256=[precision.digest(t) for t in inputs], sensitivity_sha256=precision.digest(sensitivity),
                ordinary_calibration_population=ordinary, metrics=metric)
            rows.append(row)
            if ordinary and kind in ("gaussian", "full_kimi"):
                for label, wrong in (("zero", torch.zeros_like(result)), ("negated", -result),
                                     ("scaled_1p25", result * 1.25)):
                    bad = precision.metrics(wrong, references[name])
                    floor = metric["relative_l2"]
                    controls.append(dict(case=row["id"], corruption=label,
                        relative_l2=bad["relative_l2"], floor=floor,
                        rejected_by_three_times_own_floor=(floor is not None and bad["relative_l2"] is not None
                            and bad["relative_l2"] > 3 * floor)))
            save()
            print(json.dumps(dict(index=len(rows), key=key, kind=kind, ordinary=ordinary,
                relative_l2=metric["relative_l2"], max_relative=metric["max_relative_nonzero"],
                max_ulp=metric["rounded_reference_max_ulp"], over_one_ulp=metric["rounded_reference_over_one_ulp"])), flush=True)

    if section in ("all", "norm"):
        rng = torch.Generator().manual_seed(8008)
        for dtype, dt in DTYPES.items():
            for shape in ((2, 192, 2, 128), (1, 4096, 32, 128)):
                x = torch.randn(shape, generator=rng).to(dt).requires_grad_()
                gy = torch.randn(shape, generator=rng).bfloat16()
                y = old._prepare_inputs(x, x, None, None, use_qk_l2norm_in_kernel=True)[0]
                actual, = torch.autograd.grad(y, x, gy)
                reference = norm_reference(x, gy)
                record("norm", dtype, "full_kimi" if shape[1] == 4096 else "gaussian",
                    [x], gy, dict(dx=actual), dict(dx=reference), True)
                del x, gy, y, actual, reference
            for scale in (0., 1e-20, 1e-8, 1e-4, 30., 1e10, 1e18, 1e20):
                x = (torch.randn((1, 64, 2, 128), generator=rng) * scale).to(dt).requires_grad_()
                gy = torch.randn(x.shape, generator=rng).bfloat16()
                y = old._prepare_inputs(x, x, None, None, use_qk_l2norm_in_kernel=True)[0]
                actual, = torch.autograd.grad(y, x, gy)
                record("norm", dtype, f"scale_{scale:g}", [x], gy, dict(dx=actual),
                    dict(dx=norm_reference(x, gy)), scale <= 30.)
            x = torch.randn((1, 64, 2, 128), generator=rng).to(dt).requires_grad_()
            gy = x.detach().bfloat16()
            y = old._prepare_inputs(x, x, None, None, use_qk_l2norm_in_kernel=True)[0]
            actual, = torch.autograd.grad(y, x, gy)
            record("norm", dtype, "near_null_parallel_sensitivity_observation", [x], gy,
                dict(dx=actual), dict(dx=norm_reference(x, gy)), False)

    if section in ("all", "gate"):
        rng = torch.Generator().manual_seed(8009)
        for types in itertools.product(DTYPES, repeat=3):
            for kind, shape in (("gaussian", (2, 192, 8, 128)), ("full_kimi", (1, 4096, 32, 128)),
                                ("threshold", (2, 64, 2, 128))):
                hv = shape[-2]
                if kind == "threshold":
                    values = torch.tensor([-40., -20., -1., 0., 1., 19.999998092651367, 20., 20.000001907348633, 40., 80.])
                    g = values.repeat((torch.tensor(shape).prod().item() + len(values) - 1) // len(values))[:torch.tensor(shape).prod().item()].reshape(shape)
                    bias = torch.zeros(hv * 128)
                else:
                    g = torch.randn(shape, generator=rng)
                    bias = torch.linspace(-.2, .2, hv * 128)
                a = torch.linspace(-3., 2.7, hv)
                g, a, bias = [t.to(DTYPES[d]).requires_grad_() for t, d in zip((g, a, bias), types)]
                gy = torch.randn(shape, generator=rng)
                y = old._prepare_inputs(None, None, g, None, A_log=a, dt_bias=bias, use_gate_in_kernel=True)[2]
                grads = torch.autograd.grad(y, (g, a, bias), gy)
                reference = gate_reference(g, a, bias, gy)
                record("gate", "_".join(types), kind, [g, a, bias], gy,
                    dict(zip(("dg", "dA_log", "ddt_bias"), grads)), reference, True)
                del g, a, bias, gy, y, grads, reference
        for dtype, dt in DTYPES.items():
            for a_value in (-120., -104., -100., 80., 88., 89., 100.):
                g = torch.linspace(-2., 2., 128).reshape(1, 1, 1, 128).to(dt).requires_grad_()
                a = torch.tensor([a_value], dtype=dt, requires_grad=True)
                bias = torch.zeros(128, dtype=dt, requires_grad=True)
                gy = torch.randn(g.shape, generator=rng)
                y = old._prepare_inputs(None, None, g, None, A_log=a, dt_bias=bias, use_gate_in_kernel=True)[2]
                grads = torch.autograd.grad(y, (g, a, bias), gy)
                record("gate", "_".join([dtype] * 3), f"alog_{a_value:g}", [g, a, bias], gy,
                    dict(zip(("dg", "dA_log", "ddt_bias"), grads)), gate_reference(g, a, bias, gy), False)

    if section in ("all", "beta"):
        rng = torch.Generator().manual_seed(8010)
        for dtype, dt in DTYPES.items():
            for kind, source in (("gaussian", torch.randn((2, 192, 8), generator=rng)),
                    ("full_kimi", torch.randn((1, 4096, 32), generator=rng)),
                    ("zero", torch.zeros(1, 64, 2)),
                    ("saturation", torch.tensor([-120., -104., -100., -90., -88., -40., -20., -1., 0., 1., 16., 20., 40., 100.]).reshape(1, 14, 1))):
                x = source.to(dt).requires_grad_()
                gy = torch.randn(x.shape, generator=rng)
                y = old._prepare_inputs(None, None, None, x, use_beta_sigmoid_in_kernel=True)[3]
                actual, = torch.autograd.grad(y, x, gy)
                record("beta", dtype, kind, [x], gy, dict(dbeta=actual),
                    dict(dbeta=beta_reference(x, gy)), kind != "saturation")

    payload = json.loads((output / "pre-kernel.json").read_text())
    payload["complete"] = True
    (output / "pre-kernel.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    floors = {}
    for row in rows:
        if not row["ordinary_calibration_population"]:
            continue
        metric = row["metrics"]
        assert metric["relative_l2"] is not None
        f = floors.setdefault(row["key"], dict(relative_l2=0., max_relative_nonzero=0.,
            reference_zero_actual_nonzero=0, rounded_reference_max_ulp=0, cases=0))
        f["cases"] += 1
        for key in ("relative_l2", "max_relative_nonzero", "rounded_reference_max_ulp"):
            f[key] = max(f[key], metric[key])
        f["reference_zero_actual_nonzero"] += metric["zero_reference_nonzero_actual"]
    summary = dict(environment=environment, complete=True, records=len(rows), groups=len(floors),
        calibration_sha256=hashlib.sha256((output / "pre-kernel.json").read_bytes()).hexdigest(),
        floors=floors, negative_controls=len(controls), all_controls_rejected=all(c["rejected_by_three_times_own_floor"] for c in controls),
        budgets_frozen=False, pending="Owner clarification of BF16 gradient ULP and newly observed endpoint/ill-conditioned cases; no new acceptance class is introduced.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("environment", "floors")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--section", choices=("all", "norm", "gate", "beta"), default="all")
    args = parser.parse_args()
    run(args.output, args.section)
