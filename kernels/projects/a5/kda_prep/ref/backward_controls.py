"""Pre-kernel formula corruption controls against the frozen BF-08 budgets.

No candidate or device kernel is imported. Structural faults use independent
FP64 derivatives, then the declared raw-gradient output rounding. Near-limit
perturbations quantify the actual rejection boundary, including that rounding.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path

import torch

from backward_calibrate import DTYPES, ROOT, beta_reference, gate_reference, load, norm_reference


def run(output):
    torch.set_num_threads(1)
    precision = load("_bf08_control_metrics", ROOT / "ref/calibrate.py")
    budget_path = ROOT / "backward_budgets.json"
    budget = json.loads(budget_path.read_text())
    generator = torch.Generator().manual_seed(8011)
    rows, reference_checks = [], []

    def record(key, kind, reference, wrong, metadata=None, required=True):
        group = budget["groups"][key]
        metric = precision.metrics(wrong, reference)
        l2 = metric["relative_l2"]
        relative = metric["max_relative_nonzero"]
        reasons = []
        if l2 is None or l2 > group["relative_l2_limit"]:
            reasons.append("relative_l2")
        if relative is None or relative > group["elementwise_relative_limit"]:
            reasons.append("elementwise_relative")
        if metric["zero_reference_nonzero_actual"]:
            reasons.append("reference_zero")
        if group["ulp_limit_each_reference"] is not None and (
            metric["rounded_reference_max_ulp"] > group["ulp_limit_each_reference"]
            or metric["rounded_reference_sign_differences"]
        ):
            reasons.append("rounded_fp64_ulp")
        row = dict(key=key, corruption=kind, metrics=metric,
                   l2_limit=group["relative_l2_limit"],
                   relative_limit=group["elementwise_relative_limit"],
                   l2_to_limit=l2 / group["relative_l2_limit"] if l2 is not None else None,
                   rejection_reasons=reasons, rejected=bool(reasons), metadata=metadata,
                   required_rejection=required)
        rows.append(row)
        if required:
            assert reasons, row
        return row

    def boundary(key, reference, dtype):
        # Search only the corruption magnitude, never the acceptance limit.
        # Retain the first rounded perturbation just above the fixed L2 line.
        limit = budget["groups"][key]["relative_l2_limit"]
        picked = None
        for multiplier in (1.01, 1.05, 1.1, 1.25, 1.5, 2.):
            wrong = (reference * (1 + multiplier * limit)).to(dtype)
            norm = torch.linalg.vector_norm(reference)
            l2 = float(torch.linalg.vector_norm(wrong.double() - reference) / norm)
            if l2 > limit:
                picked = (multiplier, wrong)
                break
        assert picked is not None
        result = record(key, "scale_just_above_frozen_l2_limit", reference, picked[1],
                        dict(multiplier_of_limit=picked[0], formula="round(reference*(1+multiplier*limit))"))
        assert 1 < result["l2_to_limit"] < 2.1

    def reference_metric(actual, reference):
        delta = (actual - reference).abs()
        return dict(relative_l2=float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)),
                    max_abs=float(delta.max()))

    for dtype, dt in DTYPES.items():
        x = torch.randn((2, 192, 2, 128), generator=generator).to(dt)
        gy = torch.randn(x.shape, generator=generator).bfloat16()
        z, d = x.double(), gy.double()
        square = (z * z).sum(-1, keepdim=True) + 1e-6
        r = square.rsqrt()
        dot = (z * d).sum(-1, keepdim=True)
        reference = norm_reference(x, gy)
        key = f"norm:{dtype}:dx"
        record(key, "omit_projection_term", reference, (d * r).to(dt))
        record(key, "omit_r_cubed_factor", reference, (d * r - z * dot).to(dt))
        omitted = (z * d)[..., -1:]
        record(key, "omit_last_dot_product_element", reference,
               (d * r - z * (dot - omitted) * r.pow(3)).to(dt))
        boundary(key, reference, dt)
        # Independent autograd FP64 vs the analytic equation. This tests the
        # derivative oracle, not any candidate implementation.
        leaf = z.clone().requires_grad_()
        y = leaf / (leaf.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        automatic, = torch.autograd.grad(y, leaf, d)
        reference_checks.append(dict(key=key, analytic_vs_fp64_autograd=reference_metric(automatic, reference)))

    for types in itertools.product(DTYPES, repeat=3):
        shape = (2, 192, 8, 128)
        g = torch.randn(shape, generator=generator).to(DTYPES[types[0]])
        a = torch.linspace(-3., 2.7, 8).to(DTYPES[types[1]])
        bias = torch.linspace(-.2, .2, 8 * 128).to(DTYPES[types[2]])
        gy = torch.randn(shape, generator=generator)
        reference = gate_reference(g, a, bias, gy)
        u = g.double() + bias.double().view(8, 128)
        decay = -a.double().exp().view(8, 1)
        sensitivity = gy.double()
        softplus = torch.logaddexp(u, torch.zeros_like(u))
        unchained = sensitivity * decay
        da_blocks = (sensitivity * softplus).reshape(384, 8, 128)
        dg_blocks = reference["dg"].reshape(384, 8, 128)
        faulty = dict(dg=unchained,
                      dA_log=da_blocks[:-32].sum((0, 2)) * decay.reshape(-1),
                      ddt_bias=dg_blocks[:-32].sum(0).reshape(-1))
        for name, dt in zip(("dg", "dA_log", "ddt_bias"), types):
            key = f"gate:{'_'.join(types)}:{name}"
            record(key, "omit_sigmoid_factor" if name == "dg" else "omit_last_32_bt_rows",
                   reference[name], faulty[name].to(DTYPES[dt]))
            boundary(key, reference[name], DTYPES[dt])
        leaves = [t.double().requires_grad_() for t in (g, a, bias)]
        gg, aa, bb = leaves
        yy = -aa.exp().view(8, 1) * torch.nn.functional.softplus(gg + bb.view(8, 128), threshold=20)
        automatic = torch.autograd.grad(yy, leaves, sensitivity)
        for name, result in zip(("dg", "dA_log", "ddt_bias"), automatic):
            reference_checks.append(dict(key=f"gate:{'_'.join(types)}:{name}",
                analytic_vs_fp64_autograd=reference_metric(result, reference[name])))

    for dtype, dt in DTYPES.items():
        x = torch.randn((2, 192, 8), generator=generator).to(dt)
        gy = torch.randn(x.shape, generator=generator)
        reference = beta_reference(x, gy)
        key = f"beta:{dtype}:dbeta"
        record(key, "omit_one_minus_sigmoid", reference, (gy.double() * x.double().sigmoid()).to(dt))
        boundary(key, reference, dt)
        leaf = x.double().requires_grad_()
        automatic, = torch.autograd.grad(leaf.sigmoid(), leaf, gy.double())
        reference_checks.append(dict(key=key, analytic_vs_fp64_autograd=reference_metric(automatic, reference)))

    for check in reference_checks:
        assert check["analytic_vs_fp64_autograd"]["relative_l2"] < 1e-12
    payload = dict(scope="CPU pre-implementation corruption controls; no candidate kernel",
                   python=platform.python_version(), torch=torch.__version__, seed=8011,
                   driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   budgets_sha256=hashlib.sha256(budget_path.read_bytes()).hexdigest(),
                   controls=rows, reference_checks=reference_checks,
                   complete=True, all_required_rejected=all(r["rejected"] for r in rows if r["required_rejection"]))
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(controls=len(rows), reference_checks=len(reference_checks),
                         all_required_rejected=payload["all_required_rejected"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
