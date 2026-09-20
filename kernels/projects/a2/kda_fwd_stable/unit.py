"""A2 stable KDA forward unit: five kernels, public token-major BF16/FP32 ABI, no host-side conversion.

CPU references (``reference.py``) import no compiler. ``execute`` launches the five A2 kernels in order; the
host only allocates NaN-filled outputs and takes metadata-only ``view``s of the contiguous token-major inputs
(D-PM-35/37). The chain-internal stages keep the chunked head-major layout ``[B, HV, C, 64, *]``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from reference import independent_reference, make_inputs as repair_inputs, reference_stages as _reference_stages

HERE = Path(__file__).resolve().parent
SCALE = 128 ** -0.5
BLOCK_DIMS = (1, 2)


def make_inputs(case):
    p = case["parameters"]
    if (p.get("L", 64), p.get("K", 128), p.get("V", 128)) != (64, 128, 128):
        raise ValueError("This unit requires L=64 and K=V=128")
    if "gate_multiplier" in p:
        # The A5 stable unit's input distribution and seed order, so the cases are comparable.
        b, h, hv, c = (p[k] for k in ("B", "H", "HV", "C"))
        gen = torch.Generator().manual_seed(case["seed"])
        q, k = [(torch.randn(b, c * 64, h, 128, generator=gen) * .04).bfloat16() for _ in range(2)]
        v = (torch.randn(b, c * 64, hv, 128, generator=gen) * .04).bfloat16()
        g = -torch.rand(b, c * 64, hv, 128, generator=gen) * .03 * p["gate_multiplier"]
        beta = torch.rand(b, c * 64, hv, generator=gen) * .45 + .05
        h0 = torch.randn(b, hv, 128, 128, generator=gen) * .01
        if p.get("initial_state") == "zero":
            h0.zero_()
        x = dict(q=q, k=k, v=v, g=g, beta=beta, h0=h0)
    else:
        x = repair_inputs({**p, "seed": case["seed"]})
    return {"q": x["q"], "k": x["k"], "v": x["v"], "g_raw": x["g"],
            "beta": x["beta"], "initial_state": x["h0"]}


def validate_inputs(inputs, case=None):
    q = inputs["q"]
    if q.ndim != 4:
        raise ValueError(f"q must be [B,T,H,128]; got rank {q.ndim}")
    b, t, h, width = q.shape
    hv = inputs["v"].shape[2]
    if min(b, t, h, hv) <= 0 or t % 64 or width != 128 or hv % h:
        raise ValueError(f"Require positive T % 64 == 0, K=V=128 and HV % H == 0; got B={b} T={t} H={h} HV={hv} K={width}")
    specs = {"q": ((b, t, h, 128), torch.bfloat16), "k": ((b, t, h, 128), torch.bfloat16),
             "v": ((b, t, hv, 128), torch.bfloat16), "g_raw": ((b, t, hv, 128), torch.float32),
             "beta": ((b, t, hv), torch.float32), "initial_state": ((b, hv, 128, 128), torch.float32)}
    for name, (shape, dtype) in specs.items():
        value = inputs[name]
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu"
                or not value.is_contiguous() or not bool(value.isfinite().all())):
            raise ValueError(f"{name} requires finite contiguous CPU {dtype} {shape}; got {value.dtype} {tuple(value.shape)}")
    if len({t.untyped_storage().data_ptr() for t in inputs.values()}) != len(specs):
        raise ValueError("Input allocations must not alias")
    if case is not None and case.get("block_dim", 1) not in BLOCK_DIMS:
        raise ValueError(f"block_dim must be one of {BLOCK_DIMS} (the A2 values measured so far); got {case.get('block_dim')}")


def reference(inputs):
    validate_inputs(inputs)
    x = {"q": inputs["q"], "k": inputs["k"], "v": inputs["v"], "g": inputs["g_raw"],
         "beta": inputs["beta"], "h0": inputs["initial_state"]}
    result = independent_reference(x)
    b, t, hv, d = x["g"].shape
    gc = x["g"].reshape(b, t // 64, 64, hv, d).cumsum(2).permute(0, 3, 1, 2, 4).contiguous()
    return {"o": result["o"], "final_state": result["final_state"], "g_cumsum": gc}


def reference_stages(inputs):
    validate_inputs(inputs)
    return _reference_stages(inputs)


def _kernels():
    out = {}
    for module, fn in (("gate", "kda_sub1_gate_a2_kernel"), ("intra", "kda_sub2_score_a2_kernel"),
                       ("triangular_inverse", "tril_inverse64_a2_kernel"), ("wy", "kda_sub3_wy_a2_kernel"),
                       ("recurrent", "kda_sub45_a2_kernel")):
        name = f"_a2_kda_fwd_{module}"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, HERE / "kernels" / f"{module}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
        out[module] = getattr(sys.modules[name], fn)
    return out


def _execute_chain(inputs, options):
    from _unit_runner import launch_kernel

    validate_inputs(inputs)
    if options["device"] != "a2" or options["backend"] != "cce" or options["block_dim"] not in BLOCK_DIMS:
        raise ValueError(f"Only a2/cce and block_dim in {BLOCK_DIMS} are declared; got "
                         f"{options['device']}/{options['backend']}/{options['block_dim']}")
    kern = _kernels()
    q, k, v, g, beta, h0 = (inputs[n] for n in ("q", "k", "v", "g_raw", "beta", "initial_state"))
    b, t, h, _ = q.shape
    hv, c = v.shape[2], t // 64
    nan = float("nan")
    # host work: allocation of NaN-seeded outputs and metadata-only views (no dtype or layout conversion)
    q2, k2 = q.view(b * t, h * 128), k.view(b * t, h * 128)
    v2, g2, beta2 = v.view(b * t, hv * 128), g.view(b * t, hv * 128), beta.view(b * t, hv)

    gc = torch.full((b, hv, c, 64, 128), nan)
    eg = torch.full((b, hv, c, 64, 128), nan)
    gc, eg = launch_kernel(kern["gate"], (g2, gc, eg, b, hv, c), options)

    aqk = torch.full((b, hv, c, 64, 64), nan, dtype=torch.bfloat16)
    strict = torch.full((b, hv, c, 64, 64), nan)
    aqk, strict = launch_kernel(kern["intra"], (q2, k2, gc, beta2, aqk, strict, b, h, hv, c, SCALE), options)

    akk = torch.full((b, hv, c, 64, 64), nan, dtype=torch.bfloat16)
    akk = launch_kernel(kern["triangular_inverse"], (strict, akk, b, hv, c), options)

    w, u, qg, kg = (torch.full((b, hv, c, 64, 128), nan, dtype=torch.bfloat16) for _ in range(4))
    w, u, qg, kg = launch_kernel(kern["wy"], (q2, k2, v2, beta2, akk, gc, w, u, qg, kg, b, h, hv, c), options)

    o = torch.full((b * t, hv * 128), nan, dtype=torch.bfloat16)
    final_state = torch.full((b, hv, 128, 128), nan)
    o, final_state = launch_kernel(kern["recurrent"], (q2, aqk, kg, w, u, eg, h0, o, final_state, b, h, hv, c, SCALE),
                                   options)
    return {"o": o.view(b, t, hv, 128), "final_state": final_state, "g_cumsum": gc, "eg": eg, "Aqk": aqk,
            "strict": strict, "Akk": akk, "w": w, "u": u, "qg": qg, "kg": kg}


def execute(inputs, options):
    result = _execute_chain(inputs, options)
    return {name: result[name] for name in ("o", "final_state", "g_cumsum")}


def execute_stages(inputs, options):
    result = _execute_chain(inputs, options)
    return {name: result[name] for name in ("eg", "g_cumsum", "Aqk", "strict", "Akk", "w", "u", "qg", "kg")}
