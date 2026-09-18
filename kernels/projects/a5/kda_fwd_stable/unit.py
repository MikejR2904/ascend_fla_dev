"""Standalone stable KDA unit with the repaired Aqk handoff.

CPU references import no compiler. Native performance uses verify_repair.py;
the canonical launcher includes transfer/process overhead and is diagnostic.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

from repair_reference import independent_reference, make_inputs as repair_inputs


def make_inputs(case):
    p = case["parameters"]
    if (p.get("L", 64), p.get("K", 128), p.get("V", 128)) != (64, 128, 128):
        raise ValueError("This unit requires L=64 and K=V=128")
    if "gate_multiplier" in p:
        # Preserve the historical stable-unit input distribution and seed order.
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
        raise ValueError("q must be [B,T,H,128]")
    b, t, h, width = q.shape
    hv = inputs["v"].shape[2]
    if min(b, t, h, hv) <= 0 or t % 64 or width != 128 or hv % h:
        raise ValueError("Require positive aligned T, K=V=128 and HV divisible by H")
    specs = {"q": ((b, t, h, 128), torch.bfloat16), "k": ((b, t, h, 128), torch.bfloat16),
             "v": ((b, t, hv, 128), torch.bfloat16), "g_raw": ((b, t, hv, 128), torch.float32),
             "beta": ((b, t, hv), torch.float32), "initial_state": ((b, hv, 128, 128), torch.float32)}
    for name, (shape, dtype) in specs.items():
        value = inputs[name]
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu"
                or not value.is_contiguous() or not bool(value.isfinite().all())):
            raise ValueError(f"{name} requires finite contiguous CPU {dtype} {shape}")
    if len({t.untyped_storage().data_ptr() for t in inputs.values()}) != len(specs):
        raise ValueError("Input allocations must not alias")
    if case is not None and case.get("block_dim", 1) not in (1, 2, 3, 4):
        raise ValueError("block_dim must be in 1..4")


def reference(inputs):
    validate_inputs(inputs)
    x = {"q": inputs["q"], "k": inputs["k"], "v": inputs["v"], "g": inputs["g_raw"],
         "beta": inputs["beta"], "h0": inputs["initial_state"]}
    result = independent_reference(x)
    b, t, hv, d = x["g"].shape
    gc = x["g"].reshape(b, t // 64, 64, hv, d).cumsum(2).permute(0, 3, 1, 2, 4).contiguous()
    return {"o": result["o"], "final_state": result["final_state"], "g_cumsum": gc}


def reference_stages(inputs):
    """CPU stage semantics using direct non-positive gate differences for scores."""
    validate_inputs(inputs)
    b, t, h, _ = inputs["q"].shape
    hv, c = inputs["v"].shape[2], t // 64

    def chunk(value):
        return value.reshape(b, c, 64, *value.shape[2:]).transpose(1, 3).contiguous().float()

    # transpose above produces [B,H,64,C,D]; put C before the token dimension.
    q, k, v, raw = [chunk(inputs[n]).transpose(2, 3).contiguous() for n in ("q", "k", "v", "g_raw")]
    beta = inputs["beta"].reshape(b, c, 64, hv).permute(0, 3, 1, 2)
    q, k = [value.repeat_interleave(hv // h, 1) for value in (q, k)]
    gc = raw.cumsum(-2)
    eg = gc.exp()
    score = torch.zeros(b, hv, c, 64, 64)
    strict = torch.zeros_like(score)
    for i in range(64):
        decayed = k[..., :i + 1, :] * (gc[..., i:i + 1, :] - gc[..., :i + 1, :]).exp()
        score[..., i, :i + 1] = (decayed * q[..., i:i + 1, :]).sum(-1) * 128**-.5
        strict[..., i, :i] = -(decayed[..., :i, :] * k[..., i:i + 1, :]).sum(-1) * beta[..., i, None]
    aqk = score.bfloat16()
    eye = torch.eye(64).expand_as(strict)
    akk = torch.linalg.solve_triangular(eye - strict, eye, upper=False, unitriangular=True).bfloat16()
    abeta = (akk.float() * beta[..., None, :]).bfloat16().float()
    w = (abeta @ (k * eg).bfloat16().float()).bfloat16()
    u = (abeta @ v).bfloat16()
    return {"eg": eg, "g_cumsum": gc, "Aqk": aqk, "strict": strict, "Akk": akk,
            "w": w, "u": u, "qg": (q * eg).bfloat16(),
            "kg": (k * (gc[..., -1:, :] - gc).exp()).bfloat16()}


def _execute_chain(inputs, options):
    import inspect
    from _unit_runner import launch_kernel

    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from repair_runtime import run_chain, selected_kernels, _from_bhcld

    validate_inputs(inputs)
    if options["device"] != "a5" or options["backend"] != "cce" or options["block_dim"] not in (1, 2, 3, 4):
        raise ValueError("Only a5/cce and block_dim 1..4 are declared")

    def adapter(kernel):
        names = tuple(inspect.signature(kernel.fn).parameters)

        def launch(sources, scalars, outputs):
            for tensor in outputs.values():
                tensor.fill_(float("nan"))
            bound = {**sources, **outputs, **scalars}
            result = launch_kernel(kernel, tuple(bound[n] for n in names), options)
            values = (result,) if len(outputs) == 1 else result
            for target, value in zip(outputs.values(), values, strict=True):
                target.copy_(value)
        return launch

    b, t, h, _ = inputs["q"].shape
    hv = inputs["v"].shape[2]
    result = run_chain(*(inputs[n] for n in ("q", "k", "v", "g_raw", "beta")), 128**-.5,
                       inputs["initial_state"], device="a5", block_dim=options["block_dim"],
                       on_cpu=True, b=b, h=h, hv=hv, c=t // 64,
                       launchers={k: adapter(v) for k, v in selected_kernels("repaired").items()})
    result["o"] = _from_bhcld(result["o"], on_cpu=True)
    return result


def execute(inputs, options):
    result = _execute_chain(inputs, options)
    return {name: result[name] for name in ("o", "final_state", "g_cumsum")}


def execute_stages(inputs, options):
    result = _execute_chain(inputs, options)
    return {name: result[name] for name in ("eg", "g_cumsum", "Aqk", "strict", "Akk", "w", "u", "qg", "kg")}
