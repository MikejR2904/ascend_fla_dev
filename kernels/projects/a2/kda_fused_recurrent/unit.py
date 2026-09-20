"""A2 KDA decode unit: token-by-token recurrence (T <= 16) on the public token-major BF16/FP32 tensors.

The kernel reads ``q``/``k``/``v`` (BF16) and ``g``/``beta`` (FP32) in their public layout, does the BF16 -> FP32
casts, the ``scale`` multiply and the GQA head map itself, and writes ``o`` in BF16; the host only allocates
NaN-seeded outputs and takes metadata-only views (D-PM-35/37). CPU references import no compiler.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from reference import independent_reference, make_inputs as _make_inputs

HERE = Path(__file__).resolve().parent
SCALE = 128 ** -0.5
T_MAX = 16
BLOCK_DIMS = (1, 2)


def make_inputs(case):
    return _make_inputs(case)


def validate_inputs(inputs, case=None):
    q = inputs["q"]
    if q.ndim != 4:
        raise ValueError(f"q must be [B,T,H,128]; got rank {q.ndim}")
    b, t, h, width = q.shape
    hv = inputs["v"].shape[2]
    if min(b, t, h, hv) <= 0 or t > T_MAX or width != 128 or hv % h:
        raise ValueError(f"Require 1 <= T <= {T_MAX}, K=V=128 and HV % H == 0; got B={b} T={t} H={h} HV={hv} K={width}")
    specs = {"q": ((b, t, h, 128), torch.bfloat16), "k": ((b, t, h, 128), torch.bfloat16),
             "v": ((b, t, hv, 128), torch.bfloat16), "g": ((b, t, hv, 128), torch.float32),
             "beta": ((b, t, hv), torch.float32), "initial_state": ((b, hv, 128, 128), torch.float32)}
    for name, (shape, dtype) in specs.items():
        value = inputs[name]
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu"
                or not value.is_contiguous() or not bool(value.isfinite().all())):
            raise ValueError(f"{name} requires finite contiguous CPU {dtype} {shape}; got {value.dtype} {tuple(value.shape)}")
    if case is not None and case.get("block_dim", 1) not in BLOCK_DIMS:
        raise ValueError(f"block_dim must be one of {BLOCK_DIMS} (the A2 values measured so far); got {case.get('block_dim')}")


def reference(inputs):
    validate_inputs(inputs)
    return independent_reference(inputs, SCALE)


def _kernel():
    name = "_a2_kda_fused_recurrent_step"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "kernels" / "step.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name].kda_fused_recurrent_a2_kernel


def execute(inputs, options):
    from _unit_runner import launch_kernel

    validate_inputs(inputs)
    if options["device"] != "a2" or options["backend"] != "cce" or options["block_dim"] not in BLOCK_DIMS:
        raise ValueError(f"Only a2/cce and block_dim in {BLOCK_DIMS} are declared; got "
                         f"{options['device']}/{options['backend']}/{options['block_dim']}")
    q, k, v, g, beta, h0 = (inputs[n] for n in ("q", "k", "v", "g", "beta", "initial_state"))
    b, t, h, _ = q.shape
    hv = v.shape[2]
    o = torch.full((b * t, hv * 128), float("nan"), dtype=torch.bfloat16)
    final_state = torch.full((b, hv, 128, 128), float("nan"))
    o, final_state = launch_kernel(_kernel(), (q.view(b * t, h * 128), k.view(b * t, h * 128), v.view(b * t, hv * 128),
                                               g.view(b * t, hv * 128), beta.view(b * t, hv), h0, o, final_state,
                                               b, t, h, hv, SCALE), options)
    return {"o": o.view(b, t, hv, 128), "final_state": final_state}
