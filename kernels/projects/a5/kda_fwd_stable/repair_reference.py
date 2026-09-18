"""Runtime-generated FP32 references for the Aqk handoff repair.

The independent recurrence stores state as [value, key] and uses batched
matrix-vector products. It does not consume a kernel checkpoint or an FLA
intermediate. FLA is loaded separately, by file, only for CPU verification.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch


def make_inputs(case: dict) -> dict:
    b, h, hv, c = (case[k] for k in ("B", "H", "HV", "C"))
    gen = torch.Generator(device="cpu").manual_seed(case.get("seed", 2026))
    q, k = (torch.nn.functional.normalize(
        torch.randn(b, c * 64, h, 128, generator=gen), dim=-1) for _ in range(2))
    raw = -torch.rand(b, c * 64, hv, 128, generator=gen) * .03
    cum = raw.view(b, c, 64, hv, 128).cumsum(2)
    span = (cum.amax(2) - cum.amin(2)).max()
    x = {"q": q.bfloat16(), "k": k.bfloat16(),
         "v": (torch.randn(b, c * 64, hv, 128, generator=gen) * .04).bfloat16(),
         "g": raw * (case.get("span", 46.) / span),
         "beta": torch.rand(b, c * 64, hv, generator=gen) * .45 + .05,
         "h0": torch.randn(b, hv, 128, 128, generator=gen) * .01}
    if case.get("initial_state", "random") == "zero":
        x["h0"].zero_()
    return {name: value.contiguous() for name, value in x.items()}


def independent_reference(x: dict) -> dict:
    if any(t.device.type != "cpu" for t in x.values()):
        raise ValueError("Correctness references must run on Torch CPU")
    b, t, h, width = x["q"].shape
    hv = x["v"].shape[2]
    group = torch.arange(hv) // (hv // h)
    state = x["h0"].float().transpose(-1, -2).reshape(b * hv, width, width).contiguous()
    output = torch.empty(b, t, hv, width, dtype=torch.float32)
    states = []
    for token in range(t):
        key = x["k"][:, token, group].float().reshape(b * hv, width, 1)
        query = x["q"][:, token, group].float().reshape(b * hv, width, 1)
        decay = x["g"][:, token].exp().reshape(b * hv, 1, width)
        state = state * decay
        prediction = torch.bmm(state, key)
        innovation = x["v"][:, token].float().reshape(b * hv, width, 1) - prediction
        innovation = innovation * x["beta"][:, token].reshape(b * hv, 1, 1)
        state = state + torch.bmm(innovation, key.transpose(1, 2))
        output[:, token] = (torch.bmm(state, query) * (width ** -.5)).reshape(b, hv, width)
        if token % 64 == 63:
            states.append(state.reshape(b, hv, width, width).transpose(-1, -2).contiguous().clone())
    return {"o": output, "final_state": states[-1], "chunk_states": torch.stack(states, 1)}


def fla_reference(x: dict) -> dict:
    if any(t.device.type != "cpu" for t in x.values()):
        raise ValueError("FLA golden must run on Torch CPU")
    path = os.environ.get("FLA_KDA_NAIVE")
    if not path or not Path(path).is_file():
        raise RuntimeError("Set FLA_KDA_NAIVE to the pinned fla/ops/kda/naive.py")
    spec = importlib.util.spec_from_file_location("fla_kda_repair_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state, outputs, states = x["h0"].clone(), [], []
    for start in range(0, x["q"].shape[1], 64):
        output, state = module.naive_recurrent_kda(
            *(x[name][:, start:start + 64].float() for name in ("q", "k", "v", "g", "beta")),
            initial_state=state, output_final_state=True)
        outputs.append(output)
        states.append(state.clone())
    return {"o": torch.cat(outputs, 1), "final_state": state,
            "chunk_states": torch.stack(states, 1)}


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, e = actual.detach().cpu().float(), expected.detach().cpu().float()
    residual, norm = (a - e).norm(), e.norm()
    rel = float(residual / norm) if float(norm) else (0. if float(residual) == 0 else float("inf"))
    finite = bool(a.isfinite().all() and e.isfinite().all())
    close = bool(torch.allclose(a, e, rtol=.02, atol=.02))
    return {"relative_l2": rel, "max_abs_diff": float((a - e).abs().max()),
            "finite": finite, "allclose": close, "passed": finite and close and rel <= .05}


def grid_cases() -> list[dict]:
    # Single-owner, repeated heads, multiple GQA groups, real Kimi head count.
    heads = [(1, 1), (1, 2), (1, 4), (2, 4), (4, 4), (4, 8), (32, 32)]
    return [{"id": f"b1_h{h}_hv{hv}_c{c}_{state}", "B": 1, "H": h,
             "HV": hv, "C": c, "initial_state": state, "span": 46., "seed": 2026}
            for c in range(1, 7) for h, hv in heads for state in ("zero", "random")]
