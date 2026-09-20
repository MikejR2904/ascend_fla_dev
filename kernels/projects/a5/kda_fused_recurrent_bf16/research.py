"""CPU-only, pre-implementation BF16 calibration for KDA decode.

Both goldens consume the same BF16-rounded q/k/v as FP32 tensors. The ordered
candidate is a precision experiment, never the hardware correctness oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import time
from pathlib import Path

import torch

from ascend_fla.reference.kda import kda_recurrent_ref

FLA_COMMIT = "9c8e42e762fce087c27b673af4922795d9edb85e"
FLA_NAIVE_SHA256 = "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016"
OUTPUT_LIMIT = 1e-2
FLOOR_MULTIPLIER = 3.0
STATE_LIMIT = 1e-5


def digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def metrics(got, ref):
    got, ref = got.double(), ref.double()
    diff = got - ref
    return {"relative_l2": float(diff.norm() / ref.norm().clamp_min(1e-30)),
            "max_abs": float(diff.abs().max()),
            "finite": bool(torch.isfinite(got).all())}


def cases():
    rows = []
    for (b, h), t, group, state in itertools.product(
            ((1, 1), (2, 2)), (1, 2, 16), (1, 2, 4, 8), (False, True)):
        rows.append(dict(id=f"grid_b{b}_h{h}_t{t}_g{group}_s{int(state)}",
                         B=b, H=h, T=t, G=group, state=state))
    for t in range(3, 16):
        rows.append(dict(id=f"tail_t{t}", B=1, H=3, T=t, G=2, state=True))
    for t, h, group in ((1, 32, 1), (16, 32, 1), (1, 4, 8), (16, 8, 4)):
        rows.append(dict(id=f"real_t{t}_h{h}_g{group}", B=1, H=h, T=t,
                         G=group, state=True))
    for gate, state in itertools.product(("zero", "tiny", "deep", "underflow", "strong_weak"), (False, True)):
        rows.append(dict(id=f"gate_{gate}_s{int(state)}", B=1, H=2, T=16,
                         G=4, state=state, gate=gate))
    for scale in (0.0, 1.0, 0.3, -1.0):
        rows.append(dict(id=f"scale_{scale}", B=2, H=2, T=16, G=4,
                         state=True, scale=scale))
    for beta in (0.0, 1.0):
        rows.append(dict(id=f"beta_{beta}", B=1, H=2, T=16, G=4,
                         state=True, beta=beta))
    rows.append(dict(id="zero_inputs", B=1, H=1, T=16, G=1,
                     state=False, zero=True))
    return rows


def make_inputs(p, seed=20260920):
    gen = torch.Generator().manual_seed(seed)
    b, t, h, hv = p["B"], p["T"], p["H"], p["H"] * p["G"]
    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)
    def rand(*shape):
        return torch.rand(*shape, generator=gen, dtype=torch.float32)
    q, k = randn(b, t, h, 128), randn(b, t, h, 128)
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q, k = q.bfloat16(), k.bfloat16()
    v = (randn(b, t, hv, 128) * .2).bfloat16()
    g = -rand(b, t, hv, 128) * .1
    beta = .1 + .8 * rand(b, t, hv)
    state = randn(b, hv, 128, 128) * .05 if p["state"] else None
    gate = p.get("gate")
    if gate in ("zero", "tiny", "deep", "underflow"):
        g.fill_({"zero": 0., "tiny": -1e-5, "deep": -155., "underflow": -1000.}[gate])
    elif gate == "strong_weak":
        g.fill_(-1e-5)
        g[:, 0].fill_(-154.9992)
    if "beta" in p:
        beta.fill_(p["beta"])
    if p.get("zero"):
        q.zero_(); k.zero_(); v.zero_()
    return dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                scale=p.get("scale", 128 ** -.5), output_final_state=True)


def load_fla(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != FLA_NAIVE_SHA256:
        raise ValueError("FLA KDA naive source does not match the selected v0.5.2 identity")
    spec = importlib.util.spec_from_file_location("bf06_pinned_kda_naive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_kda


def references(x, fla):
    cpu = {name: value.float() if isinstance(value, torch.Tensor) else value
           for name, value in x.items()}
    assert all(value.device.type == "cpu" for value in cpu.values()
               if isinstance(value, torch.Tensor))
    return dict(A=fla(**cpu), B=kda_recurrent_ref(**cpu))


def ordered_candidate(x, *, round_state=False, old_bf16_scale=False):
    q, k, v, g, beta = (x[n].float() for n in ("q", "k", "v", "g", "beta"))
    b, t, h, _ = q.shape
    hv = v.shape[2]
    q = ((x["q"] * x["scale"]).float() if old_bf16_scale else q * x["scale"])
    q, k = (z.repeat_interleave(hv // h, dim=2) for z in (q, k))
    state = (torch.zeros(b, hv, 128, 128) if x["initial_state"] is None
             else x["initial_state"].clone())
    out = torch.empty_like(v)
    for ti in range(t):
        acc = torch.zeros(b, hv, 128)
        for ki in range(128):
            row = state[:, :, ki] * g[:, ti, :, ki].exp().unsqueeze(-1)
            state[:, :, ki] = row
            acc = acc + row * k[:, ti, :, ki].unsqueeze(-1)
        delta = v[:, ti] - acc
        acc = torch.zeros_like(acc)
        for ki in range(128):
            kb = k[:, ti, :, ki] * beta[:, ti]
            row = state[:, :, ki] + delta * kb.unsqueeze(-1)
            state[:, :, ki] = row
            acc = acc + row * q[:, ti, :, ki].unsqueeze(-1)
        out[:, ti] = acc
        if round_state:
            state = state.bfloat16().float()
    return out.bfloat16(), state


def compare(result, refs):
    answer = {}
    for name, (o, state) in refs.items():
        floor = metrics(o.bfloat16().float(), o)["relative_l2"]
        om, sm = metrics(result[0], o), metrics(result[1], state)
        budget = min(OUTPUT_LIMIT, FLOOR_MULTIPLIER * floor)
        answer[name] = dict(o=om, final_state=sm, output_floor=floor,
                            output_budget=budget, state_budget=STATE_LIMIT,
                            passed=om["finite"] and sm["finite"] and
                            om["relative_l2"] <= budget and
                            sm["relative_l2"] <= STATE_LIMIT)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fla-naive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fla = load_fla(args.fla_naive)
    started = time.monotonic()
    rows = []
    for p in cases():
        x = make_inputs(p)
        refs = references(x, fla)
        chosen = compare(ordered_candidate(x), refs)
        rounded = compare(ordered_candidate(x, round_state=True), refs)
        old = compare(ordered_candidate(x, old_bf16_scale=True), refs)
        rows.append(dict(case=p, input_hashes={n: digest(v) for n, v in x.items()
                                              if isinstance(v, torch.Tensor)},
                         oracle_agreement={"o": metrics(refs["A"][0], refs["B"][0]),
                                           "final_state": metrics(refs["A"][1], refs["B"][1])},
                         chosen_fp32_state=chosen, bf16_state_negative_control=rounded,
                         old_bf16_scale=old))
        print(p["id"], "PASS" if all(v["passed"] for v in chosen.values()) else "FAIL", flush=True)
    report = dict(stage="CPU precision calibration before kernel implementation",
                  python=platform.python_version(), torch=torch.__version__,
                  fla_commit=FLA_COMMIT, fla_naive_sha256=FLA_NAIVE_SHA256,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  budgets=dict(output_relative_l2=OUTPUT_LIMIT, output_floor_multiplier=FLOOR_MULTIPLIER,
                               final_state_relative_l2=STATE_LIMIT),
                  cases=rows, elapsed_seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    passed = sum(all(x["passed"] for x in row["chosen_fp32_state"].values()) for row in rows)
    negative_rejected = sum(not all(x["passed"] for x in row["bf16_state_negative_control"].values()) for row in rows)
    print(json.dumps(dict(cases=len(rows), chosen_passed=passed,
                          bf16_state_negative_rejected=negative_rejected,
                          elapsed_seconds=report["elapsed_seconds"])))
    if passed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
