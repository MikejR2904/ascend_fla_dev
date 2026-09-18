"""A5K-01 native verification and same-device timing. Outputs go to scratch.

One invocation owns one block_dim. Correctness always uses CPU references;
timed calls always use NPU tensors. Existing public dispatch/gates are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[3]
sys.path.insert(0, str(REPO))

import torch

from repair_reference import fla_reference, grid_cases, independent_reference, make_inputs, metrics
from repair_runtime import compiled_chain, run_chain, _from_bhcld


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def source_identity():
    paths = [UNIT / "repair_runtime.py", UNIT / "repair_reference.py", Path(__file__),
             *sorted((UNIT / "kernels").glob("*.py"))]
    result = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    from ascend_fla.ops.kda.chunk import _kernels_root
    for name in ("recurrent.py", "triangular_inverse.py"):
        result[f"upstream-kda/kernels/{name}"] = hashlib.sha256((_kernels_root() / "kernels" / name).read_bytes()).hexdigest()
    result["fla/ops/kda/naive.py"] = hashlib.sha256(Path(os.environ["FLA_KDA_NAIVE"]).read_bytes()).hexdigest()
    return result


def execute(x, case, bd, variant):
    return run_chain(*(x[n] for n in ("q", "k", "v", "g", "beta")), 128**-.5, x["h0"],
                     device="a5", block_dim=bd, on_cpu=False, b=case["B"], h=case["H"],
                     hv=case["HV"], c=case["C"], variant=variant)


def final_outputs(chain):
    return {"o": _from_bhcld(chain["o"], on_cpu=False), "final_state": chain["final_state"]}


def timing(call, warmup, repeat):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeat):
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        value = call()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
        del value
    return {"warmup": warmup, "repeat": repeat, "synchronize_each_sample": True,
            "samples_us": samples, "median_us": statistics.median(samples)}


def verify(case, bd, out_dir):
    x = make_inputs(case)
    ref, fla = independent_reference(x), fla_reference(x)
    oracle_checks = {name: metrics(ref[name], fla[name]) for name in ref}
    if not all(v["passed"] and v["relative_l2"] <= 1e-5 for v in oracle_checks.values()):
        raise AssertionError(f"Independent CPU oracles disagree: {oracle_checks}")
    dev = {name: tensor.to("npu") for name, tensor in x.items()}
    row = {"case": case, "block_dim": bd, "cpu_oracles": oracle_checks, "variants": {}}
    chains, results = {}, {}
    for variant in ("baseline", "repaired"):
        chains[variant] = execute(dev, case, bd, variant)
        torch.npu.synchronize()
        results[variant] = {k: v.cpu() for k, v in final_outputs(chains[variant]).items()}
        row["variants"][variant] = {
            "vs_independent": {k: metrics(v, ref[k]) for k, v in results[variant].items()},
            "vs_fla": {k: metrics(v, fla[k]) for k, v in results[variant].items()},
            "stage_sha256": {k: digest(v) for k, v in chains[variant].items() if v is not None},
        }
    exact = {k: torch.equal(v, chains["baseline"][k])
             for k, v in chains["repaired"].items() if v is not None}
    row["baseline_bitwise"] = exact
    row["per_head_chunk"] = []
    for b in range(case["B"]):
        for head in range(case["HV"]):
            for chunk in range(case["C"]):
                sl = (b, slice(chunk * 64, (chunk + 1) * 64), head)
                row["per_head_chunk"].append({"b": b, "head": head, "chunk": chunk,
                    "o_vs_independent": metrics(results["repaired"]["o"][sl], ref["o"][sl]),
                    "o_vs_fla": metrics(results["repaired"]["o"][sl], fla["o"][sl])})
    # Prefix launches observe every chunk state without changing the kernel ABI.
    states = []
    for c in range(1, case["C"] + 1):
        if c == case["C"]:
            state = results["repaired"]["final_state"]
        else:
            prefix = {k: (v[:, :c * 64].contiguous() if k != "h0" else v) for k, v in dev.items()}
            state = execute(prefix, {**case, "C": c}, bd, "repaired")["final_state"].cpu()
        states.append(state)
    states = torch.stack(states, 1)
    for r in row["per_head_chunk"]:
        sl = (r["b"], r["chunk"], r["head"])
        r["state_vs_independent"] = metrics(states[sl], ref["chunk_states"][sl])
        r["state_vs_fla"] = metrics(states[sl], fla["chunk_states"][sl])
    row["chunk_states_sha256"] = digest(states)
    row["passed"] = (
        all(exact[k] for k in exact if k not in ("o", "final_state"))
        and (case["C"] % 2 == 1 or all(exact.values()))
        and all(v["passed"] for r in row["per_head_chunk"]
                for k, v in r.items() if k.endswith(("independent", "fla"))))
    row["baseline_correct"] = all(v["passed"] for d in ("vs_independent", "vs_fla")
                                     for v in row["variants"]["baseline"][d].values())
    path = out_dir / (case["id"] + ".json")
    path.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({"case": case["id"], "passed": row["passed"],
                      "baseline_correct": row["baseline_correct"],
                      "repaired": row["variants"]["repaired"]["vs_independent"]}), flush=True)
    return row


def profile(bd, out_dir, warmup, repeat):
    from ascend_fla.reference.kda import kda_chunk_vectorized
    rows = []
    for c in (16, 64):
        case = dict(id=f"kimi_t{c*64}", B=1, H=32, HV=32, C=c, span=46., initial_state="random")
        x = make_inputs(case)
        ref, fla = independent_reference(x), fla_reference(x)
        dev = {k: v.to("npu") for k, v in x.items()}
        got = {v: execute(dev, case, bd, v) for v in ("baseline", "repaired")}
        finals = {v: final_outputs(y) for v, y in got.items()}
        checks = {v: {k: metrics(t, ref[k]) for k, t in output.items()} for v, output in finals.items()}
        fla_checks = {v: {k: metrics(t, fla[k]) for k, t in output.items()} for v, output in finals.items()}
        equality = {k: torch.equal(finals["baseline"][k], finals["repaired"][k]) for k in ref if k != "chunk_states"}
        def composition():
            return kda_chunk_vectorized(*(dev[k] for k in ("q", "k", "v", "g", "beta")),
                                       initial_state=dev["h0"], output_final_state=True, check_gate_range=False)

        def native_composition(variant):
            return final_outputs(execute(dev, case, bd, variant))
        torch_result = composition()
        torch_checks = {k: metrics(v, ref[k]) for k, v in zip(("o", "final_state"), torch_result)}
        if not (all(equality.values()) and all(m["passed"] for check in checks.values() for m in check.values())
                and all(m["passed"] for check in fla_checks.values() for m in check.values())
                and all(m["passed"] for m in torch_checks.values())):
            failure = {"case": case, "checks": checks, "fla_checks": fla_checks,
                       "bitwise": equality, "torch_checks": torch_checks}
            (out_dir / "profile-correctness-failure.json").write_text(json.dumps(failure, indent=2))
            raise AssertionError("Only correctness-passing cases may be timed")
        rounds = []
        for round_id in range(3):
            before = timing(lambda: native_composition("baseline"), warmup, repeat)
            candidate = timing(lambda: native_composition("repaired"), warmup, repeat)
            after = timing(lambda: native_composition("baseline"), warmup, repeat)
            rounds.append({"round": round_id, "baseline_before": before, "candidate": candidate,
                           "baseline_after": after,
                           "ratio_vs_faster_baseline": min(before["median_us"], after["median_us"]) / candidate["median_us"]})
        row = {"case": case, "block_dim": bd, "checks": checks, "fla_checks": fla_checks, "baseline_bitwise": equality,
               "torch_checks": torch_checks, "rounds": rounds,
               "torch_npu_composition": timing(composition, warmup, repeat),
               "timed_scope": "unit composition including allocation/input and output layout/five launches; compilation and validation excluded for both native and Torch NPU"}
        rows.append(row)
        (out_dir / "profile.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({"case": case["id"], "ratios": [r["ratio_vs_faster_baseline"] for r in rounds],
                          "torch_us": row["torch_npu_composition"]["median_us"]}), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("grid", "profile"))
    parser.add_argument("--block-dim", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--case", default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    import torch_npu
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"sources": source_identity(), "torch": torch.__version__, "torch_npu": torch_npu.__version__,
               "block_dim": args.block_dim, "reference_device": "cpu", "execution_device": "npu",
               "budget": {"rtol": .02, "atol": .02, "max_relative_l2": .05}}
    (args.output / "identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for variant in ("baseline", "repaired"):
        print(f"compile {variant} bd={args.block_dim}", flush=True)
        compiled_chain(args.block_dim, variant)
    if args.mode == "profile":
        profile(args.block_dim, args.output, args.warmup, args.repeat)
    else:
        cases = grid_cases()
        if args.case in ("kimi_t1024", "kimi_t4096"):
            cases = [dict(id=args.case, B=1, H=32, HV=32,
                          C=int(args.case.split("t")[-1]) // 64,
                          span=46., initial_state="random", seed=2026)]
        if args.case != "all":
            cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            parser.error("Unknown case")
        rows = [verify(case, args.block_dim, args.output) for case in cases]
        passed = all(r["passed"] for r in rows)
        (args.output / "summary.json").write_text(json.dumps({"cases": len(rows), "passed": passed,
            "baseline_failures": [r["case"]["id"] for r in rows if not r["baseline_correct"]]}, indent=2) + "\n")
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
