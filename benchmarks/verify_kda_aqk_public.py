"""Verify and time actual public KDA forward entries on A5, one block_dim per process.

Correctness uses independent CPU FP32 references. Profiling uses NPU tensors,
including public layout and validation overhead. The explicit original recurrent
override is only a diagnostic negative control; candidate calls are unmodified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
UNIT = REPO / "kernels/projects/a5/kda_fwd_stable"
sys.path[:0] = [str(REPO), str(UNIT)]

import torch
from ascend_fla.ops.kda import chunk as public
from ascend_fla.ops.kda import chunk_bwd as backward
from repair_reference import fla_reference, grid_cases, independent_reference, make_inputs, metrics
from repair_runtime import compiled_chain
from verify_repair import digest, timing


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def identities():
    paths = [Path(__file__), REPO / "ascend_fla/ops/kda/chunk.py",
             REPO / "ascend_fla/ops/kda/chunk_bwd.py", UNIT / "repair_runtime.py",
             UNIT / "repair_reference.py", UNIT / "verify_repair.py",
             REPO / "ascend_fla/reference/kda.py", *sorted((UNIT / "kernels").glob("*.py")),
             *sorted((REPO / "kernels/projects/a5/kda_bwd_stable/kernels").glob("*.py"))]
    result = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name, root in (("upstream-fwd", public._kernels_root()),
                       ("upstream-bwd", backward._bwd_kernels_root())):
        for path in sorted((root / "kernels").glob("*.py")):
            result[f"{name}/kernels/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["fla/ops/kda/naive.py"] = hashlib.sha256(Path(os.environ["FLA_KDA_NAIVE"]).read_bytes()).hexdigest()
    return result


@contextmanager
def original_recurrent(block_dim):
    """Keep every public operation and stable stage except recurrent unchanged."""
    saved = public._compiled_chain
    baseline = compiled_chain(block_dim, "baseline")

    def select(device, bd, impl="stable"):
        assert (device, bd, impl) == ("a5", block_dim, "stable")
        return baseline

    public._compiled_chain = select
    try:
        yield
    finally:
        public._compiled_chain = saved


def invoke(x, bd, *, caches=False, impl="stable"):
    args = [x[k] for k in ("q", "k", "v", "g", "beta")]
    kwargs = dict(initial_state=x["h0"], block_dim=bd, layout_device="npu", impl=impl)
    if caches:
        return public.chunk_kda_fwd_with_caches(*args, **kwargs)
    return public.chunk_kda_fwd(*args, output_final_state=True, **kwargs)


def cpu_oracles(x):
    ref, fla = independent_reference(x), fla_reference(x)
    agreement = {k: metrics(ref[k], fla[k]) for k in ref}
    if not all(v["passed"] and v["relative_l2"] <= 1e-5 for v in agreement.values()):
        raise AssertionError(f"Independent CPU oracles disagree: {agreement}")
    return ref, fla, agreement


def verify(case, bd, out, *, with_caches=True):
    x = make_inputs(case)
    ref, fla, agreement = cpu_oracles(x)
    dev = {k: v.to("npu") for k, v in x.items()}
    # Candidate execution calls both public entries without overriding dispatch.
    o, state = invoke(dev, bd)
    caches, original_caches = {}, {}
    plain_cached = None
    if with_caches:
        oc, sc, caches = invoke(dev, bd, caches=True)
        plain_cached = dict(o=torch.equal(o, oc), final_state=torch.equal(state, sc))
    torch.npu.synchronize()
    with original_recurrent(bd):
        if with_caches:
            original_o, original_state, original_caches = invoke(dev, bd, caches=True)
        else:
            original_o, original_state = invoke(dev, bd)
    torch.npu.synchronize()
    cache_exact = {k: torch.equal(v, original_caches[k]) for k, v in caches.items()}
    baseline_exact = dict(o=torch.equal(o, original_o), final_state=torch.equal(state, original_state))
    outputs = dict(o=o.cpu(), final_state=state.cpu())
    row = dict(case=case, block_dim=bd, cpu_oracles=agreement,
               scope="forward_and_caches" if with_caches else "forward_only_diagnostic",
               vs_independent={k: metrics(v, ref[k]) for k, v in outputs.items()},
               vs_fla={k: metrics(v, fla[k]) for k, v in outputs.items()},
               baseline_vs_independent=dict(o=metrics(original_o, ref["o"]),
                                            final_state=metrics(original_state, ref["final_state"])),
               plain_cached_exact=plain_cached, baseline_exact=baseline_exact,
               cache_baseline_exact=cache_exact,
               output_sha256={k: digest(v) for k, v in outputs.items()},
               cache_sha256={k: digest(v) for k, v in caches.items()})
    row["cache_abi"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype), device=v.device.type)
                        for k, v in caches.items()}
    if with_caches:
        assert set(caches) == set(public.BWD_CACHE_NAMES)
    assert all(v.dtype == torch.bfloat16 and v.device.type == "npu" for v in caches.values())
    # Observe all chunk states through the public final-state ABI.
    states = []
    for c in range(1, case["C"] + 1):
        if c == case["C"]:
            states.append(outputs["final_state"])
        else:
            prefix = {k: v if k == "h0" else v[:, :c * 64].contiguous() for k, v in dev.items()}
            states.append(invoke(prefix, bd)[1].cpu())
    states = torch.stack(states, 1)
    row["chunk_states_sha256"] = digest(states)
    rows = []
    for b in range(case["B"]):
        for h in range(case["HV"]):
            for c in range(case["C"]):
                oslice = (b, slice(c * 64, (c + 1) * 64), h)
                sslice = (b, c, h)
                rows.append(dict(b=b, head=h, chunk=c,
                    o_vs_independent=metrics(outputs["o"][oslice], ref["o"][oslice]),
                    o_vs_fla=metrics(outputs["o"][oslice], fla["o"][oslice]),
                    state_vs_independent=metrics(states[sslice], ref["chunk_states"][sslice]),
                    state_vs_fla=metrics(states[sslice], fla["chunk_states"][sslice])))
    row["per_head_chunk"] = rows
    rejected = []
    if case["C"] % 2 and case["B"] * case["HV"] > bd:
        for cached in (False, True):
            try:
                invoke(dev, bd, caches=cached, impl="upstream")
            except ValueError as exc:
                if "silently corrupt" not in str(exc):
                    raise
                rejected.append("cached" if cached else "plain")
            else:
                raise AssertionError("Unsafe upstream public call was not rejected")
    row["unsafe_upstream_rejected"] = rejected
    row["passed"] = ((not with_caches or all(plain_cached.values())) and all(cache_exact.values())
        and baseline_exact["final_state"] and (case["C"] % 2 == 1 or baseline_exact["o"])
        and all(m["passed"] for name in ("vs_independent", "vs_fla") for m in row[name].values())
        and all(m["passed"] for item in rows for k, m in item.items() if "_vs_" in k))
    write(out / (case["id"] + ".json"), row)
    print(json.dumps(dict(case=case["id"], passed=row["passed"], public=row["vs_independent"],
                          baseline=row["baseline_vs_independent"])), flush=True)
    if not row["passed"]:
        raise AssertionError(f"Public validation failed: {case['id']}")
    return row


def profile(bd, out, warmup, repeat):
    from ascend_fla.reference.kda import kda_chunk_vectorized

    rows = []
    for c in (16, 64):
        case = dict(id=f"kimi_t{c * 64}", B=1, H=32, HV=32, C=c,
                    span=46., initial_state="random", seed=2026)
        correctness = verify(case, bd, out)
        x = make_inputs(case)
        ref, fla, _ = cpu_oracles(x)
        dev = {k: v.to("npu") for k, v in x.items()}

        def torch_npu():
            return kda_chunk_vectorized(*(dev[k] for k in ("q", "k", "v", "g", "beta")),
                                       initial_state=dev["h0"], output_final_state=True)

        y = torch_npu()
        checks = {label: {k: metrics(v, oracle[k]) for k, v in zip(("o", "final_state"), y)}
                  for label, oracle in (("independent", ref), ("fla", fla))}
        if not all(m["passed"] for check in checks.values() for m in check.values()):
            write(out / "torch-correctness-failure.json", checks)
            raise AssertionError("Torch NPU performance baseline failed correctness")
        rounds = []
        for index in range(3):
            with original_recurrent(bd):
                before = timing(lambda: invoke(dev, bd), warmup, repeat)
            candidate = timing(lambda: invoke(dev, bd), warmup, repeat)
            with original_recurrent(bd):
                after = timing(lambda: invoke(dev, bd), warmup, repeat)
            rounds.append(dict(round=index, baseline_before=before, candidate=candidate,
                               baseline_after=after, ratio_vs_faster_baseline=
                               min(before["median_us"], after["median_us"]) / candidate["median_us"]))
        row = dict(case=case, block_dim=bd, public_correctness_passed=correctness["passed"],
                   torch_checks=checks, rounds=rounds,
                   torch_npu_composition=timing(torch_npu, warmup, repeat),
                   timed_scope="Public forward, gate check enabled, NPU layout/allocation/five launches/output layout; compilation and cached-forward excluded. Torch NPU also uses its gate check. Original control changes only recurrent, outside sample timing.")
        rows.append(row)
        write(out / "profile.json", rows)
        print(json.dumps(dict(case=case["id"], ratios=[r["ratio_vs_faster_baseline"] for r in rounds],
                              torch_us=row["torch_npu_composition"]["median_us"])), flush=True)


def compare(root, output):
    """Require the complete grid and identical public/cache bytes across blocks."""
    expected = {case["id"]: case for case in grid_cases()}
    rows, identities_by_bd = {}, {}
    for bd in (1, 2, 3, 4):
        folder = root / f"grid-bd{bd}"
        identity = json.loads((folder / "identity.json").read_text())
        identities_by_bd[bd] = identity["sources"]
        assert identity["block_dim"] == bd
        assert identity["budget"] == dict(rtol=.02, atol=.02, max_relative_l2=.05)
        assert identity["reference_device"] == "cpu" and identity["execution_device"] == "npu"
        summary = json.loads((folder / "summary.json").read_text())
        assert summary == dict(cases=len(expected), passed=True)
        rows[bd] = {name: json.loads((folder / f"{name}.json").read_text()) for name in expected}
        for name, row in rows[bd].items():
            assert row["case"] == expected[name] and row["block_dim"] == bd and row["passed"]
            assert row["scope"] == "forward_and_caches", "Partial diagnostics cannot qualify the public grid"
            assert all(row["plain_cached_exact"].values()) and all(row["cache_baseline_exact"].values())
            assert row["baseline_exact"]["final_state"]
            assert row["case"]["C"] % 2 or row["baseline_exact"]["o"]
            assert len(row["per_head_chunk"]) == row["case"]["B"] * row["case"]["HV"] * row["case"]["C"]
            for item in row["per_head_chunk"]:
                assert all(m["passed"] for k, m in item.items() if "_vs_" in k)
            if row["case"]["C"] % 2 and row["case"]["B"] * row["case"]["HV"] > bd:
                assert row["unsafe_upstream_rejected"] == ["plain", "cached"]
    for bd in (2, 3, 4):
        assert identities_by_bd[bd] == identities_by_bd[1]
        for name in expected:
            for field in ("output_sha256", "cache_sha256", "chunk_states_sha256"):
                assert rows[bd][name][field] == rows[1][name][field], (bd, name, field)
    metrics_rows = [item for cases in rows.values() for row in cases.values() for item in row["per_head_chunk"]]
    maxima = {key: max(item[key]["relative_l2"] for item in metrics_rows)
              for key in ("o_vs_independent", "o_vs_fla", "state_vs_independent", "state_vs_fla")}
    result = dict(passed=True, cases=len(expected) * 4, head_chunk_rows=len(metrics_rows),
                  all_cache_output_prefix_bytes_equal_across_bd=True, max_relative_l2=maxima,
                  original_output_failures={str(bd): sum(not row["baseline_vs_independent"]["o"]["passed"]
                                                        for row in cases.values()) for bd, cases in rows.items()})
    write(output, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("grid", "profile", "compare", "forward"),
                        help="forward is a partial diagnostic only; grid/profile require cached-forward")
    parser.add_argument("--block-dim", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--case", default="all")
    parser.add_argument("--root", type=Path, help="Parent of grid-bd1 through grid-bd4 for compare")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()
    if args.mode == "compare":
        if args.root is None:
            parser.error("compare requires --root")
        compare(args.root, args.output)
        return
    import torch_npu

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = public.kda_fwd_kernels("stable")["recurrent"]
    assert Path(inspect.getsourcefile(selected.fn)).resolve() == (UNIT / "kernels/recurrent.py").resolve()
    assert selected.name == "kda_sub45_aqk_repaired_kernel"
    write(args.output / "identity.json", dict(sources=identities(), block_dim=args.block_dim,
        torch=torch.__version__, torch_npu=torch_npu.__version__,
        reference_device="cpu", execution_device="npu", candidate_entry="public stable",
        mode=args.mode, cached_forward_required=args.mode != "forward",
        budget=dict(rtol=.02, atol=.02, max_relative_l2=.05)))
    # Every vendor must register before the first custom-op execution.
    for label, call in (
        ("public stable forward", lambda: public._compiled_chain("a5", args.block_dim, "stable")),
        ("original negative control", lambda: compiled_chain(args.block_dim, "baseline")),
    ):
        print(f"compile {label}, bd={args.block_dim}", flush=True)
        call()
    if args.mode != "forward":
        print(f"compile public stable backward, bd={args.block_dim}", flush=True)
        backward._compiled_chain("a5", args.block_dim, "stable")
    if args.mode == "profile":
        profile(args.block_dim, args.output, args.warmup, args.repeat)
        return
    cases = grid_cases()
    if args.case in ("kimi_t1024", "kimi_t4096"):
        cases = [dict(id=args.case, B=1, H=32, HV=32, C=int(args.case.split("t")[-1]) // 64,
                      span=46., initial_state="random", seed=2026)]
    if args.case != "all":
        cases = [case for case in cases if case["id"] == args.case]
    if not cases:
        parser.error("Unknown case")
    rows = [verify(case, args.block_dim, args.output, with_caches=args.mode != "forward") for case in cases]
    write(args.output / "summary.json", dict(cases=len(rows), passed=all(r["passed"] for r in rows)))


if __name__ == "__main__":
    main()
