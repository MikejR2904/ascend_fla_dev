"""Verify complete, source-matched native grids and cross-block byte equality."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from repair_reference import grid_cases


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(root):
    by_bd, identities = {}, []
    expected = {case["id"] for case in grid_cases()}
    for bd in (1, 2, 3, 4):
        directory = root / f"grid-bd{bd}"
        summary = json.loads((directory / "summary.json").read_text())
        require(summary["passed"] and summary["cases"] == len(expected), f"Incomplete grid bd={bd}")
        identity = json.loads((directory / "identity.json").read_text())
        require(identity["budget"] == dict(rtol=.02, atol=.02, max_relative_l2=.05), "Budget changed")
        identities.append(identity["sources"])
        by_bd[bd] = {case: json.loads((directory / f"{case}.json").read_text()) for case in expected}
        require(all(row["passed"] for row in by_bd[bd].values()), f"Failed native case bd={bd}")
    require(all(identity == identities[0] for identity in identities), "Grid source identities differ")
    rows = []
    for case in grid_cases():
        case_id = case["id"]
        first = by_bd[1][case_id]
        for bd, cases in by_bd.items():
            row = cases[case_id]
            require(row["case"] == case and row["block_dim"] == bd, f"Case identity mismatch: {case_id}")
            require(row["variants"]["repaired"]["stage_sha256"] == first["variants"]["repaired"]["stage_sha256"],
                    f"Stage bytes differ across block_dim: {case_id}, bd={bd}")
            require(row["chunk_states_sha256"] == first["chunk_states_sha256"],
                    f"Chunk state bytes differ across block_dim: {case_id}, bd={bd}")
            require(row["baseline_bitwise"]["final_state"], f"Repair changed state: {case_id}, bd={bd}")
            require(all(value for name, value in row["baseline_bitwise"].items() if name != "o"),
                    f"Repair changed an intermediate: {case_id}, bd={bd}")
            if case["C"] % 2 == 0:
                require(all(row["baseline_bitwise"].values()), f"Even-C baseline differs: {case_id}, bd={bd}")
        metric = {}
        for oracle in ("independent", "fla"):
            for field in ("o", "state"):
                values = [r[f"{field}_vs_{oracle}"] for r in first["per_head_chunk"]]
                require(len(values) == case["B"] * case["HV"] * case["C"], f"Missing slice: {case_id}")
                require(all(v["passed"] for v in values), f"Slice failed: {case_id}")
                metric[f"{field}_vs_{oracle}"] = {
                    "max_head_chunk_relative_l2": max(v["relative_l2"] for v in values),
                    "max_abs_diff": max(v["max_abs_diff"] for v in values)}
        rows.append({"case": case, "metrics": metric,
                     "baseline_failure_bd": [bd for bd in by_bd if not by_bd[bd][case_id]["baseline_correct"]]})
    return {"passed": True, "cases": 4 * len(expected), "unique_inputs": len(expected),
            "block_dims": [1, 2, 3, 4], "sources": identities[0],
            "cross_bd_all_stage_and_chunk_state_bytes_equal": True,
            "even_c_all_stage_baseline_bytes_equal": True, "baseline_state_bytes_equal_all_cases": True,
            "baseline_failures": {bd: sum(not r["baseline_correct"] for r in cases.values()) for bd, cases in by_bd.items()},
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Contains grid-bd1 through grid-bd4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "sources")}))


if __name__ == "__main__":
    main()
