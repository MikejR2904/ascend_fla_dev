"""Recompute the BF-08 first-commit budget table using only the stdlib."""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = ROOT / "evidence/backward/calibration"
    budget_path = ROOT / "backward_budgets.json"
    budget = json.loads(budget_path.read_text())
    groups, coarse, records = {}, [], []
    for section, digest in budget["calibration_sha256"].items():
        path = evidence / (section + ".json")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        payload = json.loads(path.read_text())
        assert payload["complete"]
        coarse.extend(payload["negative_controls"])
        records.extend(payload["rows"])
        for row in payload["rows"]:
            if not row["ordinary_calibration_population"]:
                continue
            m = row["metrics"]
            g = groups.setdefault(row["key"], dict(ids=[], l2=0., element=0., ulp=0, dtype=row["output_dtype"]))
            g["ids"].append(row["id"])
            g["l2"] = max(g["l2"], m["relative_l2"])
            g["element"] = max(g["element"], m["max_relative_nonzero"])
            g["ulp"] = max(g["ulp"], m["rounded_reference_max_ulp"])
    assert set(groups) == set(budget["groups"])
    for key, got in groups.items():
        frozen = budget["groups"][key]
        assert frozen["calibration_case_ids"] == got["ids"]
        assert frozen["floor_relative_l2"] == got["l2"]
        assert frozen["floor_max_relative_nonzero"] == got["element"]
        assert frozen["old_to_rounded_fp64_max_ulp"] == got["ulp"]
        assert frozen["relative_l2_limit"] == min(.01, 3 * got["l2"])
        assert frozen["elementwise_relative_limit"] == 3 * got["element"]
        expected_ulp = 1 if got["dtype"] == "torch.bfloat16" and got["ulp"] <= 1 else None
        assert frozen["ulp_limit_each_reference"] == expected_ulp
    assert len(records) == 144 and len(groups) == 28 and len(coarse) == 168
    assert all(c["rejected_by_three_times_own_floor"] for c in coarse)
    # Re-evaluate coarse controls against the group-wide frozen limit too.
    by_id = {r["id"]: r for r in records}
    for control in coarse:
        key = by_id[control["case"]]["key"]
        assert control["relative_l2"] > budget["groups"][key]["relative_l2_limit"]
    controls = json.loads((evidence / "controls.json").read_text())
    assert controls["complete"] and controls["all_required_rejected"]
    assert controls["budgets_sha256"] == hashlib.sha256(budget_path.read_bytes()).hexdigest()
    assert controls["driver_sha256"] == hashlib.sha256((ROOT / "ref/backward_controls.py").read_bytes()).hexdigest()
    assert len(controls["controls"]) == 60 and len(controls["reference_checks"]) == 28
    boundary = []
    for row in controls["controls"]:
        g = budget["groups"][row["key"]]
        m = row["metrics"]
        assert row["l2_limit"] == g["relative_l2_limit"]
        assert row["relative_limit"] == g["elementwise_relative_limit"]
        assert m["relative_l2"] > g["relative_l2_limit"] or m["max_relative_nonzero"] > g["elementwise_relative_limit"] or (
            g["ulp_limit_each_reference"] is not None and m["rounded_reference_max_ulp"] > 1)
        if row["corruption"] == "scale_just_above_frozen_l2_limit":
            assert 1 < row["l2_to_limit"] < 2.1
            boundary.append(row)
    assert {r["key"] for r in boundary} == set(groups)
    for row in controls["reference_checks"]:
        assert row["analytic_vs_fp64_autograd"]["relative_l2"] < 1e-12
    print(json.dumps(dict(records=len(records), groups=len(groups), rejected_controls=len(coarse) + 60,
                          near_limit_controls=len(boundary), reference_checks=28,
                          scope="CPU calibration and frozen comparison criteria only", passed=True)))


if __name__ == "__main__":
    main()
