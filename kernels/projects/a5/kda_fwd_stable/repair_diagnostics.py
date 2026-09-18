"""Reduced A2-04 pipe-model regression using the shipped repaired source.

The unchanged upstream and qg-only mutation are intentional negative controls.
They run on CPU models only. They must retain the odd-C handoff hazard.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

UNIT = Path(__file__).resolve().parent
REPO = UNIT.parents[3]
sys.path.insert(0, str(REPO))

import torch

from repair_runtime import selected_kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("aqk_original_diagnosis", REPO / "benchmarks/diag_c1_multihead.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    unit = diag.unit_root(None)
    variants = {name: diag.load_kernel(unit, name) for name in ("upstream", "qg-only")}
    fixed = selected_kernels("repaired")["recurrent"]
    lines = (UNIT / "kernels/recurrent.py").read_text().splitlines()
    anchors = {"write": next(i + 1 for i, line in enumerate(lines) if diag.AQK_WRITE in line),
               "read": next(i + 1 for i, line in enumerate(lines) if diag.AQK_READ in line)}
    variants["repaired"] = fixed, anchors
    rows = []
    passed = True
    for c in (1, 2, 3, 5):
        results = {}
        for name, (kernel, locations) in variants.items():
            result = diag.run_case(kernel, locations, B=1, HV=2, C=c, bd=1,
                                   pipesim=True, seed=2026, timeout=180)
            results[name] = result
            hazards = len(result["hazards"])
            correct = all(h["replay_bitwise"] for h in result["heads"])
            expected_clean = name == "repaired" or c % 2 == 0
            ok = (hazards == 0 and correct and not result["deadlock"]) if expected_clean else (hazards > 0 and not correct)
            row = {"variant": name, "C": c, "B": 1, "H": 1, "HV": 2, "block_dim": 1,
                   "hazard_count": hazards, "replay_correct": correct, "deadlock": result["deadlock"],
                   "heads": result["heads"], "reads": result["reads"], "events": result["events"],
                   "replay_self_check": result["replay_self_check"], "passed": ok}
            rows.append(row)
            passed &= ok
            print(json.dumps({k: v for k, v in row.items() if k not in ("reads", "events", "heads")}), flush=True)
        if c % 2 == 0:
            for key in ("o", "final_state", "replay"):
                passed &= torch.equal(results["repaired"][key], results["upstream"][key])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "reduced pipe model, not silicon", "rows": rows,
                                       "passed": passed}, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
