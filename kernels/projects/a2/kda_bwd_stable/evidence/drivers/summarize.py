"""Per-case relative L2 of the six gradients, from the device (or any launcher) receipts."""
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
OUT = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
contract = json.loads((pathlib.Path(sys.argv[2]) / "contract.json").read_text())
params = {c["id"]: (c["parameters"], c["block_dim"]) for c in contract["cases"]}

rows, failed = [], []
for d in sorted(root.iterdir()):
    s = d / "summary.json"
    if not s.is_dir() and s.exists():
        data = json.loads(s.read_text())
        for case in data.get("cases", []):
            cid = case.get("id") or d.name
            p, bd = params.get(cid, ({}, case.get("block_dim")))
            cmp = case.get("comparison", {})
            row = [cid, p.get("B", "?"), p.get("HV", "?"), p.get("C", "?"), bd]
            for name in OUT:
                e = cmp.get(name, {})
                row.append(e.get("relative_l2_error"))
                if e and not e.get("passed", True):
                    failed.append((cid, name, e.get("relative_l2_error")))
            rows.append(row)

hdr = ["case", "B", "HV", "C", "bd"] + [f"{n} relL2" for n in OUT]
w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
def fmt(v):
    return f"{v:.3e}" if isinstance(v, float) else str(v)
print(" | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
print("-|-".join("-" * w[i] for i in range(len(hdr))))
for r in rows:
    print(" | ".join(fmt(v).ljust(w[i]) for i, v in enumerate(r)))
print(f"\ncases: {len(rows)}   over-budget entries: {len(failed)}")
for f in failed:
    print("  FAILED", f)
for i, name in enumerate(OUT):
    vals = [r[5 + i] for r in rows if isinstance(r[5 + i], float)]
    if vals:
        print(f"{name:6s} relative L2  min={min(vals):.3e}  max={max(vals):.3e}")
