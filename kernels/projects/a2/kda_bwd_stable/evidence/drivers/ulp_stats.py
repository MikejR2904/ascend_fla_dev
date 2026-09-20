"""Per-checkpoint error distribution in units of the BF16 ulp, for the seven stage_outputs budgets.

BF16 keeps 8 total significand bits (1 implicit + 7 stored), so the ulp at a value x is
2^(floor(log2|x|) - 7). Reported: max, median, p99, and the fraction of elements above 1 ulp.
"""
import json, os, sys
sys.path.insert(0, os.environ["UNIT"]); os.chdir(os.environ["UNIT"])
import torch, unit as U

contract = json.load(open("contract.json"))
budgeted = sorted(contract["comparison"].get("stage_outputs", {}))
cases = [c for c in contract["cases"] if c["id"] in os.environ.get("CASES", "single_chunk_bd1,multi_chunk_bd1").split(",")]

def ulp(x):
    a = x.abs()
    e = torch.where(a > 0, torch.floor(torch.log2(a.clamp_min(1e-38))), torch.full_like(a, -126.0))
    return torch.pow(2.0, e - 7.0)

print(f"{'checkpoint':30s} {'case':18s} {'max_ulp':>9s} {'p99_ulp':>9s} {'>1ulp':>10s} {'rel_l2':>10s} {'|ref|max':>10s}")
worst = 0.0
worst_l2 = {}
for case in cases:
    inputs = U.make_inputs(case)
    ref = U.reference_stages(inputs)
    opts = {"device": "a2", "backend": "cce", "block_dim": case["block_dim"], "launcher": os.environ["LAUNCHER"],
            "timeout": 1800, "board": None, "out_dir": f"{os.environ['OUT']}/{case['id']}"}
    got = U.execute_stages(inputs, opts)
    for name in budgeted:
        a, b = got[name].float(), ref[name].float()
        d = (a - b).abs()
        u = d / ulp(b)
        u = u[torch.isfinite(u)]
        over = float((u > 1.0).float().mean())
        mx = float(u.max())
        worst = max(worst, mx)
        q = torch.quantile(u.flatten().float(), torch.tensor([0.5, 0.99]))
        l2 = float(d.pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(1e-30))
        worst_l2[name] = max(worst_l2.get(name, 0.0), l2)
        print(f"{name:30s} {case['id']:18s} {mx:9.3f} {float(q[1]):9.3f} "
              f"{over*100:9.4f}% {l2:10.3e} {float(b.abs().max()):10.3e}")
print(f"\nworst elementwise: {worst:.1f} ulp")
for k, v in sorted(worst_l2.items()):
    print(f"  worst relative L2 {k:30s} {v:.3e}")
