"""block_dim 1 vs 2 must be bitwise equal: core partitioning does not change the arithmetic.

AGENTS.md section 6 calls this out as a criterion that needs no reference. Run on the device.
"""
import json, os, sys, hashlib
sys.path.insert(0, os.environ["UNIT"]); os.chdir(os.environ["UNIT"])
import torch, unit as U

contract = json.load(open("contract.json"))
cases = {c["id"]: c for c in contract["cases"]}
OUT = ("dq", "dk", "dv", "dbeta", "dg", "dh0")
ids = os.environ.get("CASES", "grid_c2_hv4_bd1,odd_chunks_c3_bd1,batch2_grouped").split(",")
ok = True
for cid in ids:
    case = cases[cid]
    inputs = U.make_inputs(case)
    got = {}
    for bd in (1, 2):
        opts = {"device": "a2", "backend": "cce", "block_dim": bd, "launcher": os.environ["LAUNCHER"],
                "timeout": 1800, "board": None, "out_dir": f"{os.environ['OUT']}/{cid}_bd{bd}"}
        got[bd] = U.execute(inputs, opts)
    line = [cid]
    for name in OUT:
        a, b = got[1][name], got[2][name]
        same = bool(torch.equal(a, b))
        ok &= same
        h = hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]
        line.append(f"{name}={'bitwise' if same else 'DIFFER'}({h})")
    print(" ".join(line), flush=True)
print("ALL BITWISE EQUAL" if ok else "MISMATCH FOUND")
