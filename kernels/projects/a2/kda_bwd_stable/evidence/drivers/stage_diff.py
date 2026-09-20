import json, sys, os
sys.path.insert(0, os.environ["UNIT"])
os.chdir(os.environ["UNIT"])
import torch, unit as U
contract = json.load(open("contract.json"))
case = next(c for c in contract["cases"] if c["id"] == os.environ.get("CASE", "single_chunk_bd1"))
inputs = U.make_inputs(case)
ref = U.reference_stages(inputs)
opts = {"device": "a2", "backend": "cce", "block_dim": case["block_dim"],
        "launcher": os.environ.get("LAUNCHER", "aclnn"), "timeout": 1800,
        "output": os.environ.get("OUT"), "board": None, "out_dir": os.environ.get("OUT")}
got = U.execute_stages(inputs, opts)
rows = []
for name in ref:
    a, b = got[name].float(), ref[name].float()
    d = (a - b).abs()
    l2 = float(d.pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(1e-30))
    rows.append((name, float(d.max()), l2))
for name, mx, l2 in rows:
    flag = "  <<<" if mx > 1e-3 else ""
    print(f"{name:34s} max_abs={mx:.3e} rel_l2={l2:.3e}{flag}")
