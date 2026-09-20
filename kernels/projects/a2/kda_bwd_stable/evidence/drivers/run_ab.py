import json, os, sys
unit = os.environ["UNIT"]; sys.path.insert(0, unit); os.chdir(unit)
import torch, unit as U
contract = json.load(open("contract.json"))
case = next(c for c in contract["cases"] if c["id"] == os.environ["CASE"])
inputs = U.make_inputs(case); ref = U.reference_stages(inputs)
opts = {"device": "a2", "backend": "cce", "block_dim": case["block_dim"], "launcher": "aclnn",
        "timeout": 1800, "board": None, "out_dir": os.environ["OUT"]}
got = U.execute_stages(inputs, opts)
print(f"variant={os.environ['VARIANT']} case={os.environ['CASE']}")
for name in ("finalize_pair.qk_left", "finalize_pair.qk_right"):
    a, b = got[name].float(), ref[name].float()
    d = (a - b).abs()
    print(f"  {name:26s} max_abs={float(d.max()):.6e} rel_l2="
          f"{float(d.pow(2).sum().sqrt()/b.pow(2).sum().sqrt()):.3e}")
    n = d.shape[2]
    for c0 in range(0, n, 64):
        sl = d.narrow(2, c0, min(64, n - c0))
        print(f"      chunk rows {c0:4d}-{c0+sl.shape[2]-1:4d}: max_abs={float(sl.max()):.6e}")
