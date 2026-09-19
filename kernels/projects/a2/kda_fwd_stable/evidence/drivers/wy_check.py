import sys, torch, importlib.util, json, hashlib, pathlib
kd, launcher = sys.argv[1], sys.argv[2]
A5U = pathlib.Path(kd).parents[2] / "a5" / "kda_fwd_stable"; sys.path.insert(0, str(A5U))
import unit as a5unit
spec = importlib.util.spec_from_file_location("wy_a2", kd + "/wy.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
K = M.kda_sub3_wy_a2_kernel
EXEC = {}
def seq_gc(g, b, hv, c):
    gc = g.reshape(b, c, 64, hv, 128).permute(0, 3, 1, 2, 4).clone()
    for r in range(1, 64): gc[..., r, :] = gc[..., r, :] + gc[..., r - 1, :]
    return gc.contiguous()
def run(x, akk, bd):
    b, t, h, _ = x["q"].shape; hv = x["v"].shape[2]; c = t // 64
    gc = seq_gc(x["g_raw"], b, hv, c)
    outs = [torch.full((b, hv, c, 64, 128), float("nan"), dtype=torch.bfloat16) for _ in range(4)]
    args = (x["q"].view(b * t, h * 128), x["k"].view(b * t, h * 128), x["v"].view(b * t, hv * 128), x["beta"].view(b * t, hv),
            akk.contiguous(), gc, *outs, b, h, hv, c)
    if launcher == "sim":
        from ascriptor.backends.sim.launch import run_kernel
        return run_kernel(K, *args, block_dim=bd, timeout=3000, seed_outputs=True), None
    if launcher == "pipesim":
        from ascriptor.backends.sim.pipesim import simulate
        from ascriptor.passes import PIPELINE, PassManager
        from ascriptor.passes.autosync import check_balance
        low = PassManager(PIPELINE).run(K.ir()); bal = check_balance(low)
        s = simulate(low, args, block_dim=bd, timeout=3000, seed_outputs=True, check_gm=True)
        return s.outputs, dict(balance=[str(x) for x in bal], hazards=len(s.hazards), deadlock=bool(s.report.get("deadlock")), cycles=s.cycles,
                               hazard_samples=[str(h)[:260] for h in s.hazards[:3]])
    from ascriptor.runtime import OpExec
    ex = EXEC.get(bd) or EXEC.setdefault(bd, OpExec(K, launcher="aclnn", backend="cce", device="a2", block_dim=bd,
                                                  out_dir=pathlib.Path(sys.argv[3]) / f"wy_bd{bd}", timeout=900, seed_outputs=True))
    return ex(*args), None
shapes = [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3)] if launcher != "aclnn" else \
         [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3), (1, 1, 2, 1, 1.0, 5), (1, 16, 32, 2, 40.0, 6)]
for (b, h, hv, c, mult, seed) in shapes:
    x = a5unit.make_inputs({"parameters": {"B": b, "H": h, "HV": hv, "C": c, "gate_multiplier": mult}, "seed": seed})
    ref = a5unit.reference_stages(x)
    for bd in ((1,) if launcher != "aclnn" else (1, 2)):
        out, extra = run(x, ref["Akk"], bd)
        rl = lambda o, r: ((o.double() - r.double()).norm() / r.double().norm()).item()
        rec = dict(B=b, H=h, HV=hv, C=c, span_mult=mult, bd=bd, nan=sum(int(o.float().isnan().sum()) for o in out))
        for name, o in zip(("w", "u", "qg", "kg"), out):
            rec[f"{name}_rel_l2"] = rl(o, ref[name]); rec[f"{name}_sha"] = hashlib.sha256(o.float().numpy().tobytes()).hexdigest()[:12]
        if extra: rec.update(extra)
        print("WY", launcher, json.dumps(rec), flush=True)
