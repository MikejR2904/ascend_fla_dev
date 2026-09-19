import sys, torch, importlib.util, json, hashlib, pathlib
kd, launcher = sys.argv[1], sys.argv[2]
A5U = pathlib.Path(kd).parents[2] / "a5" / "kda_fwd_stable"
sys.path.insert(0, str(A5U))
import unit as a5unit  # CPU stage references only (no compiler import)
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, kd + "/" + file); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
S = load("intra_a2", "intra.py")
def case(b, h, hv, c, mult, seed):
    return a5unit.make_inputs({"parameters": {"B": b, "H": h, "HV": hv, "C": c, "gate_multiplier": mult}, "seed": seed})
def seq_gc(g, b, hv, c):
    gc = g.reshape(b, c, 64, hv, 128).permute(0, 3, 1, 2, 4).clone()
    for r in range(1, 64): gc[..., r, :] = gc[..., r, :] + gc[..., r - 1, :]
    return gc.contiguous()
EXEC = {}
def run(x, bd):
    b, t, h, _ = x["q"].shape; hv = x["v"].shape[2]; c = t // 64
    gc = seq_gc(x["g_raw"], b, hv, c)
    aqk = torch.full((b, hv, c, 64, 64), float("nan"), dtype=torch.bfloat16); st = torch.full((b, hv, c, 64, 64), float("nan"))
    args = (x["q"].view(b * t, h * 128), x["k"].view(b * t, h * 128), gc, x["beta"].view(b * t, hv), aqk, st, b, h, hv, c, 128 ** -0.5)
    if launcher in ("sim", "pipesim"):
        if launcher == "sim":
            from ascriptor.backends.sim.launch import run_kernel
            return run_kernel(S.kda_sub2_score_a2_kernel, *args, block_dim=bd, timeout=3000, seed_outputs=True), None
        from ascriptor.backends.sim.pipesim import simulate
        from ascriptor.passes import PIPELINE, PassManager
        from ascriptor.passes.autosync import check_balance
        low = PassManager(PIPELINE).run(S.kda_sub2_score_a2_kernel.ir()); bal = check_balance(low)
        sim = simulate(low, args, block_dim=bd, timeout=3000, seed_outputs=True, check_gm=True)
        return sim.outputs, dict(balance=[str(x) for x in bal], hazards=len(sim.hazards), deadlock=bool(sim.report.get("deadlock")), cycles=sim.cycles,
                                 hazard_samples=[str(h)[:200] for h in sim.hazards[:2]])
    from ascriptor.runtime import OpExec
    ex = EXEC.get(bd) or EXEC.setdefault(bd, OpExec(S.kda_sub2_score_a2_kernel, launcher="aclnn", backend="cce", device="a2", block_dim=bd,
                                                  out_dir=pathlib.Path(sys.argv[3]) / f"scores_bd{bd}", timeout=900, seed_outputs=True))
    return ex(*args), None
shapes = [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3)] if launcher != "aclnn" else \
         [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3), (1, 1, 1, 2, 1.0, 4), (1, 1, 2, 1, 1.0, 5), (1, 16, 32, 2, 40.0, 6)]
for (b, h, hv, c, mult, seed) in shapes:
    x = case(b, h, hv, c, mult, seed); ref = a5unit.reference_stages(x)
    for bd in ((1,) if launcher != "aclnn" else (1, 2)):
        out, extra = run(x, bd); a, s = out
        rl = lambda o, r: ((o.double() - r.double()).norm() / r.double().norm()).item()
        rec = dict(B=b, H=h, HV=hv, C=c, span_mult=mult, bd=bd, aqk_rel_l2=rl(a, ref["Aqk"]), aqk_max_abs=float((a.float() - ref["Aqk"].float()).abs().max()),
                   strict_rel_l2=rl(s, ref["strict"]), strict_max_abs=float((s - ref["strict"]).abs().max()),
                   nan=int(a.float().isnan().sum() + s.isnan().sum()), upper_nonzero=int((torch.triu(a.float(), 1) != 0).sum() + (torch.triu(s, 0) != 0).sum()),
                   aqk_sha=hashlib.sha256(a.float().numpy().tobytes()).hexdigest()[:16], strict_sha=hashlib.sha256(s.numpy().tobytes()).hexdigest()[:16])
        if extra: rec.update(extra)
        print("SCORES", launcher, json.dumps(rec), flush=True)
