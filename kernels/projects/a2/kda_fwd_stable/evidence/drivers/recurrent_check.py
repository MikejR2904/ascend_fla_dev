import sys, torch, importlib.util, json, hashlib, pathlib
kd, launcher = sys.argv[1], sys.argv[2]
A5U = pathlib.Path(kd).parents[2] / "a5" / "kda_fwd_stable"; sys.path.insert(0, str(A5U))
import unit as a5unit
spec = importlib.util.spec_from_file_location("rec_a2", kd + "/recurrent.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
K = M.kda_sub45_a2_kernel
bf = lambda t: t.to(torch.bfloat16).to(torch.float64)
def emulate(q, st, h0, scale):
    """float64 emulation of this stage with the kernel's BF16 rounding points."""
    b, t, h, _ = q.shape; hv = st["w"].shape[1]; c = t // 64; grp = hv // h
    qd = q.double().reshape(b, c, 64, h, 128)
    o = torch.zeros(b, t, hv, 128, dtype=torch.float64); S_all = h0.double().clone()
    for bi in range(b):
        for j in range(hv):
            S = S_all[bi, j]
            for ci in range(c):
                eg = st["eg"][bi, j, ci].double(); w = st["w"][bi, j, ci].double(); u = st["u"][bi, j, ci].double()
                kg = st["kg"][bi, j, ci].double(); aqk = st["Aqk"][bi, j, ci].double()
                h_ = bf(S); qg = bf(qd[bi, ci, :, j // grp] * eg * scale)
                vnew = bf(u - w @ h_)
                sdec = S * eg[63][:, None]; delta = kg.T @ vnew
                o[bi, ci * 64:(ci + 1) * 64, j] = bf(qg @ h_ + aqk @ vnew)
                S = sdec + delta
            S_all[bi, j] = S
    return o, S_all
EXEC = {}
def run(x, st, bd):
    b, t, h, _ = x["q"].shape; hv = x["v"].shape[2]; c = t // 64
    o = torch.full((b * t, hv * 128), float("nan"), dtype=torch.bfloat16); fs = torch.full((b, hv, 128, 128), float("nan"))
    args = (x["q"].view(b * t, h * 128), st["Aqk"].contiguous(), st["kg"].contiguous(), st["w"].contiguous(), st["u"].contiguous(),
            st["eg"].float().contiguous(), x["initial_state"], o, fs, b, h, hv, c, 128 ** -0.5)
    if launcher == "sim":
        from ascriptor.backends.sim.launch import run_kernel
        r = run_kernel(K, *args, block_dim=bd, timeout=3000, seed_outputs=True); return (r[0].view(b, t, hv, 128), r[1]), None
    if launcher == "pipesim":
        from ascriptor.backends.sim.pipesim import simulate
        from ascriptor.passes import PIPELINE, PassManager
        from ascriptor.passes.autosync import check_balance
        low = PassManager(PIPELINE).run(K.ir()); bal = check_balance(low)
        s = simulate(low, args, block_dim=bd, timeout=3000, seed_outputs=True, check_gm=True)
        return (s.outputs[0].view(b, t, hv, 128), s.outputs[1]), dict(balance=[str(x) for x in bal], hazards=len(s.hazards),
                deadlock=bool(s.report.get("deadlock")), cycles=s.cycles, hazard_samples=[str(h)[:260] for h in s.hazards[:3]])
    from ascriptor.runtime import OpExec
    ex = EXEC.get(bd) or EXEC.setdefault(bd, OpExec(K, launcher="aclnn", backend="cce", device="a2", block_dim=bd,
                                                  out_dir=pathlib.Path(sys.argv[3]) / f"recurrent_bd{bd}", timeout=900, seed_outputs=True))
    r = ex(*args); return (r[0].view(b, t, hv, 128), r[1]), None
shapes = [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3)] if launcher != "aclnn" else \
         [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3), (1, 1, 2, 1, 1.0, 5), (1, 1, 1, 5, 1.0, 7), (1, 16, 32, 2, 40.0, 6)]
for (b, h, hv, c, mult, seed) in shapes:
    x = a5unit.make_inputs({"parameters": {"B": b, "H": h, "HV": hv, "C": c, "gate_multiplier": mult}, "seed": seed})
    st = a5unit.reference_stages(x); full = a5unit.reference(x)
    eo, es = emulate(x["q"], st, x["initial_state"], 128 ** -0.5)
    for bd in ((1,) if launcher != "aclnn" else (1, 2)):
        (o, fs), extra = run(x, st, bd)
        rl = lambda a, r: ((a.double() - r.double()).norm() / r.double().norm()).item()
        rec = dict(B=b, H=h, HV=hv, C=c, span_mult=mult, bd=bd, o_vs_emul=rl(o, eo), state_vs_emul=rl(fs, es),
                   o_vs_fp32_ref=rl(o, full["o"]), state_vs_fp32_ref=rl(fs, full["final_state"]),
                   o_max_abs_vs_ref=float((o.double() - full["o"].double()).abs().max()),
                   nan=int(o.float().isnan().sum() + fs.isnan().sum()),
                   o_sha=hashlib.sha256(o.float().numpy().tobytes()).hexdigest()[:12], s_sha=hashlib.sha256(fs.numpy().tobytes()).hexdigest()[:12])
        if extra: rec.update(extra)
        print("RECURRENT", launcher, json.dumps(rec), flush=True)
