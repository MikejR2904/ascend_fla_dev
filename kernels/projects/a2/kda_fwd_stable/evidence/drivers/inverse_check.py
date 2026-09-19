import sys, torch, importlib.util, json, hashlib, pathlib
kd, launcher = sys.argv[1], sys.argv[2]
A5U = pathlib.Path(kd).parents[2] / "a5" / "kda_fwd_stable"; sys.path.insert(0, str(A5U))
import unit as a5unit
spec = importlib.util.spec_from_file_location("inv_a2", kd + "/triangular_inverse.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
K = M.tril_inverse64_a2_kernel
EXEC = {}
def run(strict, bd):
    b, hv, c = strict.shape[:3]
    inv = torch.full((b, hv, c, 64, 64), float("nan"), dtype=torch.bfloat16)
    args = (strict.contiguous(), inv, b, hv, c)
    if launcher == "sim":
        from ascriptor.backends.sim.launch import run_kernel
        return run_kernel(K, *args, block_dim=bd, timeout=3000, seed_outputs=True), None
    if launcher == "pipesim":
        from ascriptor.backends.sim.pipesim import simulate
        from ascriptor.passes import PIPELINE, PassManager
        from ascriptor.passes.autosync import check_balance
        low = PassManager(PIPELINE).run(K.ir()); bal = check_balance(low)
        s = simulate(low, args, block_dim=bd, timeout=3000, seed_outputs=True, check_gm=True)
        out = s.outputs[0] if isinstance(s.outputs, (list, tuple)) else s.outputs
        return out, dict(balance=[str(x) for x in bal], hazards=len(s.hazards), deadlock=bool(s.report.get("deadlock")), cycles=s.cycles,
                         hazard_samples=[str(h)[:260] for h in s.hazards[:3]])
    from ascriptor.runtime import OpExec
    ex = EXEC.get(bd) or EXEC.setdefault(bd, OpExec(K, launcher="aclnn", backend="cce", device="a2", block_dim=bd,
                                                  out_dir=pathlib.Path(sys.argv[3]) / f"inverse_bd{bd}", timeout=900, seed_outputs=True))
    return ex(*args), None
shapes = [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3)] if launcher != "aclnn" else \
         [(1, 1, 1, 1, 1.0, 1), (1, 1, 2, 2, 40.0, 2), (2, 2, 4, 3, 90.0, 3), (1, 1, 2, 1, 1.0, 5), (1, 16, 32, 2, 40.0, 6)]
for (b, h, hv, c, mult, seed) in shapes:
    x = a5unit.make_inputs({"parameters": {"B": b, "H": h, "HV": hv, "C": c, "gate_multiplier": mult}, "seed": seed})
    ref = a5unit.reference_stages(x); strict = ref["strict"].float(); akk_ref = ref["Akk"]
    # FP32 reference of the same inverse, for an error floor that is not itself BF16-rounded
    eye = torch.eye(64, dtype=torch.float64).expand(strict.shape).contiguous()
    akk64 = torch.linalg.solve_triangular(eye - strict.double(), eye, upper=False, unitriangular=True)
    for bd in ((1,) if launcher != "aclnn" else (1, 2)):
        out, extra = run(strict, bd)
        rl = lambda o, r: ((o.double() - r.double()).norm() / r.double().norm()).item()
        rec = dict(B=b, HV=hv, C=c, span_mult=mult, bd=bd, vs_ref_bf16_rel_l2=rl(out, akk_ref), vs_fp64_rel_l2=rl(out, akk64),
                   bf16_floor_rel_l2=rl(akk64.bfloat16(), akk64), max_abs_vs_fp64=float((out.double() - akk64).abs().max()),
                   nan=int(out.float().isnan().sum()), upper_nonzero=int((torch.triu(out.float(), 1) != 0).sum()),
                   diag_not_one=int((torch.diagonal(out.float(), dim1=-2, dim2=-1) != 1).sum()),
                   sha=hashlib.sha256(out.float().numpy().tobytes()).hexdigest()[:16])
        if extra: rec.update(extra)
        print("INVERSE", launcher, json.dumps(rec), flush=True)
