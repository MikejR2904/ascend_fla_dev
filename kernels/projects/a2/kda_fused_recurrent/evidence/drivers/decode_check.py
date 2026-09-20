import sys, torch, importlib.util, json, hashlib, pathlib
kd, launcher = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("step_a2", kd + "/step.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def ref(q, k, v, g, beta, h0, scale):
    """Independent float64 recurrence on the given (already BF16-rounded) inputs."""
    b, t, h, _ = q.shape; hv = v.shape[2]; grp = hv // h
    q, k, v, g, beta, S = (x.double() for x in (q, k, v, g, beta, h0))
    S = S.clone(); o = torch.zeros(b, t, hv, 128, dtype=torch.float64)
    for bi in range(b):
        for j in range(hv):
            hh = j // grp
            for tt in range(t):
                S[bi, j] = S[bi, j] * g[bi, tt, j].exp()[:, None]
                d = v[bi, tt, j] - k[bi, tt, hh] @ S[bi, j]
                S[bi, j] = S[bi, j] + beta[bi, tt, j] * torch.outer(k[bi, tt, hh], d)
                o[bi, tt, j] = (q[bi, tt, hh] * scale) @ S[bi, j]
    return o, S

EXEC = {}
def run(q, k, v, g, beta, h0, scale, bd, out_dir):
    b, t, h, _ = q.shape; hv = v.shape[2]
    o = torch.full((b * t, hv * 128), float("nan"), dtype=torch.bfloat16); fs = torch.full((b, hv, 128, 128), float("nan"))
    args = (q.view(b * t, h * 128), k.view(b * t, h * 128), v.view(b * t, hv * 128), g.view(b * t, hv * 128),
            beta.view(b * t, hv), h0, o, fs, b, t, h, hv, scale)
    if launcher == "sim":
        from ascriptor.backends.sim.launch import run_kernel
        oo, ff = run_kernel(m.kda_fused_recurrent_a2_kernel, *args, block_dim=bd, timeout=1800, seed_outputs=True)
    else:
        from ascriptor.runtime import OpExec
        ex = EXEC.get(bd) or EXEC.setdefault(bd, OpExec(m.kda_fused_recurrent_a2_kernel, launcher="aclnn", backend="cce", device="a2",
                    block_dim=bd, out_dir=pathlib.Path(sys.argv[3]) / f"decode_bd{bd}", timeout=900, seed_outputs=True))
        oo, ff = ex(*args)
    return oo.view(b, t, hv, 128), ff

torch.manual_seed(1)
shapes = [(1, 1, 1, 1), (1, 1, 2, 4), (2, 4, 2, 2), (1, 8, 4, 8)] if launcher == "sim" else [(1, 1, 1, 1), (1, 1, 2, 4), (2, 4, 2, 2), (1, 8, 4, 8), (1, 1, 16, 32), (2, 16, 4, 8)]
for (b, t, h, hv) in shapes:
    q = (torch.randn(b, t, h, 128) * 0.1).bfloat16(); k = torch.nn.functional.normalize(torch.randn(b, t, h, 128), dim=-1).bfloat16()
    v = (torch.randn(b, t, hv, 128) * 0.1).bfloat16(); g = -torch.rand(b, t, hv, 128) * 0.5
    beta = torch.rand(b, t, hv) * 0.9 + 0.05; h0 = torch.randn(b, hv, 128, 128) * 0.05; scale = 128 ** -0.5
    ro, rs = ref(q, k, v, g, beta, h0, scale)
    for bd in ((1,) if launcher == "sim" else (1, 2)):
        oo, ff = run(q, k, v, g, beta, h0, scale, bd, f"{sys.argv[3]}/dec_b{b}_t{t}_h{h}_hv{hv}_bd{bd}" if len(sys.argv) > 3 else "")
        rl = lambda a, r: ((a.double() - r).norm() / r.norm()).item()
        rec = dict(B=b, T=t, H=h, HV=hv, bd=bd, o_rel_l2=rl(oo, ro), o_max_abs=float((oo.double() - ro).abs().max()),
                   o_bf16_floor=rl(ro.bfloat16(), ro), state_rel_l2=rl(ff, rs), state_max_abs=float((ff.double() - rs).abs().max()),
                   nan=int(oo.float().isnan().sum() + ff.isnan().sum()), o_sha=hashlib.sha256(oo.float().numpy().tobytes()).hexdigest()[:16],
                   state_sha=hashlib.sha256(ff.numpy().tobytes()).hexdigest()[:16])
        print("DECODE", launcher, json.dumps(rec), flush=True)
