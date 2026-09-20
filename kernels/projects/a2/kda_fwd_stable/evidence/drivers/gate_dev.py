import sys, torch, pathlib, importlib.util, json, hashlib
kd = sys.argv[1]; out_dir = pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("gate_a2", kd + "/gate.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from ascriptor.runtime import OpExec
torch.manual_seed(0)
res = []
for (b, hv, c) in ((1, 1, 1), (1, 2, 2), (2, 3, 3), (1, 32, 4)):
    t = c * 64
    g = -torch.rand(b, t, hv, 128) * 0.03 * 40
    seq = g.reshape(b, c, 64, hv, 128).permute(0, 3, 1, 2, 4).clone()
    for r in range(1, 64):
        seq[..., r, :] = seq[..., r, :] + seq[..., r - 1, :]
    for bd in (1, 2):
        gc = torch.full((b, hv, c, 64, 128), float("nan")); eg = torch.full_like(gc, float("nan"))
        ex = OpExec(m.kda_sub1_gate_a2_kernel, launcher="aclnn", backend="cce", device="a2", block_dim=bd,
                    out_dir=out_dir / f"gate_b{b}_hv{hv}_c{c}_bd{bd}", timeout=600, seed_outputs=True)
        o_gc, o_eg = ex(g.view(b * t, hv * 128), gc, eg, b, hv, c)
        ref_eg = seq.exp()
        rl = lambda a, r: ((a.double() - r.double()).norm() / r.double().norm()).item()
        rec = dict(B=b, HV=hv, C=c, bd=bd, gc_bitwise_seq=bool(torch.equal(o_gc, seq)), gc_max_abs=float((o_gc - seq).abs().max()),
                   eg_rel_l2=rl(o_eg, ref_eg), eg_max_abs=float((o_eg - ref_eg).abs().max()), nan=int(o_gc.isnan().sum() + o_eg.isnan().sum()),
                   gc_sha=hashlib.sha256(o_gc.numpy().tobytes()).hexdigest()[:16], eg_sha=hashlib.sha256(o_eg.numpy().tobytes()).hexdigest()[:16])
        print("GATE", json.dumps(rec), flush=True); res.append(rec)
json.dump(res, open(out_dir / "gate_device.json", "w"), indent=1)
