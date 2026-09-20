import sys, torch, pathlib, importlib.util
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location("gate_a2", sys.argv[1] + "/gate.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from ascriptor.backends.sim.launch import run_kernel
torch.manual_seed(0)
for (b, hv, c) in ((1, 1, 1), (1, 2, 2), (2, 3, 3)):
    t = c * 64
    g = -torch.rand(b, t, hv, 128) * 0.03 * 40
    gc = torch.full((b, hv, c, 64, 128), float("nan")); eg = torch.full_like(gc, float("nan"))
    out = run_kernel(m.kda_sub1_gate_a2_kernel, g.view(b * t, hv * 128), gc, eg, b, hv, c, block_dim=1, timeout=600, seed_outputs=True)
    ref_gc = g.reshape(b, c, 64, hv, 128).cumsum(2).permute(0, 3, 1, 2, 4)
    ref_eg = ref_gc.exp()
    o_gc, o_eg = out
    rl = lambda a, r: ((a.double() - r.double()).norm() / r.double().norm()).item()
    print(f"B{b} HV{hv} C{c}: g_cumsum bitwise={torch.equal(o_gc, ref_gc)} rel_l2={rl(o_gc, ref_gc):.3e} max_abs={(o_gc-ref_gc).abs().max().item():.3e} | "
          f"eg rel_l2={rl(o_eg, ref_eg):.3e} max_abs={(o_eg-ref_eg).abs().max().item():.3e} nan={int(o_eg.isnan().sum())}")
# sequential FP32 running sum (the kernel's definition) for the last shape
seq = g.reshape(b, c, 64, hv, 128).permute(0, 3, 1, 2, 4).clone()
for r in range(1, 64):
    seq[..., r, :] = seq[..., r, :] + seq[..., r - 1, :]
print("vs sequential fp32 cumsum: bitwise =", torch.equal(o_gc, seq), "max_abs =", (o_gc - seq).abs().max().item())
print("eg vs torch.exp(kernel g_cumsum): rel_l2 = %.3e" % rl(o_eg, o_gc.exp()))
