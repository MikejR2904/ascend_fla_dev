"""Real-weights validation of the GDN-2 a2 training op — config auto-inferred from
the checkpoint state-dict (these repos ship no config.json). Strict-loads (proves
ABI), captures real per-head q,k,v,g,b,w at real layers, drives the fp32 oracle and
my deployed autograd.Function, compares forward + all 7 gradients.

    ASCEND_RT_VISIBLE_DEVICES=<d> python real_weights_validate2.py <ckpt.pth>
"""
import sys, glob, re, importlib.util
import torch, torch_npu  # noqa
import numpy as np
import pathlib as _pl
_REPO = str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "ascend_fla").is_dir()))
sys.path.insert(0, _REPO)
from ascend_fla.models import GDN2Config, GDN2ForCausalLM
from ascend_fla.reference.gdn2 import gdn2_recurrent_reference

CKPT = sys.argv[1]
DEV = "npu:0"

_spec = importlib.util.spec_from_file_location(
    "gdn2_ag", _REPO + "/kernels/projects/a2/gdn2_recurrent_train/autograd.py")
_ag = importlib.util.module_from_spec(_spec); sys.modules["gdn2_ag"] = _ag; _spec.loader.exec_module(_ag)
gdn2_recurrent = _ag.gdn2_recurrent

# ---- infer config from state-dict shapes ----
raw = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
vocab, n_embd = sd["transformer.wte.weight"].shape
key_dim = sd["transformer.h.0.attn.q_proj.weight"].shape[0]
value_dim = sd["transformer.h.0.attn.v_proj.weight"].shape[0]
num_heads = sd["transformer.h.0.attn.A_log"].shape[0]
head_dim = key_dim // num_heads
inter = sd["transformer.h.0.mlp.swiglu.w1.weight"].shape[0]
nl = max(int(m.group(1)) for k in sd for m in [re.search(r"transformer\.h\.(\d+)\.", k)] if m) + 1
cfg = GDN2Config(vocab_size=int(vocab), n_embd=int(n_embd), n_layer=int(nl),
                 intermediate_size=int(inter), num_heads=int(num_heads),
                 num_v_heads=value_dim // head_dim, head_dim=int(head_dim), conv_size=4)
print(f"=== inferred config: vocab {vocab} n_embd {n_embd} n_layer {nl} "
      f"num_heads {num_heads} head_dim {head_dim} value_dim {value_dim} inter {inter} ===")

print(f"=== loading {CKPT} strict=True ===")
model = GDN2ForCausalLM.from_checkpoint(CKPT, config=cfg, device="cpu",
                                        dtype=torch.float32, strict=True, core_backend="torch")
model.eval()
print("  strict load OK  params:", model.parameter_count)

CAP = {}
layers = model.transformer.h
for li in (0, nl // 2, nl - 1):
    mixer = layers[li].attn
    orig = mixer._run_core

    def make(orig, li):
        def hook(q, k, v, g, b, w, s):
            CAP[li] = tuple(x.detach().clone() for x in (q, k, v, g, b, w)); return orig(q, k, v, g, b, w, s)
        return hook
    mixer._run_core = make(orig, li)

torch.manual_seed(7)
with torch.inference_mode():
    model(torch.randint(0, int(vocab), (1, 64), dtype=torch.long))


def cmp(name, got, ref, tol=2e-3):
    g = got.detach().cpu().float().reshape(-1).numpy(); r = ref.detach().cpu().float().reshape(-1).numpy()
    rel = np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-30); ok = np.isfinite(g).all() and rel < tol
    print(f"    {name:4s} relL2 {rel:.3e}  {'OK' if ok else 'FAIL'}"); return ok


allok = True
for li in sorted(CAP):
    q, k, v, g, b, w = CAP[li]; B, T, H, K = q.shape; V = v.shape[-1]
    print(f"=== layer {li}: B{B} T{T} H{H} K{K} V{V}  g[{g.min():.2f},{g.max():.2f}] "
          f"b[{b.min():.3f},{b.max():.3f}] w[{w.min():.3f},{w.max():.3f}] ===")
    S0 = torch.zeros(B, H, K, V)
    leaves = [t.clone().double().requires_grad_(True) for t in (q, k, v, g, b, w)]
    S0d = S0.clone().double().requires_grad_(True)
    o_ref, sf_ref = gdn2_recurrent_reference(*leaves, S0d, scale=128 ** -0.5, use_qk_l2norm=True, qk_norm_eps=1e-6)
    do = torch.randn(B, T, H, V, dtype=torch.float64); dSf = torch.randn(B, H, K, V, dtype=torch.float64)
    (o_ref * do).sum().add((sf_ref * dSf).sum()).backward()
    gref = {n: t.grad for n, t in zip(("dq", "dk", "dv", "dg", "db", "dw", "dh0"), leaves + [S0d])}
    dv_ = lambda t: t.clone().float().to(DEV).requires_grad_(True)
    qn, kn, vn, gn, bn, wn = (dv_(t) for t in (q, k, v, g, b, w)); S0n = dv_(S0)
    o, sf = gdn2_recurrent(qn, kn, vn, gn, bn, wn, S0n)
    (o * do.float().to(DEV)).sum().add((sf * dSf.float().to(DEV)).sum()).backward()
    gmy = {"dq": qn.grad, "dk": kn.grad, "dv": vn.grad, "dg": gn.grad, "db": bn.grad, "dw": wn.grad, "dh0": S0n.grad}
    allok &= cmp("o", o, o_ref); allok &= cmp("Sf", sf, sf_ref)
    for n in ("dq", "dk", "dv", "dg", "db", "dw", "dh0"):
        allok &= cmp(n, gmy[n], gref[n])
print("REAL-WEIGHTS VALIDATION", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
