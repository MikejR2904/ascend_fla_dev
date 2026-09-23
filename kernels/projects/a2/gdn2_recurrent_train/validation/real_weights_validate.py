"""Real-weights validation of the GDN-2 a2 training op against the LLM-OS-Models
checkpoints. Strict-loads the checkpoint (proves the architecture ABI, since these
repos ship no config.json), runs a real forward capturing the actual per-head
q,k,v,g,b,w that enter the recurrence, then drives both the repo's fp32 oracle and
my deployed autograd.Function on those real tensors and compares forward + all 7
gradients.

Usage:  ASCEND_RT_VISIBLE_DEVICES=<d> python real_weights_validate.py <ckpt.pth> <config-preset>
        config-preset in {370m, 1.3b}
"""
import sys, glob, importlib.util
import torch, torch_npu  # noqa
import numpy as np

import pathlib as _pl
_REPO = str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "ascend_fla").is_dir()))
sys.path.insert(0, _REPO)
from ascend_fla.models import GDN2Config, GDN2ForCausalLM
from ascend_fla.reference.gdn2 import gdn2_recurrent_reference

CKPT = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: %s <checkpoint.pth>" % sys.argv[0])
PRESET = sys.argv[2] if len(sys.argv) > 2 else "370m"
DEV = "npu:0"

# --- deployed training op (compiles both a2 kernels) ---
_spec = importlib.util.spec_from_file_location(
    "gdn2_ag", _REPO + "/kernels/projects/a2/gdn2_recurrent_train/autograd.py")
_ag = importlib.util.module_from_spec(_spec); sys.modules["gdn2_ag"] = _ag; _spec.loader.exec_module(_ag)
gdn2_recurrent = _ag.gdn2_recurrent


def cfg_370m():
    return GDN2Config(vocab_size=32000, n_embd=1024, n_layer=16, intermediate_size=2048,
                      num_heads=16, num_v_heads=16, head_dim=128, conv_size=4)


CONFIG = cfg_370m() if PRESET == "370m" else GDN2Config.gdn2_1_3b()

# ---- Part A: strict checkpoint load (architecture ABI confirmation) ----
print(f"=== loading {CKPT} strict=True (preset {PRESET}) ===")
model = GDN2ForCausalLM.from_checkpoint(CKPT, config=CONFIG, device="cpu",
                                        dtype=torch.float32, strict=True, core_backend="torch")
model.eval()
print("  strict load OK  params:", model.parameter_count,
      " n_layer:", CONFIG.n_layer, " num_heads:", CONFIG.num_heads, " head_dim:", CONFIG.head_dim)

# ---- Part B: real forward, capture real q,k,v,g,b,w at chosen layers ----
CAP = {}
layers = model.transformer.h if hasattr(model.transformer, "h") else model.transformer.layers
for li in (0, CONFIG.n_layer // 2):
    mixer = layers[li].attn
    orig = mixer._run_core

    def make(orig, li):
        def hook(q, k, v, g, b, w, initial_state):
            CAP[li] = tuple(x.detach().clone() for x in (q, k, v, g, b, w))
            return orig(q, k, v, g, b, w, initial_state)
        return hook
    mixer._run_core = make(orig, li)

torch.manual_seed(7)
input_ids = torch.randint(0, CONFIG.vocab_size, (1, 64), dtype=torch.long)
with torch.inference_mode():
    model(input_ids)
print("  captured layers:", sorted(CAP), " per-head shapes q/v:",
      tuple(CAP[0][0].shape), tuple(CAP[0][2].shape))


def compare(name, got, ref, tol=2e-3):
    g = got.detach().cpu().float().reshape(-1).numpy(); r = ref.detach().cpu().float().reshape(-1).numpy()
    rel = np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-30)
    ok = np.isfinite(g).all() and rel < tol
    print(f"    {name:4s} relL2 {rel:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


allok = True
for li in sorted(CAP):
    q, k, v, g, b, w = CAP[li]
    B, T, H, K = q.shape; V = v.shape[-1]
    print(f"=== layer {li}: real-weights recurrence  B{B} T{T} H{H} K{K} V{V} ===")
    print(f"    g range [{g.min():.3f},{g.max():.3f}] b [{b.min():.3f},{b.max():.3f}] "
          f"w [{w.min():.3f},{w.max():.3f}] q.norm {q.norm(dim=-1).mean():.3f}")
    S0 = torch.zeros(B, H, K, V, dtype=torch.float32)

    # reference oracle (fp32), leaves for autograd
    leaves = [t.clone().double().requires_grad_(True) for t in (q, k, v, g, b, w)]
    S0d = S0.clone().double().requires_grad_(True)
    o_ref, sf_ref = gdn2_recurrent_reference(*leaves, S0d, scale=128 ** -0.5,
                                             use_qk_l2norm=True, qk_norm_eps=1e-6)
    do = torch.randn(B, T, H, V, dtype=torch.float64); dSf = torch.randn(B, H, K, V, dtype=torch.float64)
    (o_ref * do).sum().add((sf_ref * dSf).sum()).backward()
    gref = {n: t.grad for n, t in zip(("dq", "dk", "dv", "dg", "db", "dw", "dh0"), leaves + [S0d])}

    # my deployed op on device
    dv_ = lambda t: t.clone().float().to(DEV).requires_grad_(True)
    qn, kn, vn, gn, bn, wn = (dv_(t) for t in (q, k, v, g, b, w))
    S0n = dv_(S0)
    o, sf = gdn2_recurrent(qn, kn, vn, gn, bn, wn, S0n)
    (o * do.float().to(DEV)).sum().add((sf * dSf.float().to(DEV)).sum()).backward()
    gmy = {"dq": qn.grad, "dk": kn.grad, "dv": vn.grad, "dg": gn.grad,
           "db": bn.grad, "dw": wn.grad, "dh0": S0n.grad}

    allok &= compare("o", o, o_ref)
    allok &= compare("Sf", sf, sf_ref)
    for n in ("dq", "dk", "dv", "dg", "db", "dw", "dh0"):
        allok &= compare(n, gmy[n], gref[n])

print("REAL-WEIGHTS VALIDATION", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
