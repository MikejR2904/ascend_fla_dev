"""Verify the (previously crashing) T=1 backward: correctness vs golden, and through
the autograd.Function fwd+bwd path (the exact path that raised the vector-core
exception before the delta-checkpoint change)."""
import sys, os, importlib.util
import torch, torch_npu  # noqa
import numpy as np
D = _REPO + "/kernels/projects/a2/gdn2_recurrent_train"
import pathlib as _pl
_REPO = str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "ascend_fla").is_dir()))
sys.path.insert(0, _REPO); sys.path.insert(0, D)
from golden import gdn2_recurrent_backward
spec = importlib.util.spec_from_file_location("ag", os.path.join(D, "autograd.py"))
ag = importlib.util.module_from_spec(spec); spec.loader.exec_module(ag)
gdn2_recurrent = ag.gdn2_recurrent
DEV = "npu:0"; SCALE = 128 ** -0.5

allok = True
for T in (1, 2):
    B, H, K, V = 1, 16, 128, 128
    gg = torch.Generator().manual_seed(5 + T)
    q = torch.randn(B, T, H, K, generator=gg); k = torch.randn(B, T, H, K, generator=gg)
    v = torch.randn(B, T, H, V, generator=gg) * .5
    g = -torch.rand(B, T, H, K, generator=gg) * 2
    b = torch.rand(B, T, H, K, generator=gg) * 2
    w = torch.rand(B, T, H, V, generator=gg)
    S0 = torch.randn(B, H, K, V, generator=gg) * .1
    do = torch.randn(B, T, H, V, generator=gg); dSf = torch.randn(B, H, K, V, generator=gg)
    ref = gdn2_recurrent_backward(q, k, v, g, b, w, S0, do, dSf, scale=SCALE)
    # autograd.Function path (the one that crashed)
    dv_ = lambda t: t.clone().float().to(DEV).requires_grad_(True)
    qn, kn, vn, gn, bn, wn, S0n = (dv_(t) for t in (q, k, v, g, b, w, S0))
    o, fs = gdn2_recurrent(qn, kn, vn, gn, bn, wn, S0n)
    (o * do.float().to(DEV)).sum().add((fs * dSf.float().to(DEV)).sum()).backward()
    got = {"dq": qn.grad, "dk": kn.grad, "dv": vn.grad, "dg": gn.grad, "db": bn.grad, "dw": wn.grad, "dh0": S0n.grad}
    worst = 0.0; ok = True
    for n in ("dq", "dk", "dv", "dg", "db", "dw", "dh0"):
        ko = got[n].detach().cpu().numpy().reshape(-1); ro = ref[n].numpy().reshape(-1)
        rel = np.linalg.norm(ko - ro) / (np.linalg.norm(ro) + 1e-30); worst = max(worst, rel)
        ok &= np.isfinite(ko).all() and rel < 1e-3
    allok &= ok
    print(f"[autograd T={T}] fwd+bwd worst grad relL2 {worst:.3e}  {'OK' if ok else 'FAIL'}")
print("T1 VERIFY", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
