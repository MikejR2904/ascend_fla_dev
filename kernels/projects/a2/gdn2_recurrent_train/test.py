"""End-to-end check: GDN2Recurrent autograd.Function gradients vs torch autograd
through the reference recurrence (double precision). Run on an a2 NPU host:

    ASCEND_RT_VISIBLE_DEVICES=<d> python test.py
"""
import os, sys
import torch, torch_npu  # noqa
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autograd import gdn2_recurrent, Q_SCALE, QK_EPS

DEV = "npu:0"; B, T, H, K, V = 1, 4, 2, 128, 128
gg = torch.Generator().manual_seed(21)
mkr = lambda *s: torch.randn(*s, generator=gg)
q = mkr(B, T, H, K); k = mkr(B, T, H, K); v = mkr(B, T, H, V) * 0.5
g = -torch.rand(B, T, H, K, generator=gg) * 2.0
b = torch.rand(B, T, H, K, generator=gg) * 2.0
w = torch.rand(B, T, H, V, generator=gg)
S0 = torch.randn(B, H, K, V, generator=gg) * 0.1
do = mkr(B, T, H, V); dSf = mkr(B, H, K, V)


def l2n(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + QK_EPS)


ins = [t.clone().double().requires_grad_(True) for t in (q, k, v, g, b, w, S0)]
qd, kd, vd, gd, bd, wd, S0d = ins
S = S0d; outs = []
for t in range(T):
    qn = l2n(qd[:, t]) * Q_SCALE; kn = l2n(kd[:, t])
    S = S * torch.exp(gd[:, t])[..., None]
    erase = torch.einsum("bhk,bhkv->bhv", bd[:, t] * kn, S)
    delta = wd[:, t] * vd[:, t] - erase
    S = S + kn[..., None] * delta[..., None, :]
    outs.append(torch.einsum("bhk,bhkv->bhv", qn, S))
o_ref = torch.stack(outs, 1); Sf_ref = S
(o_ref * do.double()).sum().add((Sf_ref * dSf.double()).sum()).backward()
ref = {n: t.grad for n, t in zip(("dq", "dk", "dv", "dg", "db", "dw", "dh0"), ins)}

dv_ = lambda t: t.clone().float().to(DEV).requires_grad_(True)
qn_, kn_, vn_, gn_, bn_, wn_, S0n_ = (dv_(t) for t in (q, k, v, g, b, w, S0))
o, fs = gdn2_recurrent(qn_, kn_, vn_, gn_, bn_, wn_, S0n_)
(o * do.float().to(DEV)).sum().add((fs * dSf.float().to(DEV)).sum()).backward()
got = {"dq": qn_.grad, "dk": kn_.grad, "dv": vn_.grad, "dg": gn_.grad,
       "db": bn_.grad, "dw": wn_.grad, "dh0": S0n_.grad}

print("GDN2Recurrent autograd.Function vs torch autograd:")
allok = True
for n in ("dq", "dk", "dv", "dg", "db", "dw", "dh0"):
    ko = got[n].cpu().numpy().reshape(-1); ro = ref[n].numpy().reshape(-1)
    rel = np.linalg.norm(ko - ro) / (np.linalg.norm(ro) + 1e-30)
    ok = np.isfinite(ko).all() and rel < 1e-3; allok &= ok
    print(f"  {n:3s} relL2 {rel:.3e}  {'OK' if ok else 'FAIL'}")
print("GDN2 TRAINING OP", "PASS" if allok else "FAIL")
sys.exit(0 if allok else 1)
