"""Derive + validate the GDN-2 recurrence BACKWARD against torch autograd (CPU, fp64
for a clean check). Single head. If the analytical reverse recurrence matches
autograd, the math is correct and can be lowered to a kernel."""
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

T, K, V = 6, 8, 8
scale = 1.0 / (K ** 0.5)
eps = 1e-6


def l2n(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def fwd(q, k, v, g, b, w, S0):
    S = S0
    outs = []
    for t in range(T):
        qn = l2n(q[t]) * scale
        kn = l2n(k[t])
        S = S * torch.exp(g[t])[:, None]
        erase = (b[t] * kn) @ S
        delta = w[t] * v[t] - erase
        S = S + kn[:, None] * delta[None, :]
        outs.append(qn @ S)
    return torch.stack(outs), S


q = torch.randn(T, K, requires_grad=True); k = torch.randn(T, K, requires_grad=True)
v = torch.randn(T, V, requires_grad=True); g = (-torch.rand(T, K)).requires_grad_(True)
b = (torch.rand(T, K) * 2).requires_grad_(True); w = torch.rand(T, V).requires_grad_(True)
S0 = (torch.randn(K, V) * 0.1).requires_grad_(True)

o, Sf = fwd(q, k, v, g, b, w, S0)
do = torch.randn(T, V); dSf = torch.randn(K, V)
loss = (o * do).sum() + (Sf * dSf).sum()
loss.backward()
ref = {n: t.grad.clone() for n, t in
       [("q", q), ("k", k), ("v", v), ("g", g), ("b", b), ("w", w), ("S0", S0)]}

# ---- analytical backward (reverse recurrence) ----
with torch.no_grad():
    # replay forward, caching per-step intermediates
    S = S0.clone(); cache = []
    for t in range(T):
        qn = l2n(q[t]) * scale; kn = l2n(k[t])
        S_prev = S.clone()
        eg = torch.exp(g[t]); S_dec = S_prev * eg[:, None]
        bk = b[t] * kn
        erase = bk @ S_dec
        delta = w[t] * v[t] - erase
        S = S_dec + kn[:, None] * delta[None, :]
        cache.append((qn, kn, bk, S_prev, S_dec, eg, delta, S.clone()))

    dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv = torch.zeros_like(v)
    dg = torch.zeros_like(g); db = torch.zeros_like(b); dw = torch.zeros_like(w)
    dS = dSf.clone()   # grad wrt final state

    def dl2n(x, dxn):  # backward of l2n(x) given grad dxn
        r = torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)
        return r * dxn - (r ** 3) * ((dxn * x).sum(-1, keepdim=True)) * x

    for t in reversed(range(T)):
        qn, kn, bk, S_prev, S_dec, eg, delta, S_new = cache[t]
        # o = qn @ S_new
        dqn = S_new @ do[t]
        dS_new = dS + torch.outer(qn, do[t])
        # S_new = S_dec + kn ^ delta
        dS_dec = dS_new.clone()
        dkn = dS_new @ delta
        ddelta = kn @ dS_new
        # delta = w*v - erase
        dw[t] += ddelta * v[t]; dv[t] += ddelta * w[t]
        derase = -ddelta
        # erase = bk @ S_dec
        dbk = S_dec @ derase
        dS_dec += torch.outer(bk, derase)
        # bk = b*kn
        db[t] += dbk * kn; dkn += dbk * b[t]
        # S_dec = S_prev * eg[:,None]
        dg[t] += (dS_dec * S_prev).sum(-1) * eg
        dS_prev = dS_dec * eg[:, None]
        # qn = l2n(q)*scale ; kn = l2n(k)
        dq[t] += dl2n(q[t], dqn * scale)
        dk[t] += dl2n(k[t], dkn)
        dS = dS_prev
    dS0 = dS

mine = {"q": dq, "k": dk, "v": dv, "g": dg, "b": db, "w": dw, "S0": dS0}
print("backward gradient check vs torch autograd:")
allok = True
for n in ("q", "k", "v", "g", "b", "w", "S0"):
    e = (mine[n] - ref[n]).abs().max().item()
    ok = e < 1e-9
    allok &= ok
    print(f"  d{n:2s}: max abs err {e:.2e}  {'OK' if ok else 'MISMATCH'}")
print("ALL", "PASS" if allok else "FAIL")
