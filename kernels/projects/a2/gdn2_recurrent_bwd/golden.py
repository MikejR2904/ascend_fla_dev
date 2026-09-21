"""GDN-2 recurrent BACKWARD golden (oracle) for [B,T,H,K/V], validated against
torch autograd. This is what the backward kernel must reproduce. The kernel will
mirror this exactly: recompute the forward caching per-step S, then reverse."""
import torch

QK_EPS = 1e-6


def _l2n(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + QK_EPS)


def _dl2n(x, dxn):
    r = torch.rsqrt((x * x).sum(-1, keepdim=True) + QK_EPS)
    return r * dxn - (r ** 3) * ((dxn * x).sum(-1, keepdim=True)) * x


def gdn2_recurrent_backward(q, k, v, g, b, w, S0, do, dSf, *, scale):
    """All tensors [B,T,H,K] / [B,T,H,V]; S0,dSf [B,H,K,V]. Returns grads dict."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv = torch.zeros_like(v)
    dg = torch.zeros_like(g); db = torch.zeros_like(b); dw = torch.zeros_like(w)
    dh0 = torch.zeros_like(S0)
    for bi in range(B):
        for hi in range(H):
            # forward, cache S per step boundary: S_hist[t] = state before step t
            S = S0[bi, hi].clone()
            S_hist = [S.clone()]
            eg_l, qn_l, kn_l, bk_l, delta_l, Sdec_l = [], [], [], [], [], []
            for t in range(T):
                qn = _l2n(q[bi, t, hi]) * scale
                kn = _l2n(k[bi, t, hi])
                eg = torch.exp(g[bi, t, hi])
                S_dec = S * eg[:, None]
                bk = b[bi, t, hi] * kn
                erase = bk @ S_dec
                delta = w[bi, t, hi] * v[bi, t, hi] - erase
                S = S_dec + kn[:, None] * delta[None, :]
                S_hist.append(S.clone())
                eg_l.append(eg); qn_l.append(qn); kn_l.append(kn)
                bk_l.append(bk); delta_l.append(delta); Sdec_l.append(S_dec)
            # reverse
            dS = dSf[bi, hi].clone()
            for t in reversed(range(T)):
                qn, kn, bk, eg = qn_l[t], kn_l[t], bk_l[t], eg_l[t]
                delta, S_dec = delta_l[t], Sdec_l[t]
                S_prev, S_new = S_hist[t], S_hist[t + 1]
                dot = do[bi, t, hi]
                dqn = S_new @ dot                     # reduce over V
                dS_new = dS + torch.outer(qn, dot)
                dS_dec = dS_new.clone()
                dkn = dS_new @ delta                  # reduce over V
                ddelta = kn @ dS_new                  # reduce over K
                dw[bi, t, hi] += ddelta * v[bi, t, hi]
                dv[bi, t, hi] += ddelta * w[bi, t, hi]
                derase = -ddelta
                dbk = S_dec @ derase                  # reduce over V
                dS_dec += torch.outer(bk, derase)
                db[bi, t, hi] += dbk * kn
                dkn = dkn + dbk * b[bi, t, hi]
                dg[bi, t, hi] += (dS_dec * S_prev).sum(-1) * eg   # reduce over V
                dS_prev = dS_dec * eg[:, None]
                dq[bi, t, hi] += _dl2n(q[bi, t, hi], dqn * scale)
                dk[bi, t, hi] += _dl2n(k[bi, t, hi], dkn)
                dS = dS_prev
            dh0[bi, hi] = dS
    return {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "db": db, "dw": dw, "dh0": dh0}


if __name__ == "__main__":
    torch.manual_seed(1); torch.set_default_dtype(torch.float64)
    B, T, H, K, V = 1, 5, 3, 8, 8
    scale = 1.0 / (K ** 0.5)
    q = torch.randn(B, T, H, K, requires_grad=True)
    k = torch.randn(B, T, H, K, requires_grad=True)
    v = torch.randn(B, T, H, V, requires_grad=True)
    g = (-torch.rand(B, T, H, K)).requires_grad_(True)
    b = (torch.rand(B, T, H, K) * 2).requires_grad_(True)
    w = torch.rand(B, T, H, V).requires_grad_(True)
    S0 = (torch.randn(B, H, K, V) * 0.1).requires_grad_(True)

    def fwd(q, k, v, g, b, w, S0):
        outs = []
        S = S0
        for t in range(T):
            qn = _l2n(q[:, t]) * scale; kn = _l2n(k[:, t])
            S = S * torch.exp(g[:, t])[..., None]
            erase = torch.einsum("bhk,bhkv->bhv", b[:, t] * kn, S)
            delta = w[:, t] * v[:, t] - erase
            S = S + kn[..., None] * delta[..., None, :]
            outs.append(torch.einsum("bhk,bhkv->bhv", qn, S))
        return torch.stack(outs, 1), S
    o, Sf = fwd(q, k, v, g, b, w, S0)
    do = torch.randn_like(o); dSf = torch.randn_like(Sf)
    (o * do).sum().add((Sf * dSf).sum()).backward()
    ref = dict(dq=q.grad, dk=k.grad, dv=v.grad, dg=g.grad, db=b.grad, dw=w.grad, dh0=S0.grad)
    mine = gdn2_recurrent_backward(q.detach(), k.detach(), v.detach(), g.detach(),
                                   b.detach(), w.detach(), S0.detach(), do, dSf, scale=scale)
    allok = True
    for n in ("dq", "dk", "dv", "dg", "db", "dw", "dh0"):
        e = (mine[n] - ref[n]).abs().max().item(); ok = e < 1e-9; allok &= ok
        print(f"  {n:3s} max abs err {e:.2e}  {'OK' if ok else 'MISMATCH'}")
    print("GOLDEN", "PASS" if allok else "FAIL")
