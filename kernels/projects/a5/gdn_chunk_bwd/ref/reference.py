"""Independent analytical GDN adjoint; no FLA import and no autograd.

FP32 is the acceptance arithmetic. Dtype-preserving FP64 is used only for
finite-difference qualification of this formula. Device stages do not call it.
"""
import torch

NAMES = ('dq', 'dk', 'dv', 'dg', 'dbeta')


def forward(q, k, v, g, beta, *, scale=None):
    scale = q.shape[-1] ** -.5 if scale is None else scale
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    ratio = HV // H
    state = q.new_zeros(B, HV, K, V)
    out = torch.empty_like(v)
    for t in range(T):
        key = k[:, t].repeat_interleave(ratio, 1)
        query = q[:, t].repeat_interleave(ratio, 1)
        decayed = g[:, t].exp()[..., None, None] * state
        residual = v[:, t] - (key[..., None] * decayed).sum(-2)
        state = decayed + key[..., None] * (beta[:, t, :, None] * residual)[..., None, :]
        out[:, t] = (scale * query[..., None] * state).sum(-2)
    return out, state


@torch.no_grad()
def analytical(q, k, v, g, beta, do=None, dht=None, *, scale=None):
    """Return FP32/FP64 gradients, with consecutive value-head sums for q/k.

    Save only chunk boundaries; replay one chunk's primal states at a time.
    Neither inverse decay nor inversion of I-beta*k*k^T occurs.
    """
    scale = q.shape[-1] ** -.5 if scale is None else scale
    B, T, H, K = q.shape
    HV, V = v.shape[2:]
    ratio = HV // H
    key = k.repeat_interleave(ratio, 2)
    query = q.repeat_interleave(ratio, 2)
    state = q.new_zeros(B, HV, K, V)
    boundaries = []
    def step(state, t):
        d = g[:, t].exp()[..., None, None] * state
        r = v[:, t] - (key[:, t, :, :, None] * d).sum(-2)
        z = beta[:, t, :, None] * r
        state = d + key[:, t, :, :, None] * z[..., None, :]
        return d, r, z, state
    for t in range(T):
        if t % 64 == 0:
            boundaries.append(state.clone())
        state = step(state, t)[-1]
    back = torch.zeros_like(state) if dht is None else dht.clone()
    cotangent = torch.zeros_like(v) if do is None else do
    dq, dk, dv = torch.empty_like(query), torch.empty_like(key), torch.empty_like(v)
    dg, db = torch.empty_like(g), torch.empty_like(beta)
    for chunk in reversed(range((T + 63) // 64)):
        state = boundaries[chunk]
        saved = []
        begin, end = chunk * 64, min(T, (chunk + 1) * 64)
        for t in range(begin, end):
            values = step(state, t)
            state = values[-1]
            saved.append(values)
        for t in reversed(range(begin, end)):
            d, residual, z, state = saved[t-begin]
            kt, dot = key[:, t], cotangent[:, t]
            dq[:, t] = scale * (state * dot[..., None, :]).sum(-1)
            back = back + (scale * query[:, t, :, :, None]) * dot[..., None, :]
            dz = (back * kt[..., None]).sum(-2)
            dr = beta[:, t, :, None] * dz
            dk[:, t] = (back * z[..., None, :]).sum(-1) - (d * dr[..., None, :]).sum(-1)
            dv[:, t] = dr
            db[:, t] = (dz * residual).sum(-1)
            dD = back - kt[..., None] * dr[..., None, :]
            dg[:, t] = (dD * d).sum((-1, -2))
            back = g[:, t].exp()[..., None, None] * dD
    return dict(zip(NAMES, (dq.reshape(B,T,H,ratio,K).sum(3),
                           dk.reshape(B,T,H,ratio,K).sum(3), dv, dg, db)))


def metrics(actual, expected):
    result = {}
    for name in NAMES:
        assert actual[name].shape == expected[name].shape, name
        a, b = actual[name].double(), expected[name].double()
        result[name] = dict(relative_l2=((a-b).norm()/b.norm().clamp_min(1e-30)).item(),
                            max_abs=(a-b).abs().max().item(), finite=bool(torch.isfinite(a).all()))
    return result


def acceptable(values, limit=1e-4):
    return all(row['finite'] and row['relative_l2'] <= limit for row in values.values())
