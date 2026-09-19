"""Runtime-generated references for the A2 KDA decode unit (no compiler import)."""
from __future__ import annotations

import torch


def make_inputs(case: dict) -> dict:
    p = case["parameters"]
    b, t, h, hv = (p[k] for k in ("B", "T", "H", "HV"))
    gen = torch.Generator(device="cpu").manual_seed(case["seed"])
    q = (torch.randn(b, t, h, 128, generator=gen) * .1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(b, t, h, 128, generator=gen), dim=-1).bfloat16()
    v = (torch.randn(b, t, hv, 128, generator=gen) * .1).bfloat16()
    g = -torch.rand(b, t, hv, 128, generator=gen) * .5
    beta = torch.rand(b, t, hv, generator=gen) * .9 + .05
    h0 = torch.randn(b, hv, 128, 128, generator=gen) * .05
    if p.get("initial_state") == "zero":
        h0.zero_()
    return {"q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous(), "g": g.contiguous(),
            "beta": beta.contiguous(), "initial_state": h0.contiguous()}


def independent_reference(x: dict, scale: float = 128 ** -.5) -> dict:
    """Token-by-token float64 recurrence on the given (already BF16-rounded) inputs, state stored [K, V]."""
    if any(t.device.type != "cpu" for t in x.values()):
        raise ValueError("Correctness references must run on Torch CPU")
    q, k, v, g, beta, s = (x[n].double() for n in ("q", "k", "v", "g", "beta", "initial_state"))
    b, t, h, _ = q.shape
    hv = v.shape[2]
    group = torch.arange(hv) // (hv // h)
    s = s.clone()
    o = torch.zeros(b, t, hv, 128, dtype=torch.float64)
    for token in range(t):
        qt, kt = q[:, token][:, group], k[:, token][:, group]          # [b, hv, 128]
        s = s * g[:, token].exp()[..., None]                           # decay along K
        delta = v[:, token] - torch.einsum("bhk,bhkv->bhv", kt, s)
        s = s + beta[:, token][..., None, None] * kt[..., :, None] * delta[..., None, :]
        o[:, token] = torch.einsum("bhk,bhkv->bhv", qt * scale, s)
    return {"o": o.float(), "final_state": s.float()}
