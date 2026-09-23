"""Compare the a2 GDN-2 kernels against the baseline the model ships: the repo's
torch / torch_npu recurrence (core_backend='torch' == ascend_fla.reference.gdn2),
run on the SAME NPU. Times forward and forward+backward, reports speedup, and
checks the two agree numerically.

    ASCEND_RT_VISIBLE_DEVICES=<d> python compare_baseline.py
"""
import sys, time, importlib.util
import torch, torch_npu  # noqa
import numpy as np
import pathlib as _pl
_REPO = str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "ascend_fla").is_dir()))
sys.path.insert(0, _REPO)
from ascend_fla.reference.gdn2 import gdn2_recurrent_reference

_spec = importlib.util.spec_from_file_location(
    "gdn2_ag", _REPO + "/kernels/projects/a2/gdn2_recurrent_train/autograd.py")
_ag = importlib.util.module_from_spec(_spec); sys.modules["gdn2_ag"] = _ag; _spec.loader.exec_module(_ag)
gdn2_recurrent = _ag.gdn2_recurrent

DEV = "npu:0"
SHAPES = {"decode": (1, 1, 16), "train": (1, 64, 16), "batch8": (8, 64, 16)}
SCALE = 128 ** -0.5


def mkinputs(B, T, H, K=128, V=128, req=False):
    g = torch.Generator().manual_seed(3)
    f = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc)
    q = f(B, T, H, K, sc=0.5); k = f(B, T, H, K, sc=0.5); v = f(B, T, H, V, sc=0.25)
    gg = -torch.rand(B, T, H, K, generator=g) * 2.0
    b = torch.rand(B, T, H, K, generator=g) * 2.0
    w = torch.rand(B, T, H, V, generator=g)
    S0 = f(B, H, K, V, sc=0.1)
    out = []
    for t in (q, k, v, gg, b, w, S0):
        t = t.float().to(DEV)
        if req:
            t.requires_grad_(True)
        out.append(t)
    return out


def timed(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def run(tag, B, T, H, iters):
    K = V = 128
    # ---- forward-only ----
    q, k, v, g, b, w, S0 = mkinputs(B, T, H)
    do = torch.randn(B, T, H, V, device=DEV); dSf = torch.randn(B, H, K, V, device=DEV)
    base_fwd = lambda: gdn2_recurrent_reference(q, k, v, g, b, w, S0, scale=SCALE, use_qk_l2norm=True, qk_norm_eps=1e-6)
    mine_fwd = lambda: gdn2_recurrent(q, k, v, g, b, w, S0)
    # correctness
    ob, sb = base_fwd(); om, sm = mine_fwd()
    rel = np.linalg.norm((om - ob).detach().cpu().numpy()) / (np.linalg.norm(ob.detach().cpu().numpy()) + 1e-30)
    tb = timed(base_fwd, iters); tm = timed(mine_fwd, iters)
    print(f"[{tag}] FWD   torch_npu {tb:8.1f}us   a2-kernel {tm:8.1f}us   speedup {tb/tm:5.2f}x   relL2 {rel:.2e}")

    # ---- forward+backward ----
    def base_fb():
        qi, ki, vi, gi, bi, wi, Si = mkinputs(B, T, H, req=True)
        o, sf = gdn2_recurrent_reference(qi, ki, vi, gi, bi, wi, Si, scale=SCALE, use_qk_l2norm=True, qk_norm_eps=1e-6)
        (o * do).sum().add((sf * dSf).sum()).backward()

    def mine_fb():
        qi, ki, vi, gi, bi, wi, Si = mkinputs(B, T, H, req=True)
        o, sf = gdn2_recurrent(qi, ki, vi, gi, bi, wi, Si)
        (o * do).sum().add((sf * dSf).sum()).backward()
    tbf = timed(base_fb, max(3, iters // 4)); tmf = timed(mine_fb, iters)
    print(f"[{tag}] FWD+BWD torch_npu {tbf:8.1f}us   a2-kernel {tmf:8.1f}us   speedup {tbf/tmf:5.2f}x")


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, (B, T, H) in SHAPES.items():
        if only and tag != only:
            continue
        run(tag, B, T, H, iters=50)
