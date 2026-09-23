"""Dedicated standalone backward-kernel test: the a2 `gdn2_recurrent_bwd_kernel`
(block_dim=40) vs the analytic golden `gdn2_recurrent_backward` (itself checked
against torch autograd to ~1e-16). Sweeps several (T, H) shapes.

    ASCEND_RT_VISIBLE_DEVICES=<d> python test_backward.py
"""
import sys, os, importlib.util
import torch, torch_npu  # noqa
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import pathlib as _pl
_REPO = str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "ascend_fla").is_dir()))
sys.path.insert(0, _REPO); sys.path.insert(0, HERE)
from ascend_fla.runtime.compile import compile_kernel
from golden import gdn2_recurrent_backward, _l2n

DEV = "npu:0"; SCALE = 128 ** -0.5


def _load(rel, fn):
    spec = importlib.util.spec_from_file_location("m_" + fn, os.path.join(HERE, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return getattr(m, fn)


_FWD = compile_kernel(_load("kernels/fwd_states.py", "gdn2_fwd_states_kernel"), device="a2", block_dim=40, backend="cce")
_BWD = compile_kernel(_load("kernels/bwd_step.py", "gdn2_recurrent_bwd_kernel"), device="a2", block_dim=40, backend="cce")


def run(B, T, H):
    K = V = 128
    gg = torch.Generator().manual_seed(100 + T + H)
    q = torch.randn(B, T, H, K, generator=gg); k = torch.randn(B, T, H, K, generator=gg)
    v = torch.randn(B, T, H, V, generator=gg) * 0.5
    g = -torch.rand(B, T, H, K, generator=gg) * 2.0
    b = torch.rand(B, T, H, K, generator=gg) * 2.0
    w = torch.rand(B, T, H, V, generator=gg)
    S0 = torch.randn(B, H, K, V, generator=gg) * 0.1
    do = torch.randn(B, T, H, V, generator=gg); dSf = torch.randn(B, H, K, V, generator=gg)
    ref = gdn2_recurrent_backward(q, k, v, g, b, w, S0, do, dSf, scale=SCALE)

    d = lambda t: t.float().to(DEV)
    # produce per-step states from the forward kernel (the exact backward input)
    o = torch.empty(B, T, H, V, device=DEV); fs = torch.empty(B, H, K, V, device=DEV)
    st = torch.empty(B, H, T + 1, K, V, device=DEV); dl = torch.empty(B, T, H, V, device=DEV)
    _FWD({"q": d(q), "k": d(k), "v": d(v), "g": d(g), "erase_gate": d(b), "w": d(w), "initial_state": d(S0)},
         {"B": B, "T": T, "H": H, "TP1": T + 1}, {"o": o, "final_state": fs, "states": st, "delta_ckpt": dl})
    outs = {n: torch.empty(B, T, H, K if n in ("dq", "dk", "dg", "db") else V, device=DEV)
            for n in ("dq", "dk", "dv", "dg", "db", "dw")}
    outs["dh0"] = torch.empty(B, H, K, V, device=DEV)
    _BWD({"q": d(q), "k": d(k), "v": d(v), "g": d(g), "erase_gate": d(b), "w": d(w), "initial_state": d(S0),
          "dout": d(do), "dfinal": d(dSf), "states": st, "delta_ckpt": dl},
         {"B": B, "T": T, "H": H, "TP1": T + 1}, outs)
    torch.npu.synchronize()
    ok = True; worst = 0.0
    for n in ("dq", "dk", "dv", "dg", "db", "dw", "dh0"):
        ko = outs[n].cpu().numpy().reshape(-1); ro = ref[n].numpy().reshape(-1)
        rel = np.linalg.norm(ko - ro) / (np.linalg.norm(ro) + 1e-30)
        worst = max(worst, rel); ok &= np.isfinite(ko).all() and rel < 1e-3
    print(f"[B{B} T{T:3d} H{H:2d}] backward vs golden: worst relL2 {worst:.3e}  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    allok = True
    for B, T, H in [(1, 1, 16), (1, 2, 16), (1, 4, 2), (1, 16, 16), (1, 64, 16), (1, 128, 16), (2, 64, 16)]:
        allok &= run(B, T, H)
    print("BACKWARD KERNEL", "PASS" if allok else "FAIL")
    sys.exit(0 if allok else 1)
