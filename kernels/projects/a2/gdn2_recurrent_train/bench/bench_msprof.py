"""msprof/host bench driver for the GDN-2 a2 kernels. Compiles the fwd-states and
backward kernels, then launches one under a warmup+timed loop so msprof captures
device AI-Vector-Core op time. Host wall (npu.synchronize) is reported too.

    python bench_msprof.py --kernel fwd --shape decode --iters 100
"""
import sys, argparse, time, importlib.util
import torch, torch_npu  # noqa
import pathlib as _pl
_REPO = str(next(p for p in _pl.Path(__file__).resolve().parents if (p / "ascend_fla").is_dir()))
sys.path.insert(0, _REPO)
from ascend_fla.runtime.compile import compile_kernel

TR = _REPO + "/kernels/projects/a2/gdn2_recurrent_train"


def _load(rel, fn):
    spec = importlib.util.spec_from_file_location("m_" + fn, TR + "/" + rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return getattr(m, fn)


SHAPES = {"decode": (1, 1, 16), "short": (1, 16, 16), "train": (1, 64, 16),
          "batch4": (4, 64, 16), "batch8": (8, 64, 16)}
DEV = "npu:0"
BLOCK_DIM = 40


def mk(*s):
    return torch.randn(*s, dtype=torch.float32, device=DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=("fwd", "bwd"), default="fwd")
    ap.add_argument("--shape", choices=tuple(SHAPES), default="decode")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=15)
    a = ap.parse_args()
    B, T, H = SHAPES[a.shape]; K = V = 128

    q = mk(B, T, H, K); k = mk(B, T, H, K); v = mk(B, T, H, V) * 0.5
    g = -torch.rand(B, T, H, K, device=DEV) * 2.0
    b = torch.rand(B, T, H, K, device=DEV) * 2.0
    w = torch.rand(B, T, H, V, device=DEV)
    S0 = mk(B, H, K, V) * 0.1
    scal = {"B": B, "T": T, "H": H, "TP1": T + 1}

    if a.kernel == "fwd":
        fwd = compile_kernel(_load("kernels/fwd_states.py", "gdn2_fwd_states_kernel"),
                             device="a2", block_dim=BLOCK_DIM, backend="cce")
        o = mk(B, T, H, V); fs = mk(B, H, K, V); st = mk(B, H, T + 1, K, V)
        ins = {"q": q, "k": k, "v": v, "g": g, "erase_gate": b, "w": w, "initial_state": S0}
        outs = {"o": o, "final_state": fs, "states": st}
        call = lambda: fwd(ins, scal, outs)
    else:
        bwd = compile_kernel(_load("kernels/bwd_step.py", "gdn2_recurrent_bwd_kernel"),
                             device="a2", block_dim=BLOCK_DIM, backend="cce")
        st = mk(B, H, T + 1, K, V); do = mk(B, T, H, V); df = mk(B, H, K, V)
        mko = lambda last: mk(B, T, H, last)
        outs = {"dq": mko(K), "dk": mko(K), "dv": mko(V), "dg": mko(K), "db": mko(K),
                "dw": mko(V), "dh0": mk(B, H, K, V)}
        ins = {"q": q, "k": k, "v": v, "g": g, "erase_gate": b, "w": w,
               "initial_state": S0, "dout": do, "dfinal": df, "states": st}
        call = lambda: bwd(ins, scal, outs)

    for _ in range(a.warmup):
        call()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        call()
    torch.npu.synchronize()
    us = (time.perf_counter() - t0) / a.iters * 1e6
    print(f"[{a.kernel} {a.shape} B{B}T{T}H{H}] host wall {us:.2f} us/call over {a.iters} iters")


if __name__ == "__main__":
    main()
