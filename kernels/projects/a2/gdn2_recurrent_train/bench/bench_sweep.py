"""Per-process block_dim sweep for the GDN-2 a2 kernels (device-bound train shape).
One build per invocation (block_dim sweeps must be per-process). Prints host wall,
which for the T=64 train shape tracks device time (host-dispatch amortized).

    python bench_sweep.py --kernel fwd --block_dim 16 --batch 1 --T 64 --iters 200
"""
import sys, argparse, time, importlib.util
import torch, torch_npu  # noqa
sys.path.insert(0, "/workspace/ascend_fla_dev")
from ascend_fla.runtime.compile import compile_kernel
TR = "/workspace/ascend_fla_dev/kernels/projects/a2/gdn2_recurrent_train"
DEV = "npu:0"


def _load(rel, fn):
    spec = importlib.util.spec_from_file_location("m_" + fn, TR + "/" + rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return getattr(m, fn)


def mk(*s):
    return torch.randn(*s, dtype=torch.float32, device=DEV)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=("fwd", "bwd"), default="fwd")
    ap.add_argument("--block_dim", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=15)
    a = ap.parse_args()
    B, T, H, K, V = a.batch, a.T, a.H, 128, 128
    bd = a.block_dim
    q = mk(B, T, H, K); k = mk(B, T, H, K); v = mk(B, T, H, V) * 0.5
    g = -torch.rand(B, T, H, K, device=DEV) * 2.0
    b = torch.rand(B, T, H, K, device=DEV) * 2.0
    w = torch.rand(B, T, H, V, device=DEV); S0 = mk(B, H, K, V) * 0.1
    scal = {"B": B, "T": T, "H": H, "TP1": T + 1}
    if a.kernel == "fwd":
        fn = _load("kernels/fwd_states.py", "gdn2_fwd_states_kernel")
        comp = compile_kernel(fn, device="a2", block_dim=bd, backend="cce")
        o = mk(B, T, H, V); fs = mk(B, H, K, V); st = mk(B, H, T + 1, K, V)
        ins = {"q": q, "k": k, "v": v, "g": g, "erase_gate": b, "w": w, "initial_state": S0}
        outs = {"o": o, "final_state": fs, "states": st}
        call = lambda: comp(ins, scal, outs)
    else:
        fn = _load("kernels/bwd_step.py", "gdn2_recurrent_bwd_kernel")
        comp = compile_kernel(fn, device="a2", block_dim=bd, backend="cce")
        st = mk(B, H, T + 1, K, V); do = mk(B, T, H, V); df = mk(B, H, K, V)
        mko = lambda last: mk(B, T, H, last)
        outs = {"dq": mko(K), "dk": mko(K), "dv": mko(V), "dg": mko(K), "db": mko(K), "dw": mko(V), "dh0": mk(B, H, K, V)}
        ins = {"q": q, "k": k, "v": v, "g": g, "erase_gate": b, "w": w, "initial_state": S0,
               "dout": do, "dfinal": df, "states": st}
        call = lambda: comp(ins, scal, outs)
    for _ in range(a.warmup):
        call()
    torch.npu.synchronize(); t0 = time.perf_counter()
    for _ in range(a.iters):
        call()
    torch.npu.synchronize()
    us = (time.perf_counter() - t0) / a.iters * 1e6
    print(f"RESULT kernel={a.kernel} block_dim={bd} B={B} T={T} H={H} BH={B*H} host_wall_us={us:.2f}")


if __name__ == "__main__":
    main()
