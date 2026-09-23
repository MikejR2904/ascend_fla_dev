"""GDN-2 recurrent training op wired as a torch.autograd.Function.

Two hand-written Ascend a2 kernels (forward-with-states + reverse-recurrence
backward) compiled through the ascend_fla runtime bridge and driven end-to-end
by PyTorch. Validated on a2 silicon: all 7 gradients match torch autograd
through the reference recurrence to relative-L2 < 1e-4.

    o, final_state = gdn2_recurrent(q, k, v, g, b, w, initial_state)

Shapes: q,k,g,b : [B,T,H,128] ; v,w : [B,T,H,128] ; initial_state : [B,H,128,128].
"""
import os
import importlib.util
import torch  # noqa
from ascend_fla.runtime.compile import compile_kernel

Q_SCALE = 1.0 / (128 ** 0.5)
QK_EPS = 1e-6
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(rel, fn):
    path = os.path.join(_HERE, rel)
    spec = importlib.util.spec_from_file_location("gdn2_" + fn, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return getattr(m, fn)


# block_dim=40 launches all 40 a2 vector cores. Work is split ceil(B*H/40) per
# core, so any B*H up to 40 runs in a single parallel round (measured 3.85x over
# the old block_dim=8 at B*H=64; see PERF_ANALYSIS.md). Numerics are unchanged —
# each (b,h) recurrence is independent, only the core mapping changes.
BLOCK_DIM = 40
_FWD = compile_kernel(_load("kernels/fwd_states.py", "gdn2_fwd_states_kernel"),
                      device="a2", block_dim=BLOCK_DIM, backend="cce")
_BWD = compile_kernel(_load("kernels/bwd_step.py", "gdn2_recurrent_bwd_kernel"),
                      device="a2", block_dim=BLOCK_DIM, backend="cce")
# Checkpoint-free inference forward: same arithmetic, no per-step states/delta
# writes and no autograd node, so a no-grad call skips ~120 µs of host-side
# per-call overhead (68 MB checkpoint allocation + save_for_backward). Outputs
# are bit-identical to the training forward. See BENCHMARK_TRITON.md.
_INFER = compile_kernel(_load("kernels/fwd_infer.py", "gdn2_fwd_infer_kernel"),
                        device="a2", block_dim=BLOCK_DIM, backend="cce")


class GDN2Recurrent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, b, w, S0):
        B, T, H, K = q.shape; V = v.shape[-1]; dev = q.device
        c = lambda t: t.contiguous().float()
        q, k, v, g, b, w, S0 = (c(x) for x in (q, k, v, g, b, w, S0))
        o = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
        fs = torch.empty(B, H, K, V, dtype=torch.float32, device=dev)
        states = torch.empty(B, H, T + 1, K, V, dtype=torch.float32, device=dev)
        delta = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
        _FWD({"q": q, "k": k, "v": v, "g": g, "erase_gate": b, "w": w, "initial_state": S0},
             {"B": B, "T": T, "H": H, "TP1": T + 1},
             {"o": o, "final_state": fs, "states": states, "delta_ckpt": delta})
        ctx.save_for_backward(q, k, v, g, b, w, S0, states, delta)
        ctx.dims = (B, T, H, K, V)
        return o, fs

    @staticmethod
    def backward(ctx, do, dfinal):
        q, k, v, g, b, w, S0, states, delta = ctx.saved_tensors
        B, T, H, K, V = ctx.dims; dev = q.device
        mk = lambda last: torch.empty(B, T, H, last, dtype=torch.float32, device=dev)
        outs = {"dq": mk(K), "dk": mk(K), "dv": mk(V), "dg": mk(K), "db": mk(K), "dw": mk(V),
                "dh0": torch.empty(B, H, K, V, dtype=torch.float32, device=dev)}
        _BWD({"q": q, "k": k, "v": v, "g": g, "erase_gate": b, "w": w, "initial_state": S0,
              "dout": do.contiguous().float(), "dfinal": dfinal.contiguous().float(),
              "states": states, "delta_ckpt": delta},
             {"B": B, "T": T, "H": H, "TP1": T + 1}, outs)
        return outs["dq"], outs["dk"], outs["dv"], outs["dg"], outs["db"], outs["dw"], outs["dh0"]


def _forward_inference(q, k, v, g, b, w, S0):
    """No-grad forward: checkpoint-free kernel, only o + final_state allocated."""
    B, T, H, K = q.shape; V = v.shape[-1]; dev = q.device
    c = lambda t: t.contiguous().float()
    q, k, v, g, b, w, S0 = (c(x) for x in (q, k, v, g, b, w, S0))
    o = torch.empty(B, T, H, V, dtype=torch.float32, device=dev)
    fs = torch.empty(B, H, K, V, dtype=torch.float32, device=dev)
    _INFER({"q": q, "k": k, "v": v, "g": g, "erase_gate": b, "w": w, "initial_state": S0},
           {"B": B, "T": T, "H": H}, {"o": o, "final_state": fs})
    return o, fs


def gdn2_recurrent(q, k, v, g, b, w, initial_state):
    """GDN-2 fused-recurrent training op (differentiable). Returns (o, final_state).

    Dispatches to the checkpoint-free inference kernel when no gradient is needed
    (bit-identical outputs, ~120 µs less host overhead per call); otherwise runs
    the training forward that checkpoints every state for the backward."""
    tensors = (q, k, v, g, b, w, initial_state)
    if not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors)):
        return _forward_inference(q, k, v, g, b, w, initial_state)
    return GDN2Recurrent.apply(q, k, v, g, b, w, initial_state)
