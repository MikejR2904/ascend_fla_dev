"""Explicit original/repaired controls independent of public dispatch changes."""
from __future__ import annotations

import functools
from pathlib import Path

import torch

from ascend_fla.ops.kda.chunk import (
    HEAD_DIM, VALUE_DIM, L_PER_CHUNK, _to_bhcld, _from_bhcld,
    _load_kernel, kda_fwd_kernels,
)


@functools.lru_cache(None)
def selected_kernels(variant):
    functions = dict(kda_fwd_kernels("stable"))
    if variant == "repaired":
        functions["recurrent"] = _load_kernel("aqk_repair", Path(__file__).parent / "kernels",
                                              "recurrent", "kda_sub45_aqk_repaired_kernel")
    elif variant == "baseline":
        # Keep the original negative control after public stable integration.
        functions["recurrent"] = kda_fwd_kernels("upstream")["recurrent"]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return functions


@functools.lru_cache(None)
def compiled_chain(block_dim, variant):
    from ascend_fla.runtime.compile import compile_kernel
    if block_dim not in (1, 2, 3, 4):
        raise ValueError("block_dim must be in the existing domain 1..4")
    return {stage: compile_kernel(fn, device="a5", block_dim=block_dim)
            for stage, fn in selected_kernels(variant).items()}


def run_chain(q, k, v, g, beta, scale, initial_state, *, device, block_dim,
               on_cpu, b, h, hv, c, impl="stable", variant="repaired", launchers=None) -> dict[str, torch.Tensor]:
    """Same five-launch ABI as the public stable chain, selected explicitly."""
    dev = q.device
    compiled = compiled_chain(block_dim, variant) if launchers is None else launchers

    def empty(shape, dtype):
        return torch.empty(*shape, dtype=dtype, device=dev)

    qc, kc = _to_bhcld(q, h, on_cpu=on_cpu), _to_bhcld(k, h, on_cpu=on_cpu)
    vc, gc = _to_bhcld(v, hv, on_cpu=on_cpu), _to_bhcld(g, hv, on_cpu=on_cpu)
    bc = _to_bhcld(beta, hv, on_cpu=on_cpu)
    # CPU initialization also supports devices without a built-in zero operator.
    state0 = initial_state if initial_state is not None else torch.zeros(
        b, hv, HEAD_DIM, VALUE_DIM, dtype=torch.float32, device="cpu").to(dev)

    bhc = (b, hv, c, L_PER_CHUNK, HEAD_DIM)
    sq = (b, hv, c, L_PER_CHUNK, L_PER_CHUNK)

    # 1) The stable gate retains log-space cumsum for scores and WY.
    eg = empty(bhc, torch.float32)
    gate_scalars = {"B": b, "HV": hv, "C": c,
                    "length_per_chunk": L_PER_CHUNK, "head_dim": HEAD_DIM}
    if impl == "stable":
        g_cumsum = empty(bhc, torch.float32)
        compiled["gate"]({"g_raw": gc}, gate_scalars,
                         {"g_cumsum": g_cumsum, "eg": eg})
    else:
        g_cumsum = None
        compiled["gate"]({"g_raw": gc}, gate_scalars, {"eg": eg})
    gate_in = {"g_cumsum": g_cumsum} if impl == "stable" else {"eg": eg}

    # 2) Scores: Aqk and the strict lower triangle.
    aqk, strict = empty(sq, torch.bfloat16), empty(sq, torch.float32)
    compiled["scores"](
        {"q": qc, "k": kc, "beta": bc, **gate_in},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "scale": scale},
        {"Aqk": aqk, "strict": strict},
    )

    # 3) Triangular inverse; the kernel H scalar is the value head count.
    akk = empty(sq, torch.bfloat16)
    compiled["inverse"]({"a": strict}, {"B": b, "H": hv, "C": c}, {"inv": akk})

    # 4) WY representation.
    w, u, qg, kg = (empty(bhc, torch.bfloat16) for _ in range(4))
    compiled["wy"](
        {"q": qc, "k": kc, "v": vc, "beta": bc, "Akk": akk, **gate_in},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "value_dim": VALUE_DIM},
        {"w": w, "u": u, "qg": qg, "kg": kg},
    )

    # 5) Chunk recurrence.
    o_c = empty(bhc, torch.bfloat16)
    final_state = empty((b, hv, HEAD_DIM, VALUE_DIM), torch.float32)
    compiled["recurrent"](
        {"q": qc, "Aqk": aqk, "kg": kg, "w": w, "u": u, "eg": eg, "initial_state": state0},
        {"B": b, "H": h, "HV": hv, "C": c, "length_per_chunk": L_PER_CHUNK,
         "head_dim": HEAD_DIM, "value_dim": VALUE_DIM, "scale": scale},
        {"o": o_c, "final_state": final_state},
    )

    return {"eg": eg, "g_cumsum": g_cumsum, "Aqk": aqk, "strict": strict, "Akk": akk,
            "w": w, "u": u, "qg": qg, "kg": kg, "o": o_c, "final_state": final_state,
            "initial_state": state0}
