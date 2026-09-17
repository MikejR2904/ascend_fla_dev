"""A2-04: locate the missing synchronization behind ``c1-multihead-o-corrupt`` (diagnosis only).

Runs ascriptor's ``kda_sub45_fused_kernel`` (read-only: ``kernels/projects/a5/kda_fwd``) on the host
simulators. Only the recurrent kernel runs; its inputs are the unit's own CPU checkpoints
(``ref/stages.py``), so every discrepancy belongs to the recurrent kernel.

    python benchmarks/diag_c1_multihead.py --profile a5 --grid          # issue verification command
    python benchmarks/diag_c1_multihead.py --profile a2 --grid          # a2 conclusion (exits 3, see below)
    python benchmarks/diag_c1_multihead.py --profile a5 --grid --full   # + every hardware-table cell, B=2
    python benchmarks/diag_c1_multihead.py --profile a5 --bd 1 --hv 2 --c 1 2 --events   # step 2 event observation
    python benchmarks/diag_c1_multihead.py --profile a5 --grid --variant aqk-only --bitwise-against upstream

What each grid cell reports
---------------------------
* ``functional``: the functional interpreter (program order, no pipe timing), per-head ``o`` rel-L2 vs the CPU
  reference. It cannot exhibit a race and is expected to be exact.
* ``pipesim``: hazards of the lowered-event pipe model, attributed to heads. Every ``Aqk`` L1 read (Cube stage C)
  is matched against every ``Aqk`` L1 write on the same core; a pair is *unordered* when neither vector clock
  precedes the other (the same test pipesim uses for its hazard report).
* ``replay``: a model-timed value replay. Each ``Aqk`` read takes the latest overlapping ``Aqk`` write whose scheduled
  end precedes the read's scheduled start, and that tile of ``o`` is recomputed with *that* write's ``Aqk``.
  A head counts as correct only when its replay is bitwise equal to the reference. Its rel-L2 is printed too, but a
  wrong head's rel-L2 depends on how different the neighbouring head's Aqk is (random inputs here), so it is not
  comparable to the hardware magnitudes and not used as the verdict.
* hard criteria: replay ``o``/``final_state`` bitwise equal across ``bd`` for the same (B, HV, C); with
  ``--bitwise-against``, bitwise equal to another variant.

``--profile a2`` tries to build the same source under ``ascriptor.a2``, reports why it cannot, and checks the
cube-side construct statically. It exits 3 (no a2 simulation possible yet), deliberately not 0.

The kernel tree is found via ``--unit``, ``$ASCRIPTOR_KDA_FWD``, or ``$ASCRIPTOR_WORKSPACE/kernels/...``
(default workspace ``../ascriptor``). Variants are written to ``tmp/A2-04/variants/``; the ascriptor checkout is
never modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TMP = REPO / "tmp" / "A2-04"

# Source anchors in upstream recurrent.py (checked at load time, so a moved or edited line fails loudly).
AQK_WRITE = "l1_Aqk[aqk_slot][0:L, 0:L] <<= Aqk[b_aqk, hv_aqk, c_idx, 0:L, 0:L]"
AQK_READ = "l0a_delta[slot][0:L, 0:L] <<= l1_Aqk[aqk_slot][0:L, 0:L]"

_SLOT = "Var(((pair_idx - pair_begin) * C + c_idx) % 2)"
# CPU float reproducibility floor (BLAS thread count changes bits); anything larger is a real discrepancy.
REPRO_TOL = 2.5e-7
VARIANTS = {
    "upstream": [],
    # Recommended repair: rotate the Aqk hand-off slot every cycle on a core, so two credits meet two slots.
    "aqk-only": [
        ("aqk_slot = Var(c_idx % 2)\n            aqk_l1_valid.wait()", f"aqk_slot = {_SLOT}\n            aqk_l1_valid.wait()"),
        ("aqk_slot = Var(c_idx % 2)\n            qg_slot = Var(c_idx % 2)", f"aqk_slot = {_SLOT}\n            qg_slot = Var(c_idx % 2)"),
    ],
    # Alternative repair: one credit on the Aqk hand-off (drops one cycle of run-ahead).
    "aqk-sevent": [
        ("aqk_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)", "aqk_l1_valid = SEvent(Pipe.MTE1, Pipe.MTE2, preset=True)"),
        ("aqk_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)", "aqk_l1_ready = SEvent(Pipe.MTE2, Pipe.MTE1)"),
    ],
    # Conservative repair: rotate q/qg as well.
    "slot-by-cycle": [
        ("q_slot = Var(c_idx % 2)", f"q_slot = {_SLOT}"),
        ("aqk_slot = Var(c_idx % 2)\n            aqk_l1_valid.wait()", f"aqk_slot = {_SLOT}\n            aqk_l1_valid.wait()"),
        ("aqk_slot = Var(c_idx % 2)\n            qg_slot = Var(c_idx % 2)", f"aqk_slot = {_SLOT}\n            qg_slot = {_SLOT}"),
    ],
    # Negative control: rotate only q/qg. Must leave the defect in place.
    "qg-only": [
        ("q_slot = Var(c_idx % 2)", f"q_slot = {_SLOT}"),
        ("aqk_slot = Var(c_idx % 2)\n            qg_slot = Var(c_idx % 2)", f"aqk_slot = Var(c_idx % 2)\n            qg_slot = {_SLOT}"),
    ],
}


# ----------------------------------------------------------------------------------------------- loading


def unit_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    if os.environ.get("ASCRIPTOR_KDA_FWD"):
        return Path(os.environ["ASCRIPTOR_KDA_FWD"])
    ws = Path(os.environ.get("ASCRIPTOR_WORKSPACE", REPO.parent / "ascriptor"))
    return ws / "kernels/projects/a5/kda_fwd"


def variant_source(unit: Path, variant: str, facade: str = "a5") -> str:
    src = (unit / "kernels/recurrent.py").read_text()
    for anchor in (AQK_WRITE, AQK_READ, "from ascriptor.a5 import *"):
        if src.count(anchor) != 1:
            raise SystemExit(f"upstream recurrent.py changed: anchor not found exactly once: {anchor!r}")
    for old, new in VARIANTS[variant]:
        if src.count(old) != 1:
            raise SystemExit(f"variant {variant}: anchor not found exactly once: {old!r}")
        src = src.replace(old, new)
    if facade != "a5":
        src = src.replace("from ascriptor.a5 import *", f"from ascriptor.{facade} import *")
    return src


def load_kernel(unit: Path, variant: str, facade: str = "a5"):
    """Import the (possibly patched) recurrent kernel from a temp copy; line numbers are preserved."""
    if str(unit) not in sys.path:
        sys.path.insert(0, str(unit))
    src = variant_source(unit, variant, facade)
    name = f"recurrent_{facade}_{variant.replace('-', '_')}"
    path = TMP / "variants" / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(f".{os.getpid()}.part")
    part.write_text(src)
    os.replace(part, path)  # atomic: concurrent runs never import a half-written variant
    spec = importlib.util.spec_from_file_location(f"diag_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lines = src.splitlines()
    anchors = {"write": next(i for i, x in enumerate(lines) if AQK_WRITE in x) + 1,
               "read": next(i for i, x in enumerate(lines) if AQK_READ in x) + 1}
    return module.kda_sub45_fused_kernel, anchors


# ----------------------------------------------------------------------------------------------- one case


def chunk_terms(inputs):
    """Per-chunk h, v_new and scaled q exactly as ref/stages.py computes them (same ops, same casts)."""
    import torch
    from ref.gate import subkernel as gate
    from ref.oracle import to_chunked_inputs

    q, k, v, raw, beta = to_chunked_inputs(*(inputs[name] for name in ("q", "k", "v", "g_raw", "beta")), chunk_size=64)
    b, _, c, length, width = q.shape
    hv = v.shape[1]
    q = q.float().repeat_interleave(hv // q.shape[1], dim=1)
    k = k.float().repeat_interleave(hv // k.shape[1], dim=1)
    g = gate(raw)
    strict = torch.tril(torch.matmul(-(k * g * beta[..., None]), (k / g).transpose(-1, -2)), diagonal=-1)
    eye = torch.eye(length, dtype=torch.float32).expand(b, hv, c, length, length)
    akk = torch.linalg.solve_triangular(eye - strict, eye, upper=False, unitriangular=True).to(torch.bfloat16)
    a_beta = (akk.float() * beta[..., None, :]).to(torch.bfloat16).float()
    k_exp = (k * g).to(torch.bfloat16).float()
    w = torch.matmul(a_beta, k_exp).to(torch.bfloat16)
    u = torch.matmul(a_beta, v.float()).to(torch.bfloat16)
    kg = (k * (g[..., -1:, :] / g)).to(torch.bfloat16)
    state = inputs["initial_state"].clone()
    hs, news, qs = [], [], []
    for chunk in range(c):
        h = state.to(torch.bfloat16).float()
        new = (u[:, :, chunk].float() - w[:, :, chunk].float() @ h).to(torch.bfloat16).float()
        qs.append((q[:, :, chunk] * g[:, :, chunk] * (width ** -0.5)).to(torch.bfloat16).float())
        hs.append(h)
        news.append(new)
        state = state * g[:, :, chunk, -1, :, None] + kg[:, :, chunk].float().transpose(-1, -2) @ new
    return torch.stack(hs, 2), torch.stack(news, 2), torch.stack(qs, 2)


def o_with_aqk(terms, b, hv, c, aqk_bf16):
    """One (b, hv, chunk) of o computed with a given Aqk, as the reference casts it."""
    import torch

    h, new, qs = terms
    return (qs[b, hv, c] @ h[b, hv, c] + aqk_bf16.float() @ new[b, hv, c]).to(torch.bfloat16)


def rel_l2(a, b) -> float:
    import torch

    a, b = a.float(), b.float()
    if not torch.isfinite(a).all():
        return float("nan")
    return float((a - b).norm() / b.norm())


def attribute(sched, anchors, *, B, HV, C, bd):
    """Map each Aqk L1 read/write task to (core, b, hv, chunk[, tile]); find unordered pairs and the model-timed source."""
    from ascriptor.backends.sim.pipesim import _before

    pair_count = B * HV
    reads, writes = [], []
    for lane, tasks in sched.lanes.items():
        if not lane.endswith("/cube"):
            continue
        core = int(re.match(r"core(\d+)/", lane).group(1))
        pair_begin = (pair_count * core) // bd
        n_w = n_r = 0
        for t in tasks:
            m = re.search(r":(\d+):\d+", str(t.op.loc))
            line = int(m.group(1)) if m else -1
            if line == anchors["write"] and t.op.opcode.startswith("dma.gm_to_l1"):
                rec, n_w = {"core": core, "cycle": n_w, "task": t, "tile": None}, n_w + 1
                writes.append(rec)
            elif line == anchors["read"] and t.op.opcode.startswith("dma.l1_to_l0"):
                rec, n_r = {"core": core, "cycle": n_r // 2, "task": t, "tile": n_r % 2}, n_r + 1
                reads.append(rec)
            else:
                continue
            pair_idx = pair_begin + rec["cycle"] // C
            rec.update(b=pair_idx // HV, hv=pair_idx % HV, c=rec["cycle"] % C)
    out = []
    for r in reads:
        rt = r["task"]
        rkeys = {a.key: a for a in rt.accesses if a.kind == "read" and a.key[0] == "l1"}
        overl = [w for w in writes if w["core"] == r["core"] and any(
            a.kind == "write" and a.key in rkeys and a.overlaps(rkeys[a.key]) for a in w["task"].accesses)]
        mine = [w for w in overl if (w["b"], w["hv"], w["c"]) == (r["b"], r["hv"], r["c"])]
        unordered = [w for w in overl if w not in mine
                     and not _before(w["task"].clock, rt.clock) and not _before(rt.clock, w["task"].clock)]
        landed = [w for w in overl if w["task"].end <= rt.start]
        source = max(landed, key=lambda w: w["task"].end) if landed else None
        out.append({**{k: r[k] for k in ("core", "b", "hv", "c", "tile")},
                    "own_write_overlaps": bool(mine),
                    "unordered_with": [(w["b"], w["hv"], w["c"]) for w in unordered],
                    "model_source": None if source is None else (source["b"], source["hv"], source["c"])})
    return out, len(writes)



# Events on the Aqk hand-off, plus two neighbours for comparison (qg mutex, L0C output handshake).
OBSERVED = ("aqk_l1_valid", "aqk_l1_ready", "qg_mutex", "out_l0c_valid", "out_l0c_ready")


def observe_events(sched, *, B, HV, C, bd):
    """Per (core, cycle): scheduled cycle of each wait/set/lock/ready/free on OBSERVED objects, in execution order.

    Returns {core: [ {b, hv, c, ops: [(name, opcode, index_within_cycle, start, end)]} ]}. The n-th call of an
    op kind on an object within the lane is attributed to cycle n (each of these runs once per cycle, except the
    L0C output handshake, which runs once per value tile)."""
    pair_count = B * HV
    out = {}
    for lane, tasks in sched.lanes.items():
        if not lane.endswith("/cube"):
            continue
        core = int(re.match(r"core(\d+)/", lane).group(1))
        pair_begin = (pair_count * core) // bd
        counts, per_cycle = {}, {}
        for t in tasks:
            if not t.op.opcode.startswith("sync."):
                continue
            names = [getattr(o, "name", None) for o in getattr(t.op, "operands", ())]
            hit = next((n for n in names if n in OBSERVED), None)
            if hit is None:
                continue
            key = (hit, t.op.opcode)
            n = counts.get(key, 0)
            counts[key] = n + 1
            per_tile = hit.startswith("out_l0c")
            cycle = n // 2 if per_tile else n
            per_cycle.setdefault(cycle, []).append((hit, t.op.opcode.removeprefix("sync."), n, t.start, t.end))
        rows = []
        for cycle in sorted(per_cycle):
            pair_idx = pair_begin + cycle // C
            rows.append({"b": pair_idx // HV, "hv": pair_idx % HV, "c": cycle % C,
                         "ops": sorted(per_cycle[cycle], key=lambda x: x[3])})
        out[core] = rows
    return out


def explain_aqk_credits(rows, depth=2):
    """Which set released each aqk_l1_valid.wait (FIFO credits, `depth` preset), and was that set after the
    previous cycle's Aqk read? Returns one line per cycle."""
    sets = []  # (cycle position, end) of aqk_l1_valid.set in order
    waits = []
    for i, r in enumerate(rows):
        for name, op, n, start, end in r["ops"]:
            if name == "aqk_l1_valid" and op == "set":
                sets.append((i, end))
            if name == "aqk_l1_valid" and op == "wait":
                waits.append((i, start))
    lines = []
    for k, (i, start) in enumerate(waits):
        r = rows[i]
        who = f"(b={r['b']},hv={r['hv']},c={r['c']})"
        if k < depth:
            src = "a preset credit"
            released_by_prev = False
        else:
            j, set_end = sets[k - depth]
            p = rows[j]
            src = f"the valid.set of (b={p['b']},hv={p['hv']},c={p['c']}) at cycle {set_end}"
            released_by_prev = j == i - 1
        prev_set = sets[i - 1][1] if i >= 1 and i - 1 < len(sets) else None
        order = ("" if prev_set is None else
                 f"; previous cycle's valid.set (end of its Aqk read) at {prev_set} -> write wait "
                 f"{'AFTER' if start >= prev_set else 'BEFORE'} it")
        lines.append(f"{who} Aqk write wait at cycle {start} released by {src}"
                     f"{' (= previous cycle, ordered)' if released_by_prev else ''}{order}")
    return lines


def run_case(kernel, anchors, *, B, HV, C, bd, pipesim, seed, timeout):
    import torch
    from kernels.composition import chunked, poison
    from ref.inputs import make_inputs
    from ref.stages import stage_inputs

    inputs = make_inputs({"parameters": {"B": B, "H": 1, "HV": HV, "C": C, "L": 64, "K": 128, "V": 128}, "seed": seed})
    data = chunked(inputs)
    exp = stage_inputs(inputs)
    args = (data["q"], exp["score.Aqk"], exp["wy.kg"], exp["wy.w"], exp["wy.u"], exp["gate.g"], data["initial_state"],
            poison((B, HV, C, 64, 128), torch.bfloat16), poison((B, HV, 128, 128), torch.float32),
            B, 1, HV, C, 64, 128, 128, 128 ** -0.5)
    ref_o, ref_fs = exp["recurrent.o"], exp["recurrent.final_state"]
    t0 = time.time()
    result = {"B": B, "HV": HV, "C": C, "bd": bd}
    if not pipesim:
        from ascriptor.backends.sim.launch import run_kernel

        o, fs = run_kernel(kernel, *args, block_dim=bd, timeout=timeout, seed_outputs=True)
        result.update(o=o, final_state=fs)
    else:
        from ascriptor.backends.sim.pipesim import simulate
        from ascriptor.passes import PIPELINE, PassManager

        res = simulate(PassManager(PIPELINE).run(kernel.ir()), args, block_dim=bd, timeout=timeout,
                       seed_outputs=True, check_gm=True)
        o, fs = res.outputs
        reads, n_writes = attribute(res.scheduler, anchors, B=B, HV=HV, C=C, bd=bd)
        if len(reads) != 2 * B * HV * C or n_writes != B * HV * C:
            raise RuntimeError(f"attribution found {len(reads)} Aqk reads / {n_writes} writes, "
                               f"expected {2 * B * HV * C} / {B * HV * C}")
        # Model-timed replay. Self-check first: own Aqk must rebuild the reference o. Bitwise is not attainable in
        # general: the CPU reference is not bit-reproducible across BLAS thread counts (measured: 3 of 131072
        # elements differ by 1.192e-07 at B1 HV4 C4 between 1 and 11 threads), so allow one such step and report it.
        terms = chunk_terms(inputs)
        aqk = exp["score.Aqk"]
        self_check = 0.0
        for b, hv, c in itertools.product(range(B), range(HV), range(C)):
            dev = float((o_with_aqk(terms, b, hv, c, aqk[b, hv, c]).float() - ref_o[b, hv, c].float()).abs().max())
            self_check = max(self_check, dev)
            if dev > REPRO_TOL:
                raise RuntimeError(f"replay self-check failed at (b,hv,c)=({b},{hv},{c}): max_abs {dev:.3e} > {REPRO_TOL:.1e}")
        replay = ref_o.clone()
        for r in reads:
            src = r["model_source"]
            if src is None:
                raise RuntimeError(f"no Aqk write landed before read {r}; replay undefined")
            b, hv, c, tile = r["b"], r["hv"], r["c"], r["tile"]
            if src == (b, hv, c):
                continue  # read its own Aqk: this tile is exactly the reference value
            v0, v1 = tile * 64, (tile + 1) * 64
            replay[b, hv, c, :, v0:v1] = o_with_aqk(terms, b, hv, c, aqk[src])[:, v0:v1]
        result.update(replay_self_check=self_check)
        result.update(o=o, final_state=fs, replay=replay, reads=reads, hazards=res.hazards,
                      deadlock=res.report.get("deadlock"), events=observe_events(res.scheduler, B=B, HV=HV, C=C, bd=bd))
    result["secs"] = round(time.time() - t0, 1)
    heads = []
    for b in range(B):
        for hv in range(HV):
            h = {"b": b, "hv": hv, "o_rel_l2": rel_l2(result["o"][b, hv], ref_o[b, hv]),
                 "state_rel_l2": rel_l2(result["final_state"][b, hv], ref_fs[b, hv]),
                 "state_max_abs": float((result["final_state"][b, hv].float() - ref_fs[b, hv].float()).abs().max())}
            if pipesim:
                rep, ref = result["replay"][b, hv].float(), ref_o[b, hv].float()
                h["replay_rel_l2"] = rel_l2(rep, ref)
                h["replay_bitwise"] = bool(torch.equal(result["replay"][b, hv], ref_o[b, hv]))
                h["replay_max_abs"] = float((rep - ref).abs().max())
                h["replay_norm_ratio"] = float(rep.norm() / ref.norm())  # "norm looks normal" signature
                h["replay_finite"] = bool(torch.isfinite(rep).all())
            heads.append(h)
    result["heads"] = heads
    return result


# ----------------------------------------------------------------------------------------------- a2


def a2_conclusion(unit: Path) -> int:
    print("profile=a2: building the same kda_sub45_fused_kernel source under `from ascriptor.a2 import *`")
    try:
        load_kernel(unit, "upstream", facade="a2")
        print("  a2 build: OK (unexpected; extend this script to simulate it before drawing a conclusion)")
        return 1
    except Exception as e:  # noqa: BLE001 - the failure itself is the evidence
        print(f"  a2 build: FAILED  {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}")
    import ascriptor.a2 as a2
    import ascriptor.a5 as a5

    for name in ("vf", "DEvent", "SEvent", "DBuff", "Position", "Pipe"):
        print(f"  exported by ascriptor.a5={name in a5.__all__} ascriptor.a2={name in a2.__all__}: {name}")
    lines = variant_source(unit, "upstream").splitlines()
    construct = {
        "Aqk L1 write (MTE2 gm_to_l1)": AQK_WRITE,
        "Aqk L1 read (MTE1 l1_to_l0)": AQK_READ,
        "two-credit preset event (writer waits on it)": "aqk_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)",
        "two-credit ready event": "aqk_l1_ready = DEvent(Pipe.MTE2, Pipe.MTE1)",
        "slot chosen by chunk index": "aqk_slot = Var(c_idx % 2)",
        "two-slot L1 buffer": "l1_Aqk = DBuff(DT.bfloat16, [L, L], Position.L1)",
    }
    for label, needle in construct.items():
        print(f"  construct: {label}: recurrent.py line(s) {[i + 1 for i, x in enumerate(lines) if needle in x]}")
    vf_on_path = [i + 1 for i, x in enumerate(lines) if re.search(r"l1_Aqk|aqk_l1_|aqk_slot", x) and "_vf(" in x]
    print(f"  @vf calls on the Aqk hand-off lines: {vf_on_path or 'none'}")
    print("  conclusion: no a2 sim/pipesim run is possible until A2-03 derives an a2 unit (its vector side must be "
          "rewritten without @vf). The defective construct uses only names a2 also exports and no @vf, so an a2 "
          "derivation that copies this hand-off inherits it. A2-03 should adopt the aqk-only slot rotation and rerun "
          "this grid on the a2 unit.")
    return 3


# ----------------------------------------------------------------------------------------------- main


def correct_heads(heads, key, good):
    ok = [i for i, h in enumerate(heads) if (h[key] if isinstance(h[key], bool) else h[key] < good)]
    return "all" if len(ok) == len(heads) else ("none" if not ok else ",".join(map(str, ok)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=("a5", "a2"), default="a5")
    ap.add_argument("--unit", help="path to kernels/projects/a5/kda_fwd")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="upstream")
    ap.add_argument("--grid", action="store_true", help="bd in {1,2} x HV in {1,2,4} x C in {1,2,3}, B=1")
    ap.add_argument("--full", action="store_true", help="with --grid: all 12 cells of the hardware table (bd in {1,2,4} x HV in {2,4,8,16}) at C in {1,2}, and B=2")
    ap.add_argument("--bd", type=int, nargs="*", default=[1])
    ap.add_argument("--hv", type=int, nargs="*", default=[2])
    ap.add_argument("--c", type=int, nargs="*", default=[1])
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--good", type=float, default=1e-2, help="per-head rel-L2 counted as correct (the hardware table used < 0.01)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--no-functional", action="store_true", help="skip the functional-interpreter pass")
    ap.add_argument("--bitwise-against", choices=sorted(VARIANTS), help="also run this variant; compare replay o and final_state bitwise")
    ap.add_argument("--events", action="store_true", help="step 2: print Aqk / qg / L0C sync calls per head and which credit released each Aqk write")
    ap.add_argument("--json", help="write the per-head table here")
    args = ap.parse_args(argv)
    sys.dont_write_bytecode = True  # never drop __pycache__ into the read-only ascriptor checkout

    unit = unit_root(args.unit)
    if not (unit / "kernels/recurrent.py").is_file():
        raise SystemExit(f"kda_fwd unit not found at {unit}; set ASCRIPTOR_KDA_FWD or --unit")
    if args.profile == "a2":
        return a2_conclusion(unit)

    import torch

    kernel, anchors = load_kernel(unit, args.variant)
    other = load_kernel(unit, args.bitwise_against) if args.bitwise_against else None
    if args.grid:
        shapes = [(1, bd, hv, c) for bd, hv, c in itertools.product([1, 2], [1, 2, 4], [1, 2, 3])]
        if args.full:
            shapes += [(1, bd, hv, c) for bd, hv, c in itertools.product([1, 2], [8, 16], [1, 2])]
            shapes += [(1, 4, hv, c) for hv, c in itertools.product([2, 4, 8, 16], [1, 2])]
            shapes += [(2, bd, 2, c) for bd, c in itertools.product([1, 2, 4], [1, 2, 3])]
    else:
        shapes = [(args.b, bd, hv, c) for bd, hv, c in itertools.product(args.bd, args.hv, args.c)]

    print(f"profile=a5 variant={args.variant} H=1 seed={args.seed} torch_threads={torch.get_num_threads()}; "
          f"correct = replay bitwise equal to the reference (rel-L2 < {args.good:g} also shown); "
          f"Aqk L1 write at recurrent.py:{anchors['write']}, read at recurrent.py:{anchors['read']}", flush=True)
    rows, by_shape, problems = [], {}, []
    for B, bd, hv, c in shapes:
        tag = f"B={B} bd={bd} HV={hv} C={c}"
        if not args.no_functional:
            f = run_case(kernel, anchors, B=B, HV=hv, C=c, bd=bd, pipesim=False, seed=args.seed, timeout=args.timeout)
            print(f"{tag} functional: correct heads {correct_heads(f['heads'], 'o_rel_l2', args.good)} | "
                  f"o rel-L2 max {max(h['o_rel_l2'] for h in f['heads']):.3e} | "
                  f"final_state rel-L2 max {max(h['state_rel_l2'] for h in f['heads']):.3e} | {f['secs']}s", flush=True)
        r = run_case(kernel, anchors, B=B, HV=hv, C=c, bd=bd, pipesim=True, seed=args.seed, timeout=args.timeout)
        racy = [x for x in r["reads"] if x["unordered_with"]]
        stale = [x for x in r["reads"] if x["model_source"] != (x["b"], x["hv"], x["c"])]
        print(f"{tag} pipesim: hazards={len(r['hazards'])} deadlock={r['deadlock']} | Aqk reads={len(r['reads'])} "
              f"unordered-with-another-write={len(racy)} read-another-cycle's-Aqk={len(stale)} | {r['secs']}s", flush=True)
        for x in racy:
            print(f"    core{x['core']} (b={x['b']},hv={x['hv']}) chunk {x['c']} tile {x['tile']}: Aqk read unordered with the "
                  f"Aqk write of (b,hv,c)={x['unordered_with']}; model timing read {x['model_source']}", flush=True)
        if args.events:
            for core, erows in sorted(r["events"].items()):
                print(f"    events core{core} (scheduled start cycle of each call; qg_mutex/out_l0c shown for comparison):", flush=True)
                for er in erows:
                    calls = ", ".join(f"{n}.{op}@{st}" for n, op, _, st, _ in er["ops"])
                    print(f"      (b={er['b']},hv={er['hv']},c={er['c']}): {calls}", flush=True)
                for line in explain_aqk_credits(erows):
                    print(f"      credit: {line}", flush=True)
        per_head = " ".join(f"{h['replay_rel_l2']:.3e}" for h in r["heads"])
        print(f"{tag} replay: bitwise-correct heads {correct_heads(r['heads'], 'replay_bitwise', args.good)} | "
              f"rel-L2<{args.good:g} heads {correct_heads(r['heads'], 'replay_rel_l2', args.good)} | "
              f"o rel-L2 per head [{per_head}] | final_state rel-L2 max {max(h['state_rel_l2'] for h in r['heads']):.3e}",
              flush=True)
        wrong = [h for h in r["heads"] if not h["replay_bitwise"]]
        if wrong:
            print(f"    signature of the wrong heads ({len(wrong)}): o rel-L2 {min(h['replay_rel_l2'] for h in wrong):.3e}"
                  f"..{max(h['replay_rel_l2'] for h in wrong):.3e}, max_abs_diff {min(h['replay_max_abs'] for h in wrong):.3e}"
                  f"..{max(h['replay_max_abs'] for h in wrong):.3e}, |o|/|ref| {min(h['replay_norm_ratio'] for h in wrong):.4f}"
                  f"..{max(h['replay_norm_ratio'] for h in wrong):.4f}, all finite={all(h['replay_finite'] for h in wrong)}; "
                  f"final_state of the same heads rel-L2 <= {max(h['state_rel_l2'] for h in wrong):.3e}, "
                  f"max_abs_diff <= {max(h['state_max_abs'] for h in wrong):.3e}", flush=True)
        on_pair = [h for h in r["hazards"] if re.search(rf"recurrent[\w]*\.py:{anchors['read']}:", h)
                   and re.search(rf"recurrent[\w]*\.py:{anchors['write']}:", h)]
        if len(on_pair) != len(r["hazards"]):
            problems.append(f"{tag}: {len(r['hazards']) - len(on_pair)} hazard(s) not on the Aqk read/write pair")
            for h in [h for h in r["hazards"] if h not in on_pair][:5]:
                print("    OTHER HAZARD:", re.sub(r'loc\("[^"]*/', 'loc("', h), flush=True)
        if len(on_pair) != len(racy):
            problems.append(f"{tag}: {len(on_pair)} pipesim hazards but {len(racy)} attributed unordered reads")
        ref_dev = max((float((r["o"][h["b"], h["hv"]].float() - r["replay"][h["b"], h["hv"]].float()).abs().max())
                       for h in r["heads"] if h["replay_bitwise"]), default=0.0)
        print(f"    functional o vs reference on race-free heads: max_abs {ref_dev:.3e} (tolerance {REPRO_TOL:.1e}); "
              f"replay self-check max_abs {r['replay_self_check']:.3e}", flush=True)
        if ref_dev > REPRO_TOL:
            problems.append(f"{tag}: functional o differs from the reference by {ref_dev:.3e}")
        if other is not None:
            r2 = run_case(other[0], other[1], B=B, HV=hv, C=c, bd=bd, pipesim=True, seed=args.seed, timeout=args.timeout)
            eq_o, eq_s = torch.equal(r["replay"], r2["replay"]), torch.equal(r["final_state"], r2["final_state"])
            print(f"    vs {args.bitwise_against} (replay): o bitwise={eq_o} final_state bitwise={eq_s}", flush=True)
            r["vs_other"] = (eq_o, eq_s)
        by_shape.setdefault((B, hv, c), []).append((bd, r))
        rows.append({"B": B, "bd": bd, "HV": hv, "C": c, "hazards": len(r["hazards"]), "heads": r["heads"],
                     "unordered_reads": racy, "vs_other": r.get("vs_other")})

    print("bd invariance (hard criterion: replay o and final_state bitwise equal across bd for the same B, HV, C):")
    for (B, hv, c), runs in sorted(by_shape.items()):
        for bd, r in runs[1:]:
            eq_o = torch.equal(runs[0][1]["replay"], r["replay"])
            eq_s = torch.equal(runs[0][1]["final_state"], r["final_state"])
            print(f"  B={B} HV={hv} C={c}: bd={runs[0][0]} vs bd={bd}: o bitwise={eq_o} final_state bitwise={eq_s}", flush=True)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=1, default=str))
    if problems:
        print("DIAGNOSTIC PROBLEMS (the attribution does not explain every hazard):", *problems, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
