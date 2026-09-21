# GDN-2 chunk fwd/bwd (training) on A2 (Ascend910B3 / c220) — design & plan

Status: FOUNDATION. Decode (`gdn2_fused_recurrent`) is complete on A2 — 16.0x eager /
312.9x with NPU graph vs the torch_npu baseline, correct on real silicon. This file is
the training-path workstream (chunk-parallel fwd + bwd): a multi-stage effort begun here.

## Oracle (already exists)
`ascend_fla/reference/gdn2.py :: gdn2_recurrent_reference` — the token-by-token
recurrence. The chunk kernel must reproduce its `o [B,T,H,V]` and `final_state` over a
full chunk (L = 64 or 128), in fp32, within the unit relative-L2 gate. Second oracle:
the torch_npu composition (also the performance baseline).

## GDN-2 recurrence (per token/head) — the semantics to chunk-parallelize
    q,k,g,b in [K]; v,w in [V]; state [K,V]
    state *= exp(g)                    # channel-wise decay on K
    erase  = (b * k_norm) @ state      # key-side erase gate b
    state  += k_norm (outer) (w*v - erase)   # value-side write gate w, rank-1 update
    out    = q_norm @ state
Dual gates (b key-side, w value-side) plus channel-wise g distinguish GDN-2 from GDN/KDA
(KDA has a single scalar beta). So the KDA chunk kernels are a STRUCTURAL template only,
not a drop-in.

## Fwd decomposition (mirror `ops/kda/chunk.py`; 5 sub-kernels)
1. gate      : per-chunk cumulative log-decay from g[K] (log-space cumsum), in the
               numerically-stable form (see gate-range). Pure vector.
2. scores    : intra-chunk A = tril(q_g @ k_g^T), dual-gated, strict lower-triangular.
               Cube (mmad) + vector masking.
3. wy        : WY / UT-transform, apply (I - tril(diag(b) A))^-1 to the write term.
               Cube for the solves + vector.
4. inverse   : the triangular-inverse block feeding wy.
5. recurrent : inter-chunk state carry + output o. Cube (mmad) for q@state and k^update.

## The blocker: gdn2-chunk-gate-range
Real T=4096 stress (gaps.json): single-token -g up to 60.9; 64-token cumulative span up
to ~1461, far beyond KDA-stable's validated domain — exp() of a 1461 span overflows fp32.
Decode is immune (per-step exp(g_t) only); chunk pairs exp(a_i - a_j) via matmul and MUST
be designed stable:
 - factor exp(a_i - a_j) into two one-sided factors with a MID-ANCHOR at the chunk
   midpoint, so each factor is bounded by exp(+/- span/2) — doubles the usable range;
 - chunked log-sum-exp for the intra-chunk reductions; subtract the row/col max;
 - keep decay in log space until the last possible matmul.
This is the core design risk. Validate against the recurrence at REAL gate spans
(deterministically calibrate g, like KDA `_calibrate_span`), never toy spans.

## Reuse from the completed decode kernel (proven c220 techniques)
 - brcb-spread ([1,128] -> [128,8], repeat=16) + full-tile grouped mul: 64-lane groups
   are one vector repeat, so per-row broadcast is src2 rep_stride=1 / blk_stride=0, and
   the rank-1 outer is src1 per-row + src2 rep_stride=0.
 - Newton-refined rsqrt for q/k L2 norm (hardware fast rsqrt overshoots the 1e-4 gate).
 - [128,128] ops split to <=255 repeats; explicit rep-strides on strided [n,64] views;
   scalar math on [1,64] rows (unary ops misbehave on [8,8] tiles).
 - block_dim head-parallelism (build per-process only: one op, one build).
NEW vs decode: cube (mmad). Obey the c220 M-pipe settle (defect M10-081) for split-K
fp32 accumulation; a single K=128 tile is not split-K, but the intra/wy solves may be.

## Bwd (after fwd passes): dq, dk, dv, dg, db, dw, dh0
Mirror the kda_bwd (finalize_pre / finalize_post) structure; consumes the fwd
checkpoints (g_cumsum, per-chunk states, v_new). The backward gate span has its own
(tighter) stability budget — KDA found bwd precision-limited before finiteness — so set
MAX_GATE_SPAN as {forward, backward} separately, each pinned to measured data.

## Stage plan (each a real milestone, per the repo pipeline)
 S1 SPEC   : dtypes, chunk length L, token-major ABI, feature priorities.
 S2 golden : recurrence over a chunk + calibrated large-span stress cases.
 S3 DESIGN : tile/cube layout, the stable gate math, per-module contracts.
 S4 develop: the 5 fwd sub-kernels -> validate vs the recurrence on silicon; then bwd.
 S5 perf   : profile + NPU graph (as decode).

## Immediate next steps
 1. Write SPEC + emit the chunk-fwd golden (recurrence over L=64 and L=128 chunks,
    with a calibrated-span stress case).
 2. Prototype the STABLE gate sub-kernel first (mid-anchor log-space cumsum) and verify
    exp-domain bounds numerically before wiring scores/wy — the gate is the risk.
 3. Bring up cube (mmad) with a minimal q@state matmul unit on a2 before the full scores
    kernel, to shake out the M10-081 M-pipe-settle discipline in isolation.

## PROGRESS (validated this session)
- CUBE de-risked on a2: minimal [128,128]@[128,128].T matmul (L1->L0C, mode=cube)
  runs on real b3 silicon via the runtime bridge, max abs err 0.000 (exact). The
  matmul capability every chunk sub-kernel needs is confirmed. (single K=128 tile =
  not split-K; the split-K M10-081 path is still to be exercised by wy/scores.)
- TRAINING FORWARD validated: the existing gdn2_fused_recurrent kernel, run at
  training length via the bridge, matches the reference recurrence at
  T=64 (o relL2 5.4e-6, state 2.7e-6) and T=128 (o 5.7e-6, state 2.5e-6), finite.
  The recurrence is numerically SAFE at any T (per-step exp(g<=0)<=1), so this is a
  correct training forward with no gate-range issue. It is sequential (O(T)); the
  chunk-parallel form is the PERFORMANCE optimization (and the only place the
  gate-range blocker applies).

## REMAINING
- Training BACKWARD (dq,dk,dv,dg,db,dw,dh0): reverse-mode of the recurrence. Ground
  truth = torch autograd through reference/gdn2.py. Derive the reverse recurrence
  carefully (the erase term depends on the same-step decayed state -> not a plain
  linear recurrence), validate each gradient vs autograd, then implement as a fused
  reverse-recurrence kernel (mirror kda_bwd structure). This is the next focused
  effort; do the math derivation + a numpy/torch prototype BEFORE the kernel.
- Chunk-parallel fwd (perf): the 5-kernel cube+vector pipeline with the stable
  mid-anchor gate math (gate-range). Optimization over the correct recurrence fwd.

## BACKWARD MATH — DERIVED + VALIDATED (this session)
- bwd_reference_proto.py: the reverse-recurrence backward, all 7 gradients
  (dq,dk,dv,dg,db,dw,dh0) match torch autograd to ~1e-16 (fp64, single head).
  This is the validated oracle the backward kernel must reproduce. Per-step ops are
  the same primitives as the forward decode kernel (matvec S@do, outer qn^do,
  dkn=dS@delta, ddelta=kn@dS, dbk=S_dec@derase, dg=rowsum(dS_dec*S_prev)*eg,
  dS_prev=dS_dec*eg, l2norm-backward for dq/dk), so it maps onto the proven
  _spread8/_rowscale/_outer/_reduce_rows toolkit.
- REMAINING: lower it to a fused reverse-recurrence kernel (cache/recompute the
  forward states), validate on silicon vs this oracle; then the chunk-parallel perf form.

## BACKWARD KERNEL — IMPLEMENTED + VALIDATED ON SILICON (this session)
- kernels/projects/a2/gdn2_recurrent_bwd/kernels/step.py + golden.py
- All 7 gradients (dq,dk,dv,dg,db,dw,dh0) match the autograd-checked golden on real
  910B silicon: relL2 ~1e-6 (B1/T4/H2, block_dim=8, via the runtime bridge).
- Reverse recurrence uses the forward toolkit + the new _reduce_cols (validated
  separately on a2: reduce [K,V] over V -> contiguous [1,K] via halve-to-8 + cadd).
- Operator reduction (all re-validated on silicon): 708 -> 680 (share the rsqrt
  between q/k normalization and the l2norm-backward) -> 529 (phase-2-only: the
  per-step forward states are taken as the `states` input from the training
  forward, instead of recomputing the whole forward inside the backward).
- ABI: inputs q,k,v,g,erase_gate,w,initial_state,dout,dfinal,states(checkpoints);
  outputs dq,dk,dv,dg,db,dw,dh0. states[B,H,T+1,K,V] must be produced by the
  training forward (standard checkpoint flow). Gotcha: an input named `do` collides
  with a C++ keyword in the generated aclnn wrapper -> use `dout`. `TP1`(=T+1) is a
  required HostSpec scalar (symbolic GM dim).
- NEXT: a states-outputting training-forward variant + an autograd.Function tying
  fwd+bwd; then the chunk-parallel perf form.
