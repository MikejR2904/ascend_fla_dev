# A2-07 — GDN ABI plan for Qwen3-Next on A2 (research + design, no kernel)

- Wave / SoC: W0 / a2 (host-side; desktop research + `ascriptor check` / sim for evidence)
- Deliverable: this file (`docs/research/a2_gdn_abi.md`). **No kernel written or changed.**
- Depends on: A2-01 (split-K hit-table). A2-01 is still `open` at time of writing, so §2
  derives the split-K exposure from the kernel contracts directly and marks the points that
  A2-01 must confirm on silicon.
- Sources read: `docs/matrix/gaps.json`, `docs/matrix/models.json`,
  `ascriptor-kernels/projects/a5/{gdn_fwd,gdn_bwd,kda_fwd,kda_bwd}/contract.json`,
  `fla/ops/gated_delta_rule/{gate.py,chunk.py,wy_fast.py,fused_recurrent.py}`,
  `fla/layers/gated_deltanet.py`, `AGENTS.md` §2/§6/§7.

## 0. Target and the shape of the problem

Qwen3-Next-80B-A3B (`docs/matrix/models.json` → `qwen3-next-80b-a3b`) is the second wave-0
model and runs GDN (Gated DeltaNet). Its GDN mixer shape:

| field | value | source |
|---|---|---|
| hidden_size | 2048 | config.json |
| num_key_heads (H) | 16 | config.json |
| num_value_heads (HV) | 32 | config.json |
| head_k_dim / head_v_dim | 128 / 128 | config.json |
| conv_size | 4 | config.json |
| num_hidden_layers | 48 (36 linear GDN + 12 full-attn, interval 4) | config.json / derived |
| head_group_ratio (HV/H) | 2 | derived |
| key_dim / value_dim | 2048 / 4096 | derived |
| dtype | bfloat16 | config.json |

`head_k = head_v = 128` and BF16 match the fixed-size ABI exactly. **The only hard blocker is
the head grouping `HV=32 > H=16`** (`gdn-no-gqa`). Five more gaps ride along (layout, nonzero
initial state, `dh0`, state dtype, decode kernel), plus `scale-param-no-slot` and `no-tail-path`.
All of them are already solved for KDA (`AGENTS.md` §2), so KDA is the working reference for
every fix. **KDA's numbers must not be copied to GDN** (`AGENTS.md` §8): gate span, split-K
exposure and dtype behaviour are re-derived here for GDN and must be re-measured on a2 silicon.

The a5 `gdn_fwd` five stages map onto fla like this (`AGENTS.md` §2, cross-checked against
`fla/ops/gated_delta_rule`): preprocess (g cumsum / decay mask) ↔ `gate.py`
(`gdn_gate_chunk_cumsum_scalar_kernel`, `b_o = cumsum(-exp(A_log)*softplus(g))`); inverse
(strictly-lower-triangular solve) ↔ `ops/utils/solve_tril.py`; recompute (WY) ↔ `wy_fast.py`;
scores ↔ `chunk_o.py`; recurrent (inter-chunk state) ↔ `chunk_delta_h.py`. `gdn_bwd` carries all
of these plus the reverse saved values.

## 1. The six ABI gaps, one at a time

For each gap: **現状 (with source)** → **fix (referencing KDA)** → **kernels touched** →
**contract change** → **numerical risk** → **validation plan (fp32 dual oracle)** → **effort**.
The dual oracle throughout is fla's own `naive`/`naive_recurrent` versus fla's `chunk`, both in
fp32, plus the repository's per-token↔chunk cross-check; correctness is judged in fp32 by
relative-L2 / max_abs_diff (`AGENTS.md` §6), never bitwise, never bf16-elementwise.

### 1.1 gdn-no-gqa — no independent value-head dimension (the hard blocker)

- **現状**: `gdn_fwd/contract.json` `domain.shape` = "B,H,C are positive runtime dimensions;
  fixed L=64, D=128" — there is **no HV**. The recurrence and output are indexed by a single
  head axis H. `kda_fwd`/`kda_bwd` by contrast declare `H_HV: "positive; HV % H == 0"`.
- **Fix (KDA)**: add the value-head axis exactly as KDA does. KDA maps each value head to its qk
  head by `i_h = i_hv // (HV // H)` and broadcasts the qk-side tensors (q, k, g, b) across the
  `HV/H` group before the kernel body; only v and the state carry the HV axis. For Qwen3-Next
  `HV/H = 2`, so each qk head feeds two value heads. This is Grouped-Value-Attention (GVA), the
  same primitive KDA already ships.
- **Kernels touched**: `gdn_fwd`, `gdn_bwd` (both add the HV axis to v, state, output, and the
  saved histories; qk-side inputs stay H and are broadcast internally).
- **Contract change**: `domain.shape` gains `HV: "positive; HV % H == 0"`; `value`, `output`,
  `final_state`, `state_after_history`, `v_new_history` re-shaped `[B,HV,C,L,D]`; `query/key/g/beta`
  stay `[B,H,C,L,D]`.
- **Numerical risk**: none new — broadcasting is exact; the arithmetic per (b,hv) is unchanged.
- **Validation**: pick a case with `HV != H` (e.g. H=2, HV=4) and an *asymmetric* per-group value
  so a wrong `i_h` mapping cannot pass (KDA's `grouped_heads` case is the template). fp32 dual
  oracle on o and final_state.
- **Effort**: M. Mostly index/broadcast plumbing across fwd+bwd + the histories; no new math.

### 1.2 layout-not-token-major — public layout is `[B,H,C,L,D]`

- **現状**: `gdn_fwd/contract.json` `domain.layout` = "Contiguous CPU tensors in [B,H,C,L,D]".
  fla and every real caller are token-major (`[B,T,H,D]`). `kda_fwd` already declares
  "Contiguous token-major public tensors; explicit local permutation to BHCLK/BHVCLK kernel
  tensors".
- **Fix (KDA)**: keep the public interface token-major and permute to the kernel's chunk-major
  layout *inside* the launcher/kernel, as KDA does — do not push the permute onto the caller.
- **Kernels touched**: `gdn_fwd`, `gdn_bwd` entry/launcher (layout adapter), no change to the
  inner compute.
- **Contract change**: `domain.layout` → "token-major public `[B,T,H/HV,D]`; internal permute".
- **Numerical risk**: none (permute is a copy); watch that the permute is a real materialization,
  not a strided view handed to a kernel that needs contiguous (see `npu-builtin-ops-missing`:
  strided D2H/Slice is not universally available in the opp package).
- **Validation**: round-trip a non-square `[B,T,H,D]` input and assert the permuted kernel output
  matches the token-major reference; assert contiguity at the kernel boundary.
- **Effort**: S. Adapter only.

### 1.3 nonzero-initial-state — only zero initial state supported

- **現状**: `gdn_fwd/contract.json` `domain.initial_state` = "Zero initial state only".
  `kda_fwd` takes a float32 `[B,HV,128,128]` initial_state as a first-class input with both random
  and zero cases.
- **Fix (KDA)**: add the initial_state input the way `kda_fwd` does — fp32, `[B,HV,K,V]`, loaded
  into the state tile before the first chunk. Until the kernel supports it, **the layer gate must
  reject any nonzero initial_state passed to GDN** (`AGENTS.md` §7: no silent wrong result).
- **Kernels touched**: `gdn_fwd` (state init), and the recurrent/decode kernel of §1.6.
- **Contract change**: `domain.initial_state` → "fp32 `[B,HV,128,128]`, zero or arbitrary".
- **Numerical risk**: the state is a cross-chunk accumulator; feeding a nonzero fp32 state is
  numerically clean, but see §1.5 — if final_state is bf16, a fed-back state loses precision each
  segment. Fix §1.5 first or together.
- **Validation**: chain two segments (`state = fwd(seg0); o1 = fwd(seg1, state)`) and compare to a
  single `fwd(seg0++seg1)`; fp32 relative-L2. This is the decode/prefill-handoff correctness gate.
- **Effort**: S–M (couples with §1.5, §1.6).

### 1.4 d-initial-state-absent — backward emits no `dh0`

- **現状**: `gdn_bwd/contract.json` — "no d_initial_state in the preserved backward production
  ABI". `kda_bwd` emits `dh0 [B,HV,128,128]` and accepts a `dht` input.
- **Fix (KDA)**: emit `dh0` from the reverse recurrence exactly as `kda_bwd` does, and accept the
  incoming `dht` cotangent. This is what makes trainable initial state / sequence-parallel /
  segmented backward possible. Until then the gate must reject `initial_state.requires_grad`.
- **Kernels touched**: `gdn_bwd` (reverse recurrence tail + a new output).
- **Contract change**: `gdn_bwd` outputs gain `dh0 [B,HV,128,128]`; inputs gain `dht`.
- **Numerical risk**: none new; the reverse recurrence already carries the state cotangent to
  step 0 — `dh0` is that value (the repository's own a2 gdn2 backward already emits it, so the
  math is known-good).
- **Validation**: autograd of an fp32 reference recurrence with a `requires_grad` initial state vs
  the kernel `dh0`; relative-L2. (The a2 `gdn2_recurrent_bwd` work validated this exact quantity to
  ~1e-6 and is a ready template.)
- **Effort**: S.

### 1.5 state-dtype-bf16 — `final_state` is BF16, fla convention is FP32

- **現状**: `gdn_fwd/contract.json` `outputs.final_state.dtype` = bfloat16. `kda_fwd`'s
  final_state is float32.
- **Fix (KDA)**: make `final_state` (and `state_after_history`) fp32, matching KDA and the fla
  convention. The state is a cross-chunk accumulator; bf16 rounding enters the next segment's
  recurrence. **`AGENTS.md` and the corrected Gemini table both fix state at fp32** — bf16 state is
  the `state-dtype-bf16` gap, not an option.
- **Kernels touched**: `gdn_fwd` (state output dtype), the recurrent/decode kernel, and `gdn_bwd`
  saved-state reads.
- **Numerical risk**: this *is* a numerical-risk gap. The size of the bf16→fp32 improvement is
  **unmeasured on any SoC** and must not be assumed from A5. Measure the o / final_state delta
  between a bf16-state and fp32-state build at long C (e.g. C=64), and whether it grows with span.
- **Validation**: same-shape bf16-state vs fp32-state build; report the divergence and its growth
  with C and with gate span. fp32 recurrent oracle as ground truth.
- **Effort**: S (dtype change) + measurement.

### 1.6 fused-recurrent-missing — no GDN decode kernel

- **現状**: `gaps.json` — the ascriptor catalog has no `fused_recurrent` GDN unit; the six a5
  units are all chunk-path. KDA's half is done (`kernels/projects/a5/kda_fused_recurrent`: state
  resident in UB, two-pass scan, partitioned by `B*HV` one-core-per-head with no inter-core sync,
  `ascriptor check` 0/0/156 ops, silicon o relative-L2 9.6e-8–1.8e-7). GDN/DeltaNet decode is
  still entirely missing.
- **Fix (KDA)**: author a GDN `fused_recurrent` unit copying the `kda_fused_recurrent` structure —
  two-pass scan, one core per head, no inter-core sync — extended for GVA (§1.1), the fp32 nonzero
  state in/out (§1.3/§1.5) and, for GDN, the per-token gate `exp(g_i)` applied inside the scan.
  Because decode applies only `exp(g_i)` per step (magnitude ~1, never a cumsum), the decode path
  **has no gate-span ceiling** — this is the same property KDA's decode gained, and it is the
  reason decode must exist separately from chunk (see §3).
- **Kernels touched**: new `gdn_fused_recurrent` (a2 unit under `kernels/projects/a2/`), plus the
  layer cache adapter.
- **Contract change**: new unit; ABI mirrors `kda_fused_recurrent` with the HV axis.
- **Numerical risk**: low; the per-step `exp(g_i)` is well-conditioned. The correctness invariant
  is the **per-token↔chunk identity**: T single-step calls with chained state must be bit-identical
  to one T-token call (this is exactly how KDA's decode correctness was pinned).
- **Validation**: (a) fp32 recurrent oracle on o/final_state at 8 shapes; (b) the per-token↔chunk
  identity (relative-L2 = 0); (c) `block_dim` sweep to the physical vector-core ceiling — on a2
  that is 40 vector cores (`Ascend910B3.ini vector_core_cnt=40`), **not** the a5/950 ceiling; this
  is a per-SoC number A2-01 must confirm (`AGENTS.md` §8).
- **Effort**: L. New kernel family; the KDA unit is a close structural template.

### 1.7 Coupled gaps: scale-param-no-slot and no-tail-path

Two more gaps ride with the batch (`gaps.json`), not part of the core six but blocking the same
model:

- **scale-param-no-slot**: no `scale` scalar entry in the GDN ABI. fla multiplies q by
  `head_dim**-0.5 ≈ 0.0884`; without a slot this becomes a host pre-multiply (one extra full-tensor
  pass). Fix (KDA): `kda_sub2_score_kernel`/`kda_sub45_fused_kernel` already carry a `scale: f32`
  scalar; add the same to `gdn_fwd` and absorb it at q's existing read point. Contract: `scalars`
  gains `scale: f32`. Effort: S. Folded into Batch A2-K1-GDN-a.
- **no-tail-path**: `L=64` fixed, `T` must be a multiple of 64 (`gdn_fwd` domain: "fixed L=64 …
  No L/D tail path"). fla/real inference use arbitrary T. Short term: the gate rejects non-multiples
  with the nearest legal T, and any host padding is charged in the perf report (`AGENTS.md` §6). A
  real tail path (partial chunk) is a later, larger kernel change — **kept out of this batch** and
  filed as an open item; both real target models tolerate host-side padding for now.

## 2. Split-K exposure of the GDN kernels (feeds / overlaps A2-01)

A2-01 owns the authoritative split-K hit-table; it is not yet produced. From the contracts:
`gdn_fwd` `domain.core_ownership` = "BHC producers partition tiles; recurrent kernels partition BH
and retain the C recurrence. Two vector subblocks cooperate per cube core." So:

- The **chunk producers** (inverse / WY / scores) partition over `B*H*C` tiles — split-K-friendly,
  bounded by tiles, not by the recurrence.
- The **recurrent stage** partitions over `B*H` (→ `B*HV` after §1.1) and keeps the C loop serial
  per head — **the C recurrence is not split**; parallelism is exactly `B*HV`. For Qwen3-Next a
  single sequence gives `B*HV = 1*32 = 32` work items, which under-fills the 40 a2 vector cores by
  itself; batch or the V-column independence would be needed to saturate (a design input for the
  decode kernel, mirroring the a2 gdn2 finding that block_dim should target the 40-core ceiling).
- `block_dim` ceiling on a2 is **40 vector cores** (`vector_core_cnt=40`), distinct from a5/950's
  56. A2-01 must confirm the real launch ceiling and the per-stage active-core counts on a2
  silicon; this document assumes the platform-ini value and flags it as unverified.

**Item A2-01 must return**: measured active Cube/Vector cores per stage at the Qwen3-Next shape,
and whether the recurrent stage's `B*HV`-only parallelism is the binding limit at B=1.

## 3. Range table — every exp / division / cumsum in the GDN chain

Methodology per `AGENTS.md` §6: list each transcendental / division / cumulative op, give the
argument range under **real Qwen3-Next init**, and the verdict (safe / underflow-is-correct /
overflow-hazard). fp32 `exp` underflows to 0 at arg ≈ −87.3 (`FLT_MIN_NORMAL`) and overflows at
≈ +88.7 (`ln FLT_MAX`); `exp2` shifts these to ≈ ±126; bf16 shares the fp32 overflow line.

| # | op | where | argument range (Qwen3-Next init) | verdict |
|---|---|---|---|---|
| 1 | `A_log.exp()` | `gate.py` `naive_gdn_gate` / cumsum kernel | `A = U(0,16)` → `exp(A_log) ∈ [0,16]` | safe (bounded ≤16) |
| 2 | `softplus(g+dt_bias)` | same | ≥0; at init ≈ `dt ∈ [0.001,0.1]`; trained can grow | safe (monotone, no overflow for realistic g) |
| 3 | `-exp(A_log)*softplus(·)` = per-token g | same | `∈ [−1.6, 0]` at init (16·0.1); trained larger | ≤0 by construction |
| 4 | `cumsum(g)` over a 64-chunk | `gdn_gate_chunk_cumsum` | `∈ [−span, 0]`, **span a random variable, init upper bound ≈ 16·0.1·63 ≈ 100.8** | **overflow/underflow hazard — see below** |
| 5 | `exp(cumsum)` / `exp2(cumsum·log2)` decay | preprocess / decay mask | `exp(−span)`; at span 100.8, `exp(−100.8)=0` in fp32 | **underflow to 0 → 0×inf / 0/0 downstream** |
| 6 | `exp(g_i − g_j)` pairwise decay (chunk) | scores / WY | `∈ [exp(−span), exp(+span)]`; `+span` side **overflows** in fp32/bf16 past 88.7 | **overflow hazard (bwd finalize)** |
| 7 | `k / eg`, `eg_last / eg` divisions | intra / WY | `eg = exp(cumsum)`; when eg underflows to 0 → `x/0` | **NaN when #5 underflows** |
| 8 | `exp(g_i)` per-token (decode) | `fused_recurrent` | magnitude ~1 (single step, no cumsum) | **safe — no span ceiling** (why decode is separate) |
| 9 | RMSNorm `1/sqrt(mean x²+eps)` | o_norm / q,k l2norm | mean of squares ≥0; eps guards 0 | safe |

**GDN gate span, estimated separately (not KDA's 94).** GatedDeltaNet inits (fla
`layers/gated_deltanet.py:151-165`) `A = uniform(0,16)`, `A_log = log(A)`, and
`dt ∈ [dt_min=0.001, dt_max=0.1]` with `dt_bias = inv_softplus(dt)`. Per-token
`−g = exp(A_log)·softplus(a_proj(x)+dt_bias)`; at init `exp(A_log) ≤ 16`, `softplus(dt_bias) ≈ dt
≤ 0.1`, so per-token `−g ≤ 1.6` and the 63-step chunk cumulative **upper bound ≈ 100.8** — the same
*structural* bound as KDA, but the *distribution differs*: KDA draws `exp(A_log)=U(1,16)`, GDN
draws `U(0,16)` (more mass near 0, so a lower median but the same ceiling). With `HV=32` value
heads the span is `max` over 32 draws, so it sits **close to the 100.8 ceiling** at init (KDA
already showed HV=8 reaching 100.6). **Do not reuse KDA's observed 94.** Two extra cautions
specific to Qwen3-Next: (a) it is a *trained* checkpoint — `A_log`/`dt_bias` move, and a trained
GDN-family probe (`models.json` gdn2 `gate_domain_probe`) hit a 64-token span of **1461** on
adversarial valid tokens; the natural-text span must be measured on the real checkpoint, not
assumed from init; (b) the a5 `gdn_fwd/bwd` contracts only test *slow decay* (`domain.values`:
"Verification includes ... slow decay"), so the deep-span underflow/overflow of rows #4–#7 is
**untested** in the current units — exactly the situation KDA was in before `kda_fwd_stable` /
`kda_bwd_stable`.

**Conclusion of the range analysis**: GDN needs the same **midpoint-anchored symmetric
decomposition** KDA's stable units use (`exp(a_i−a_j)` split into two factors each bounded by
±span/2, doubling the finite-range ceiling to ~174), for **both** forward (underflow, row #5/#7)
and backward (overflow, row #6). The `MAX_GATE_SPAN` gate must be two-dimensional
`{impl:{forward,backward}}` as for KDA, and must cover the init ceiling 100.8 so the default model
is not rejected — but the *backward* accuracy ceiling (KDA's was 105) must be re-measured for GDN,
not copied.

## 4. torch_npu ops needed for the rest of Qwen3-Next (checklist, coverage tested at A2-10)

Real-hardware coverage is an A2-10 task; here is only the list and the "what breaks if missing".

| layer | op(s) needed | if missing |
|---|---|---|
| GDN mixer non-kernel parts | `silu` (short conv activation), `l2_normalize` (q,k), `sigmoid` (beta), `softplus` + `exp` (gate), `cumsum` | gate/qk prep falls to host — extra full-tensor passes, the `scale`/`qk-l2norm-not-in-kernel` overhead |
| short convolution | depthwise `conv1d` with cache (causal, `T<kernel_size` zero-pad) | decode conv-state handoff breaks (per-token, not average, error) |
| gated full attention (12 layers, interval 4) | SDPA / flash-attn `scaled_dot_product_attention`, RoPE (`rotary`), `softmax` | the 12 non-linear layers can't run; needed for end-to-end |
| MoE (A3B = sparse) | `topk`, `softmax`(router), grouped/`batched_matmul`, `scatter/gather` (expert dispatch) | MoE FFN can't run; this is the bulk of the 80B params |
| norms | `rms_norm` (block + `FusedRMSNormGated` with sigmoid gate), `layer_norm` | every block boundary; `FusedRMSNormGated` gate is GDN-specific |
| embedding / head | `embedding`, final `rms_norm`, `linear` (lm_head), `argmax`/sampling | I/O boundary |

Gaps to file after A2-10 measures them: any of the above absent from the a2 opp package
(`npu-builtin-ops-missing` already tracks strided `Slice`; watch `FusedRMSNormGated`, MoE
`scatter/gather`, and flash-attn availability on 910B3 specifically — **do not assume from A5**).

## 5. Proposed kernel batch decomposition (for PM → user approval)

The six gaps are all kernel changes; they split into three batches by dependency and risk. Each
batch is a single-kernel-rule-respecting unit of work with its own fp32 dual-oracle acceptance.

**Batch A2-K1-GDN-a — chunk ABI alignment (fwd+bwd), no new math.**
Content: §1.1 GQA/HV axis, §1.2 token-major internal permute, §1.3 nonzero fp32 initial state,
§1.4 `dh0` in bwd, §1.5 fp32 `final_state`, §1.7 `scale` kernel slot. Depends on: A2-01 (split-K
hit-table), A2-16. Acceptance: fp32 dual oracle on o/final_state/all grads at an asymmetric
`HV!=H` case + segment-chaining + `dh0` autograd match; state-dtype bf16-vs-fp32 divergence
measured and reported. Risk: M (plumbing, no algorithm change). This unblocks Qwen3-Next chunk
prefill/training **except** deep gate span.

**Batch A2-K1-GDN-b — gate-range stabilization (the numerical batch).**
Content: midpoint-anchored decomposition for GDN fwd (underflow) and bwd (overflow) per §3, plus a
two-dimensional `MAX_GATE_SPAN{forward,backward}` gate sized to cover the 100.8 init ceiling.
Depends on: A2-K1-GDN-a. Acceptance: fp32 recurrent oracle (the only valid oracle past span 80)
across a span sweep to ≥100.8 with all six gradients finite and within a **GDN-re-measured**
budget; explicit gate-rejection test above the backward ceiling. Risk: M–H (this is where KDA
spent the most effort; the method transfers, the thresholds do not).

**Batch A2-K1-GDN-c — decode kernel (`gdn_fused_recurrent`).**
Content: §1.6 new decode unit (two-pass scan, one core per head, no inter-core sync, GVA, fp32
state, per-token `exp(g_i)`), plus the layer cache adapter. Depends on: A2-K1-GDN-a. Acceptance:
fp32 oracle at 8 shapes + per-token↔chunk identity (relative-L2 = 0) + `block_dim` sweep to the a2
40-core ceiling. Risk: L–M (KDA's `kda_fused_recurrent` is a close template). No gate-span ceiling
on this path.

Sequencing: A2-K1-GDN-a first (unblocks the most), then -b and -c in parallel (independent).
Non-kernel host work (layer wiring, torch_npu op coverage, MoE / full-attn) is A2-10 and later, out
of this batch's scope.

## 6. Open items / honest gaps in this research

- **A2-01 not yet done** (§2): the split-K hit-table and per-stage active-core counts on a2 are
  assumed from the platform ini, not measured. A2-K1-GDN-a should not be sized final until A2-01
  lands.
- **Deep source-line citation**: this doc cites the contract fields and `gate.py` functions; the
  per-stage line-level mapping into `chunk.py`/`wy_fast.py`/`solve_tril.py`/`chunk_o.py`/
  `chunk_delta_h.py` is sketched at the function level (§0) and should be expanded when the batch
  specs are written.
- **Trained-checkpoint gate span**: estimated from init (~100.8) and bounded above by the
  adversarial gdn2 probe (1461); the natural-text span on the real Qwen3-Next checkpoint is a
  measurement A2-10 owns.
- No kernel was written or modified (acceptance item 4).
