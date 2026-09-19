# PGDN chunk forward: frozen ABI, read/write keys and numerical range

PK-03 follows FLA commit `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`.
The semantic authority is its
[`precond_gated_delta_rule/naive.py`](https://github.com/fla-org/flash-linear-attention/blob/e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2/fla/ops/precond_gated_delta_rule/naive.py),
SHA-256 `3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8`.
The public entry is `ascend_fla.ops.pgdn_chunk_fwd.chunk_pgdn`.
This is an inference forward slice, with no backward, decode, checkpoint,
training-state continuation or performance optimization support.

## ABI comparison

The pinned [`chunk.py:409`](https://github.com/fla-org/flash-linear-attention/blob/e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2/fla/ops/precond_gated_delta_rule/chunk.py#L409)
was read together with naive.py and the repository GDN wrapper at `20b5247`.
Defaults below are deliberate public restrictions, not inferred support for every
option appearing in the upstream signature.

| Item | Pinned FLA / existing GDN | This PGDN entry |
| --- | --- | --- |
| Input order | PGDN q,k,v,g_atk,g,beta_atk,beta; GDN has only g,beta | Same seven PGDN positional tensors |
| Layout | FLA token-major, variable shapes; GDN contiguous token-major | q,k `[B,T,H,128]`; v `[B,T,HV,128]`, contiguous |
| Heads | FLA naive repeats each key head consecutively for GVA; GDN supports HV divisible by H | Positive H/HV, `HV % H == 0`; value head j reads key head `j // (HV/H)` |
| T | GDN positive multiple of 64, at most 4096 | Same; no tails or empty inputs |
| Gates | PGDN g/beta per HV, g_atk/beta_atk per H | g,beta `[B,T,HV]`; g_atk,beta_atk `[B,T,H]`; all FP32 |
| Gate values | Layer produces nonpositive g/g_atk, sigmoid beta_atk; optional main beta doubling is upstream only | Finite g/g_atk ≤ 0, beta/beta_atk in [0,1]; FP32 chunk prefixes must remain finite |
| Dtypes | GDN FP32/BF16 qkv and FP32 internal stages | Matching FP32/BF16 qkv; exact widening on the input device; internal FP32 |
| Normalization | GDN performs none; PGDN naive always uses FP32 F.normalize | Enabled only; `use_qk_l2norm_in_kernel=False` rejected |
| scale | FLA default 1/sqrt(K); GDN fixed K=128 scale | None or exactly `128**-0.5` |
| Initial main state | FLA supports supplied state; GDN None only | None only; internal zero seed `[B,HV,128,128]` |
| Initial ATK state | FLA supports supplied state | None only; kernel initializes `[B,H,128]` |
| ATK x/eps | FLA defaults x=1.5, eps=1e-6 | Exactly those defaults |
| ATK center | FLA `log_atk_scale=None` selects -0.2; accepts tensor centers | None or scalar -0.2 only; even a tensor containing -0.2 is rejected |
| Output | FLA returns output and optional independent S/A | `(o, S, A)`; o has q dtype, S/A are FP32 or both None |
| Other options | FLA varlen/context/transpose features | Reject cu_seqlens, cu_seqlens_cpu, cp_context, head_first, transposed state |
| Execution | GDN A5 CCE with block_dim1/2 | Same target and block dimensions; no other target claim |
| Gradients | FLA supports autograd | Reject requires_grad inputs when grad mode is enabled |

The beta_atk bound comes directly from the pinned
[`layers/precond_gated_deltanet.py:324`](https://github.com/fla-org/flash-linear-attention/blob/e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2/fla/layers/precond_gated_deltanet.py#L324):
`b_atk_proj(hidden_states).sigmoid()`. Line 325 computes nonpositive g_atk;
lines 319–322 show main beta/g, including the optional beta×2 mode that this
slice rejects. Model-specific learned centers are not supported by this
fixed-default kernel.

The in-process wrapper requires all tensors on one NPU. Explicit board/aclnn
launchers take CPU tensors. They do not invoke a CPU reference as a fallback.
Both CPU and NPU callers receive value/shape/dtype checks before kernel launch;
NPU predicate reductions synchronize and belong to the measured public-call cost.
State tensors are rejected even when zero and correctly shaped; incorrect shapes
are diagnosed separately so the H/HV distinction cannot be silently broadcast.

## Exact key roles and chunk equations

At each token, in FP32:

```
q_norm = q / max(norm(q), 1e-12)
k_read = k / max(norm(k), 1e-12)
q_scaled = q_norm * 128**-0.5
A = exp(g_atk) * A_prev + beta_atk * k_read**2
r = log(A + 1e-6) + 0.2
M = exp(-log(1.5) * r / (1 + abs(r)))
k_write = k_read * M
D = exp(g) * S_prev
delta = beta * (v - k_read.T @ D)
S = D + k_write @ delta.T
o = q_scaled.T @ S
```

The correction uses **k_read**. Causal output scores and state writes use
**k_write**. Substituting k_write into every old GDN key occurrence is wrong.
A key-head owns A independently of its value-head group; main state S is per HV.

Within a chunk, let `p_i = sum(g_0..g_i)`. Define
`L_ij = beta_i * dot(k_read_i,k_write_j) * exp(p_i-p_j)` for j<i,
`U = solve(I+L, beta*v)` and
`W = solve(I+L, beta*k_read*exp(p))`.
Then `delta=U-W@S_in`,
`S_out=exp(p_last)*S_in + (k_write*exp(p_last-p)).T@delta`, and
`o=exp(p)*q_scaled@S_in + causal(q_scaled@k_write.T*exp(p_i-p_j))@delta`.
The solve has a unit diagonal; no inverse decay or data-dependent diagonal
reciprocal is introduced.

Reference A loads the exact pinned naive file at run time. Independent reference
B imports none of A, the production graph or the simulator. B evaluates ATK with
causal weighted sums over a chunk and evaluates the main update with one direct
triangular solve of `beta*(v-exp(p)*k_read@S_in)`. The split U/W equations are used
only to generate independent stage inputs. Tests compare grouped runs to separate
single-head calls and reject the erroneous all-write-key correction.

## Intentional normalization fork

Pinned naive.py:110–111 widens inputs before `F.normalize` with eps=1e-12.
Pinned chunk.py:295–297 instead calls `l2norm_fwd`; the implementation in
[`fla/modules/l2norm.py:104`](https://github.com/fla-org/flash-linear-attention/blob/e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2/fla/modules/l2norm.py#L104)
uses `x/sqrt(sum(x*x)+1e-6)` and writes the result in the input dtype.
These are different mathematical/rounding contracts. This kernel chooses naive
normalization in FP32. **It intentionally is not bitwise identical to the FLA
Triton chunk path near zero or at BF16 materialization boundaries.**

For a 128-wide FP32 row `(norm,0,...,0)`, CPU formula evaluation gives:

| Original norm | Naive normalized norm | Additive-epsilon formula norm | Maximum component difference |
| --- | ---: | ---: | ---: |
| 1e-2 | 1 | 0.9950372 | 0.0049628 |
| 1e-4 | 1 | 0.09950372 | 0.9004963 |
| 1e-6 | 1 | 0.0009999996 | 0.9990000 |
| 0 | 0 | 0 | 0 |

The tests compute this table, including the zero row, rather than assume the
formulas agree. A separate four-row BF16 random-input CPU probe (seed690319,
width128) measured relative L2 0.0017453919 and max_abs 0.0006299317 between naive
FP32 and the additive-epsilon formula followed by BF16 materialization. These
are CPU formula comparisons; no Triton or CUDA kernel was run. This is the same
class of difference as `kda-layer-l2norm-eps-formula-diverges` in
`docs/matrix/gaps.json`; that KDA record does not certify this PGDN implementation.

## Whole-chain range review

All six kernels use FP32 GM/UB/register storage. K/V are128, chunks are64.
For normalized finite keys, each component squared is ≤1; zero initial A,
0≤beta_atk≤1 and exp(g_atk)≤1 imply `0≤A_t≤t≤4096` in exact arithmetic.
With positive eps, log never sees zero or a negative input. The squash ratio
has absolute value <1, so `1/1.5 < M < 1.5` and `norm(k_write)≤1.5`.
These conclusions have been rederived for PGDN; normalized GDN key bounds
cannot simply be reused after preconditioning.

| Stage / expression | Range and failure consideration |
| --- | --- |
| ATK normalization sum and sqrt | Nonnegative FP32 sum of 128 squares; clamp denominator at1e-12 before division; zero maps to zero |
| exp(g_atk) | [0,1]; underflow to0 correctly discards previous A |
| beta_atk*k_read² and A update | Nonnegative; exact-arithmetic A≤4096 under the stated initial state/gates |
| log(A+1e-6) | Approximately [-13.81551,8.31777]; no invalid log argument |
| r=log(A+eps)+0.2 | Approximately [-13.61551,8.51777] |
| r/(1+abs(r)) | In (-1,1); denominator≥1 |
| exp(-log(1.5)*r/(1+abs(r))) | Exponent in (-0.405466,0.405466); no exp overflow/underflow |
| k_write=k_read*M | Component magnitude≤1.5; L2 norm≤1.5 |
| prepare q scale | q_scaled norm≤1/sqrt128, performed in FP32 |
| prepare beta*k_read | L2 norm≤1, unpreconditioned WY RHS |
| prepare beta*v | beta≤1 does not amplify a component; later dot/state sums still need magnitude headroom |
| prepare p=cumsum(g) | Nonpositive; require finite FP32 prefixes; cancellation in later differences is a precision consideration |
| score exp(p_i-p_j), j≤i | [0,1]; acausal entries are never exponentiated; underflow correctly removes a negligible causal edge |
| score q_scaled dot k_write | Absolute bound1.5/sqrt128 before decay |
| lower beta*k_read dot k_write | Absolute bound1.5 before decay; strict lower triangle |
| WY exp(p) | [0,1]; no exp(-p) factor |
| WY solve | Unit diagonal; repeated FP32 dot/subtractions, not a guarantee of arbitrary-input conditioning |
| scan exp(p_last) | [0,1]; state decay underflow removes previous-chunk influence |
| scan exp(p_last-p_i) | [0,1]; tail weighting has no positive exponential |
| scan correction and rank updates | Uses k_read-derived W for reads and k_write for writes; state/delta dot accumulations remain FP32 |
| output exp(p) and causal score product | Nonpositive exponent; q reads the updated main state, including the diagonal token |
| final BF16 o storage | Sole lossy public storage boundary; S/A remain FP32 |

Nonpositive exponents establish finite exp range, not an accuracy proof for
arbitrary finite magnitudes. Squaring enormous finite q/k can overflow the FP32
norm just as in the pinned naive; large v/state accumulation and widely separated
log-gate magnitudes also require numerical headroom. Qualification is for the
explicit generated domain below, not every finite FP32 bit pattern. No positive
exponential reanchoring, FP16/BF16 internal cast or host-side normalization is used.

The frozen generated domain uses seeded q/k/v normal draws scaled by0.05,
FP32 and BF16-valued inputs, g scales0/0.03/1/30, g_atk scales0/0.03/30,
beta and beta_atk endpoints0/1, and zero/near-zero normalized-key cases.
FP32 o/S/A and stage budgets are relative L2≤1e-4 plus atol2e-5/rtol2e-4.
BF16 output quality is reported against FP32 reference with relative L25e-3;
the state budget remains FP32. No tolerance is enlarged after observing a failure.

## Decomposition, storage and synchronization

Five stages copy GDN mechanics at repository revision20b5247 into this unit.
Their inherited module comment names the earlier GDN-2 ancestor at8786683;
the immediate source is `gdn_chunk_fwd/kernels/stages.py` at20b5247.
There are no imports or edits of GDN/GDN-2 implementations at run time.

| Kernel | Reuse/change | Outputs and work ownership |
| --- | --- | --- |
| pgdn_chunk_atk | New independent launch: naive normalization and ATK recurrence | q_norm/k_read/k_write `[B,T,H,128]`, A `[B,H,128]`; one key-head worker owns sequential tokens/chunks |
| pgdn_chunk_prepare | GDN prefix/scale/beta preparation; direct grouped reads of three distinct H-head arrays | qn,kw,gc,bk,wv `[B,N,HV,64,128]` |
| pgdn_chunk_scores | GDN causal dot mechanics; column key is k_write, row correction key is beta*k_read | lower/score `[B,N,HV,64,64]` |
| pgdn_chunk_wy | GDN guarded row solve retained; full-chunk bounds made constant | U/W `[B,N,HV,64,128]`; W RHS still uses k_read |
| pgdn_chunk_scan | GDN sequential chunk scan; state writes use k_write | incoming states, delta, S_final per HV |
| pgdn_chunk_output | GDN updated-state output equation; scores already contain k_write | o `[B,T,HV,128]` |

ATK is a separate launch because it has one recurrent state per H, whereas main
work is per HV. A separate stage computes that state once and lets grouped
preparation read it without host replication or cross-worker synchronization.
The five GDN stages remain protected by local copies and unique PGDN operator names.

UB footprints are about100.53/196/160/176/224/208 KiB respectively, within the
selected A5 profile's256 KiB. One tile slot is used; no double-buffer claim.
Explicit GM DMA row gaps account for the skipped head axis. `auto_sync` owns
DMA/VF reuse edges. Norm reductions write one scalar, then a STORE→LOAD VF
barrier precedes broadcast; existing WY/state/output landing barriers are retained.
The first vendor build emitted `-Wcce-compat`: VF induction variables were
uint16 while dynamic count/i+1 condition operands had other widths. The task ABI
has exactly64 rows per chunk. The PGDN-local repair uses constant64 loop bounds
and explicit j<=i guards, preserving the order of every active product/sum.
Obsolete dynamic count VF parameters are removed; public kernel ABI, buffers and
barriers are unchanged. The five-stage candidate rebuilt with zero such warnings;
all17 outputs matched the original source bitwise on the same bounded pipesim
case. The original GDN source was not changed. Other build-tool messages concern
CANN header deprecation, a template regex escape and an unused CMake option; they
are retained separately from kernel correctness checks.

All GM edges have one producer; public execution retires temporary buffers only
after their last consuming launch. Validation poisons outputs and checks all17
named public/intermediate arrays as well as independent leaf executions.

## Validation records

Source selection: library90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5,
kernelsb3b3f9c16df7c4626ed3c081032a1be5a753d0b1, Ascriptor0.1.0.
Host: Python3.11.15, torch2.10.0+cpu. Source receipt and detailed results are
reported in the unit's validation artifact; source emission, functional model,
pipe model, vendor compilation and native execution are separate stages.

The contract grid contains30 independent inputs × block_dim1/2: ratios1/2/4/8 ×
chunk counts1/2/3 and T4096; a B2/H3/HV12 case; H8 and H14 model-sized workloads;
normalization and gate endpoints. Reference A/B comparison also records the
H8,K128,T4096 shape associated with `precond_gated_deltanet_340M`. This is a
shape-based equivalence test, not a checkpoint or logits test.

All60 canonical reference cases and104 PGDN host tests passed. The full host
selection reported384passed/4skipped; five KDA NPU modules and the shared
hardware-compilation fixture were explicitly excluded, because this host exposes
unassigned devices. All CPU tests, including the host-only KDA decode checks,
were retained. No local NPU work was submitted. All6 static checks reported
0errors/0warnings. Complete
native hardware acceptance and measurements are pending; CPU or pipe-model
results must not be described as device execution. Board/aclnn launchers remain
untested unless those exact launchers receive their own execution receipts.

The independent CPU chunk solve against pinned recurrent A produced the following
FP32 relative L2 values (fresh seeded inputs, K=V128). B is the independent chunk
solve in this table, so its self-comparison is not a separate kernel execution.

| Shape B/T/H/HV | o | S_final | A_final |
| --- | ---: | ---: | ---: |
| 1/4096/8/8 (340M shape) | 5.1878339e-7 | 6.0343444e-7 | 2.8956342e-7 |
| Maximum over30 independent inputs | 1.8919064e-6 | 1.1988028e-6 | 3.1901925e-7 |

The maximum entries are from different cases; the machine-readable evidence
preserves each shape, seed and output metric. The table does not describe
hardware execution or checkpoint accuracy. The four host-test skips are the
pre-existing full-FLA cache-adapter tests: the full `fla` package is not installed
in the host environment. PGDN oracle A loads its pinned standalone source file
and was actually executed on all30 input sets.


Final source model checks passed on these bounded cases (FP32 internals):

| Stage | B/T/H/HV | block_dim |
| --- | --- | --- |
| pipesim | 1/64/2/2 | 1 |
| pipesim | 1/64/2/16 | 2 |
| pipesim | 1/128/2/4 | 1 |
| pipesim | 1/192/2/8 | 2 |
| sim | 1/64/2/2 | 1 |
| sim, zero q/k | 1/64/1/2 | 2 |
| sim, q/k norm 1e-6 | 1/64/1/2 | 1 |

Each canonical check validates all 17 independently fed leaf outputs and the
three outputs of the actual composition. All pipe-model execution records have
empty hazard/event-balance lists and no deadlock. An additional B2/T192/H3
ATK-only check covers repeated key-head ownership at both block dimensions;
it does not certify the full main-state composition for that batch shape.

For actual returned pipe-model outputs at B1/T64/H=HV2/K=V128, the separate
A/B comparison measured the following FP32 relative L2 at both block dimensions:

| Reference | o | S_final | A_final |
| --- | ---: | ---: | ---: |
| A: pinned CPU recurrence | 3.7924299e-7 | 2.2037465e-7 | 5.1849026e-8 |
| B: independent CPU chunk solve | 4.9374551e-7 | 4.3817887e-7 | 1.8765023e-7 |

All 17 actual arrays were bitwise equal between block_dim1/2 for this case.
BF16 output storage quality against A was 0.0016554152 (against B: 0.0016554177);
this rounds the actual model output on CPU and is not a BF16 device run.

Both final six-kernel builds completed with CANN 9.2.0 in Python3.12.13 /
torch2.12.0+cpu / Ascriptor0.1.0. The validation artifact records generated CCE,
operator-library and device-object hashes. Complete compiler logs were inspected:
no `-Wcce-compat` remains. Compilation does not establish native execution.
