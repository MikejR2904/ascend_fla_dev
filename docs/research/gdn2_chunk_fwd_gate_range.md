# GDN-2 chunk forward

Status: FP32 implementation, CPU references, reduced diagnostics and the full
T=4096/H=16/block_dim=8 native check pass. Remaining hardware scopes and
performance qualification are pending.
Task: GD2-01 (#62), originating from #61. Only the standalone forward unit and
opt-in sibling API are in scope. Existing model/operator files and PM-owned
board/matrix files remain unchanged in this task branch.

## Semantic contract

Source: FLA `fla/ops/gdn2/{naive,chunk_fwd,chunk_intra,wy_fast}.py`, revision
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`. The independent recurrence is

    D_t = exp(g_t) * S_(t-1)
    r_t = w_t * v_t - (b_t * k_t)^T D_t
    S_t = D_t + k_t r_t^T
    o_t = (scale * q_t)^T S_t

Inputs are token-major; state is K-major FP32. q/k normalization uses FP32,
eps=1e-6 and scale=128^-0.5 at the CCE boundary. Normalized keys and queries
are materialized in FP32. BF16 public inputs are widened before this boundary.
Finite g<=0, b in [0,2], w in [0,1]; no clipping. Public scope: B=1,
H=1 or 16, K=V=128, 1<=T<=4096, continuous storage, no autograd.

## Decomposition

Within each 64-token chunk, let G_i=sum_(t<=i) g_t and E_ij=exp(G_i-G_j).

    L_ij = sum_d (b_i*k_i)_d * k_jd * E_ijd, j<i
    Q_ij = sum_d (scale*q_i)_d * k_jd * E_ijd, j<=i
    (I+L) U = w*v
    (I+L) W = (b*k)*exp(G)
    Vnew = U-W*S_in
    S_out = exp(G_last)*S_in + (k*exp(G_last-G))^T Vnew
    O = (scale*q*exp(G))*S_in + Q*Vnew

Five launches: prepare normalized/padded chunk inputs and G; causal scores;
exact forward substitution for U/W; ordered state scan per head; independent
chunk outputs. Every workspace edge has a unique producer. Chunk buffers are
[B,C,H,64,128], score buffers [B,C,H,64,64], carried states [B,C,H,128,128].
C=ceil(T/64). Padding has zero q/k/b/wv and repeats the final G; it is an
identity update. Workspaces are initialized completely by their producer.
No backward caches survive the public call; no input/output aliasing is allowed.

FP32 Vector arithmetic is the initial correctness baseline. The score phase
uses causal differences directly, avoiding positive exponents even with the
observed span 1461. This also avoids multiplying a masked infinity by zero.
Forward substitution is exact triangular solving, not a truncated inverse.
FLA's `safe_gate=True` selects an anchored matrix path whose range assumptions
must not be inferred from its name. Later Cube work may use 16-token diagonal
blocks with direct differences and boundary-anchored off-diagonal blocks.
Any conversion or changed reduction tree must pass the original error gate.

At the pinned FLA revision, `chunk_fwd.py` computes chunk-local base-2 log
prefixes: fused activation uses `kda_gate_chunk_cumsum`, while already activated
gates use `chunk_local_cumsum`, both scaled by `1/ln(2)`. Its safe intra path
uses 16-token blocks with midpoint factors `exp2(G-Gmid)` and
`exp2(-(G-Gmid))`; the token-parallel path evaluates causal differences directly.
`wy_fast.py` materializes the gated keys/query and end-of-chunk scaled key.
This implementation receives already activated natural-log gates, keeps local
prefixes in FP32 and evaluates causal differences without the opposing
midpoint exponentials. Shared chunk-local representation does not transfer a
Triton precision or range qualification to the CCE backend.

## Ownership and synchronization

Prepare, score, WY and output assign complete (batch, chunk, head) work items to
vector participants. Scan assigns complete heads and visits chunks in order.
A launch boundary publishes all GM edges. The baseline uses one UB slot per
buffer and `auto_sync`; VF store/load barriers publish local scalar reductions
and solved rows. No mixed Cube/Vector pipeline or multi-buffer overlap is claimed.
Physical rows are 128 FP32 elements (512 bytes); score rows are 64 (256 bytes).
The scan working set must stay below the A5 256 KiB UB profile capacity.

## Validation and remaining gates

Compare the CPU chunk composition and CCE output/state against an independent
FP32 recurrence, including nonzero state, odd chunk counts, tails, strong decay,
and split-prefill/decode. Output and state relative L2 must be <=1e-4; report
maximum absolute error too. Real-model budgets remain 1e-3 FP32 and 1e-2 BF16.

The user requires CPU Torch for golden references and permits Torch NPU for
performance comparisons. CPU-tensor aclnn/board harness execution is supported
separately from the in-process NPU bridge.
CPU timings are not NPU performance. SSH authentication initially failed, so reduced simulator checks were used only
to diagnose the newly authored stages. The assigned directory and device health
are now confirmed. The full T=4096/H=16 canonical board workload passes.
All 12 contract cases also pass the native in-process NPU bridge at block_dim=1
and 8, with every one of the 13 stage outputs byte-identical between them.
This does not qualify block_dim=2/4 or every canonical board-launcher case.

## Host observations

Five CCE stages emit successfully at the accepted library pin. After the native
WY workaround, functional sim passes T=1/H=1 and pipesim passes T=2/H=1 and
T=65/H=16 at block_dim=1. The latter renews the combined multihead/tail diagnostic
on the changed source. CPU tests cover both zero and strong decay through T=4096/H=16;
FLA recurrent is loaded explicitly by file path in the optional oracle test.

The public graph retires each workspace after its last consumer launch. The
standalone stage checker intentionally retains intermediates for comparison.
This reduces retained allocations, without changing arithmetic or claiming a
measured device speedup. No tensor identities or private machine paths are
stored in the tracked contract.

## GD2-01 gate-range survey status

### GD2-01 baseline native timing and state handoff

On 950PR_9589 V100, CANN compiler/OPP 9.2.0, Torch 2.12.0+cpu with
torch_npu 2.12.0 explicitly loaded for execution, both timing participants use
NPU tensors. Goldens run only on CPU. FP32, H=16, block_dim=8, HF32 disabled;
each of three rounds measures baseline-before, candidate, baseline-after with
10 warmups and 50 synchronized wall-time samples per phase. Compilation and
input transfer are outside the measured calls; the public candidate includes
validation and workspace allocation. Baseline is the repository's composed
Torch NPU per-token recurrence, not an optimized FLA Triton kernel.

| T | Candidate median range (ms) | Speedup versus faster flanking baseline | Peak incremental Torch allocation |
| --- | --- | --- | --- |
| 1024 | 4.261–4.371 | 113.08–117.51x | 72,351,744 bytes |
| 4096 | 15.932–16.021 | 218.71–224.53x | 286,261,248 bytes |

Both participants pass the CPU golden before timing. Raw sample receipt SHA-256:
`06a9f5983fb25eb14d747c73ef350f95a8143e3499d9c1b9bfe69ca64645ebeb`.
These preliminary measurements do not claim GD2-02 completion or whole-model speedup.
They belong to the accepted pre-optimization source
`82dd3fec13edee5c358a336721bd8d45ef7238fb1a62a05eca668b5fbae47e66`.

Native prefill (T=65 or 4096, FP32 or BF16 q/k/v) followed by one token of the
existing recurrent decode also passes CPU golden checks. State is handed off
directly in FP32. Since the owner decode API requires q/k/v/b/w to share one
dtype, decode widens the already-quantized q/k/v to FP32 and casts its output
back to the input dtype for comparison. Maximum output relative L2 across the
four cases is 5.8482e-5, maximum state relative L2 is 1.1214e-7; every case passes
both allclose(atol=rtol=1e-4) and relative L2<=1e-4. This is synthetic operator
handoff evidence, not real-model logit/cache qualification.

### Real-checkpoint CPU gate survey

CPU Torch 2.12.0+cpu, Python 3.12.13, four threads; B=1/T=4096/H=16/K=V=128.
The model uses BF16 canonical projections, with activated core tensors converted
to FP32 for both mathematical oracles. Read-only repository model/reference code
is from `e110b6ef27b955ecaf26be06b06a302b87142ad2`. Checkpoint SHA-256 is
`4ac729c627febc431bf4b2011a9cb7ad9c30cc1d017ff74aeca15f76efb2df6d`; both transfer endpoints verified it.
Tokenizer: TinyLlama v1.1 revision `ff3c701f2424c7625fdefb9dd470f45ef18b02d6`,
vocab 32000. Random IDs use a dedicated CPU generator, seed 20260914. Natural
text repeats "The capital of France is Paris. Linear attention processes a sequence by updating a recurrent state." to 4096 tokens.

Both sample types cover all 18 layers and five sizes: 180 rows, all finite and
passing both CPU oracles at relative L2<=1e-4. Each table value is the maximum
across layers, so different columns can reach their maxima in different layers.

**legal-random-token-ids**

| Size | Max block decay | Repo o L2 | Repo state L2 | FLA o L2 | FLA state L2 |
| --- | --- | --- | --- | --- | --- |
| 4 | 169.671 | 1.7800e-6 | 4.4876e-6 | 1.7736e-6 | 4.4859e-6 |
| 8 | 235.316 | 1.7071e-6 | 4.4579e-6 | 1.7008e-6 | 4.4565e-6 |
| 16 | 440.371 | 1.7190e-6 | 4.4430e-6 | 1.7130e-6 | 4.4411e-6 |
| 32 | 772.990 | 1.7170e-6 | 4.4382e-6 | 1.7103e-6 | 4.4367e-6 |
| 64 | 1520.915 | 1.7111e-6 | 4.4467e-6 | 1.7051e-6 | 4.4454e-6 |

| Size | Repo o max abs | Repo state max abs | FLA o max abs | FLA state max abs |
| --- | --- | --- | --- | --- |
| 4 | 4.7684e-7 | 2.3603e-5 | 5.0664e-7 | 2.3603e-5 |
| 8 | 4.7684e-7 | 2.3365e-5 | 4.7684e-7 | 2.3127e-5 |
| 16 | 4.4703e-7 | 2.2888e-5 | 4.3958e-7 | 2.2888e-5 |
| 32 | 4.3213e-7 | 2.2411e-5 | 4.1723e-7 | 2.2411e-5 |
| 64 | 4.4703e-7 | 2.3127e-5 | 4.4703e-7 | 2.3127e-5 |

Max per-token decay: 61.463203; max full-sequence decay:
83964.062500.

**natural-text-repeated**

| Size | Max block decay | Repo o L2 | Repo state L2 | FLA o L2 | FLA state L2 |
| --- | --- | --- | --- | --- | --- |
| 4 | 138.732 | 2.5364e-6 | 3.5710e-6 | 2.5356e-6 | 3.5912e-6 |
| 8 | 247.264 | 2.0182e-6 | 2.9143e-6 | 2.0167e-6 | 2.9355e-6 |
| 16 | 408.172 | 2.5417e-6 | 3.1563e-6 | 2.5403e-6 | 3.1775e-6 |
| 32 | 776.791 | 2.6020e-6 | 3.1699e-6 | 2.6005e-6 | 3.1900e-6 |
| 64 | 1423.758 | 2.6005e-6 | 3.0829e-6 | 2.5994e-6 | 3.1041e-6 |

| Size | Repo o max abs | Repo state max abs | FLA o max abs | FLA state max abs |
| --- | --- | --- | --- | --- |
| 4 | 1.0878e-6 | 4.4346e-5 | 1.0878e-6 | 4.4823e-5 |
| 8 | 8.6427e-7 | 3.7432e-5 | 8.6427e-7 | 3.7909e-5 |
| 16 | 1.0729e-6 | 3.9101e-5 | 1.0431e-6 | 4.0054e-5 |
| 32 | 1.1474e-6 | 3.8624e-5 | 1.1325e-6 | 3.9577e-5 |
| 64 | 1.1176e-6 | 3.7670e-5 | 1.0878e-6 | 3.8624e-5 |

Max per-token decay: 66.358490; max full-sequence decay:
88852.812500.

The historical numerical value 1461.214 is **not reproduced exactly**: the
random replay observes 1520.914551, a delta of +59.700551. It reproduces the
large-span counterexample with a larger observed span; historical raw inputs
are unavailable to attribute the difference. No exact numerical identity to
that earlier run is claimed.

Select 64-token chunks with FP32 logarithmic prefixes and direct nonpositive
causal differences: all five sizes pass, and smaller resets show no acceptance
advantage on these samples. This qualifies the measured sample domain, not all
possible weights or inputs. Native whole-model logit/cache qualification remains
separate. Raw 180-row receipt SHA-256:
`2be57fcfed201a5a8f58e0b83af398d6d63bfb05324e1d26f84c5574ccebd5f6`.
Reproduce using the checkpoint command described below; raw receipts remain in
ignored task output rather than recorded golden tensors.

All exponential sites of the selected forward chain:

| Site | Exponent | Sign for g<=0 | Failure avoided |
| --- | --- | --- | --- |
| causal score Q/L | G_i-G_j, j<=i | nonpositive | no noncausal exp overflow before masking |
| WY right-hand side | G_i | nonpositive | no inverse cumulative decay |
| state carry | G_last | nonpositive | valid decay to zero |
| state update key | G_last-G_i | nonpositive | no reciprocal decay |
| output carry | G_i | nonpositive | no positive rescaling |

The representation is a per-chunk logarithmic prefix plus directly evaluated
causal pair differences. It is not the KDA endpoint/midpoint product of a
large positive and negative exponential. No division by exp(G) occurs.
Consequently exp(1461/2) is never evaluated. Precision, independently of
finiteness, must still be measured for prefix subtraction and triangular
solves; no universal gate-domain qualification follows from the sign proof.
The chosen baseline has 64-token chunks, with no anchored 16-token fast path.
The completed survey compares reset intervals {4,8,16,32,64}. A future blocked
Cube implementation must renew these numerical checks on its own arithmetic.

## Synthetic five-size observation (not checkpoint evidence)

CPU Torch 2.14.0, seed 20260914, B=1/T=4096/H=16/K=V=128, FP32;
each 64-token gate block is calibrated to cumulative decay 1461.214.
All outputs and states are finite. No performance timing is inferred.

| Prefix reset size | Maximum block decay | Output relative L2 | State relative L2 | Output max abs | State max abs |
| --- | --- | --- | --- | --- | --- |
| 4 | 207.451 | 2.304e-7 | 2.522e-7 | 5.122e-8 | 1.013e-6 |
| 8 | 354.658 | 4.076e-7 | 8.151e-7 | 1.378e-7 | 5.156e-6 |
| 16 | 604.186 | 7.970e-7 | 1.430e-6 | 2.086e-7 | 4.858e-6 |
| 32 | 970.827 | 1.620e-6 | 3.413e-6 | 5.178e-7 | 1.550e-5 |
| 64 | 1461.214 | 3.284e-6 | 5.908e-6 | 1.068e-6 | 1.562e-5 |

The same generated inputs also pass the pinned FLA naive CPU oracle:

| Prefix reset size | FLA output relative L2 | FLA state relative L2 |
| --- | --- | --- |
| 4 | 2.247e-7 | 2.522e-7 |
| 8 | 4.044e-7 | 8.151e-7 |
| 16 | 7.954e-7 | 1.430e-6 |
| 32 | 1.619e-6 | 3.413e-6 |
| 64 | 3.284e-6 | 5.908e-6 |

This supports continuing with the 64-token FP32 baseline on synthetic inputs.
It does not establish the real-checkpoint gate-domain limit or satisfy the
required random-token and natural-text replay. Reproduce with:

```sh
python kernels/projects/a5/gdn2_chunk_fwd/ref/survey.py --synthetic --output tmp/GD2-01/synthetic-survey.json
```

For actual model evidence, use the same script with `--checkpoint` and
`--tokenizer` pointing to local assets. It verifies the recorded checkpoint
SHA-256, uses CPU Torch only, and reports each layer and both sample types.
Checkpoint mode additionally requires `--fla-naive` (or `FLA_GDN2_NAIVE`)
pointing to the pinned `naive.py`. Its source digest is verified before import;
every reset size reports errors against both the repository and FLA recurrences.

## Native WY dependent-loop diagnosis

The first full native run of revision `9ba48cd`, B=1/T=4096/H=16,
FP32/block_dim=8, failed at stage `u` (max absolute error 0.05106529966).
All five stages compiled and executed; successful native exits did not establish
numerical correctness. Environment: Ascend 950PR_9589 V100, CANN compiler and OPP
9.2.0, Python 3.12.13, Torch 2.12.0+cpu. The OPP inventory contains ascend950,
ascend910b and ascend910_93. These observations do not apply to other devices.

Re-solving WY from its actual input blobs gave max absolute error 0.05105925724
and relative L2 0.009164325893. A diagnostic recurrence dropping the immediately
preceding row matched native output within 5.960464478e-8, locating the missing
contribution without accepting that altered recurrence as the reference.

A B=1/T=4/H=1/block_dim=1 control uses a generated strictly lower matrix with
only its first subdiagonal set to 0.25. Native artifact controls:

| WY VF form | U max absolute error | W max absolute error |
| --- | --- | --- |
| original dependent inner bound | 0.2538757324 | 0.25 |
| native float load/store overloads | 0.2538757324 | 0.25 |
| VV_ALL barriers | 0.2538757324 | 0.25 |
| fixed 64 iterations, explicit `j < i` guard | 0 | 0 |

The scoped source workaround uses the last form, preserving all terms and their
order. This establishes a failing native dependent-bound form, not whether the
vendor compiler or hardware implements it incorrectly. Artifact controls alone
are not source-kernel qualification. The source change passes 44 targeted host
tests, T=2/H=1 pipesim at block_dim=1, and static checking with zero errors and
warnings; the full host suite passes 223 tests with 5 skips. Full native source
regression passes B=1/T=4096/H=16/K=V=128/FP32/block_dim=8, including every
stage and a separate composition run against the CPU recurrent oracle:

| Output | Relative L2 | Maximum absolute error |
| --- | --- | --- |
| o | 2.400727462e-6 | 1.345761120e-7 |
| final_state | 4.140812156e-6 | 4.950910807e-6 |

Redacted failure, control and full-workload receipts are retained in ignored
`tmp/GD2-02/`. No timing conclusion follows from the CPU/native harness.
The tokenizer must be the recorded local revision; vocabulary size alone is
not proof of tokenizer provenance. The natural-text sample is repeated to
4096 tokens, and that construction is identified in the report.

The scoped task branch passes 223 host tests (5 device-dependent skips),
including 44 chunk tests with the supplied FLA oracle. All five entry checks
report zero errors and warnings. The combined T=65/H=16/block_dim=1 pipesim
case compares every intermediate and both final outputs, with no hazard or
deadlock reported. This covers head reuse together with a cross-chunk tail.

## Vector preweight candidate (GD2-03, 690f420)

This subsection records the first candidate at revision `690f420`. Run its
reproduction command from that revision to reproduce this intermediate result;
the subsequent WY candidate and current-source measurements are below.

The accepted `0397aa1` source was profiled before editing. One candidate then
changed the scan and output preweights: compute `k_i * exp(G_last-G_i)` and
`q_i * exp(G_i)` as complete FP32 channel rows, and read their scalar weights
inside the unchanged ordered sums. Raw keys/queries have no later local reader;
their private UB allocations hold the weighted values. GM inputs, outputs,
launches, ownership, precision and public signatures are unchanged.

At C=64/K=128, each preweight previously issued 8192 scalar-broadcast register
exponentials per chunk. It now issues 128 register exponentials with distinct
channels in the lanes. Scan's 128 state-decay exponentials remain unchanged.
These are source-level instruction counts, not measured cycles. Scan/output UB
allocations remain 224/208 KiB with one slot; each gains a 32 KiB local read and
write. VF STORE-to-LOAD barriers publish preweights before scalar consumption;
auto_sync retains the DMA/VF and loop-carried reuse edges. No mixed pipeline or
new lookahead is introduced. Actual HBM traffic was not measured.

Discovery measurements below use synchronized NPU wall time, 10 warmups and
50 samples, FP32 B=1/H=16/K=V=128/block_dim=8. Hardware is 950PR_9589 V100;
CANN compiler/OPP 9.2.0, with ascend950/ascend910b/ascend910_93 OPP directories;
Torch 2.12.0+cpu plus torch_npu 2.12.0. Goldens run on CPU. Each stage retains
preallocated inputs/outputs; public-call timing includes validation and allocation.
Isolated stage times include their own bridge/synchronization overhead and are
not additive device times. Unchanged-stage variation is not an optimization claim.

| Stage | T1024 baseline us | T1024 candidate us | T4096 baseline us | T4096 candidate us |
| --- | --- | --- | --- | --- |
| prepare | 997.347 | 574.751 | 2763.676 | 2850.962 |
| scores | 1034.809 | 956.495 | 3341.520 | 3220.174 |
| WY | 1496.690 | 1379.372 | 5261.228 | 5111.855 |
| scan | 1004.807 | 659.031 | 3185.163 | 2203.290 |
| output | 736.008 | 463.639 | 2194.543 | 1478.769 |
| public call | 4492.556 | 3839.584 | 16029.428 | 14685.062 |

The candidate source SHA-256 is
`a1f305eb5209420e026a1a6c82d7147f00a00c0588ea7f7f331805993d1cce5b`.
All 12 contract cases pass all 13 stage comparisons against CPU goldens at
block_dim=1 and 8. Every stage tensor is byte-identical across those block
counts and versus the accepted baseline. Static checks for all five entries
have zero errors/warnings; scan/output contain 153/122 surface operations.
Fresh functional sim T=1/H=1 and pipesim T=2/H=1 plus T=65/H=16 pass at bd=1.
The host suite remains 223 passed, 5 skipped. No precision budget is changed.
The renewed canonical board check also passes T=4096/H=16/FP32/bd=8,
including all intermediates and separate composition. Composition output/state
relative L2 are 2.400727462e-6/4.140812156e-6; maximum absolute errors are
1.345761120e-7/4.950910807e-6. Across all 12 native cases, final output/state
relative L2 maxima are 3.348794618e-6/5.677315394e-6.

`benchmark.py` compares the accepted source with the candidate through the same
public launch graph in one NPU process. Only the two changed baseline entries
are renamed to avoid CANN operator-type collisions; the identical first three
stages share artifacts. Compilation and input transfers precede timing. Both
variants pass CPU goldens first; each of three rounds measures baseline,
candidate, then baseline with 10 warmups and 50 synchronized samples per phase.
Baseline source digest and unchanged-stage identity are checked before execution.
The fixed precompiled operator selector replaces `_compiled` equally for both
variants; the public validation, allocations and five launches remain timed.

The checked-in script produces the following medians on the environment above.
Reduction uses the faster baseline of the same round, retaining both controls:

| T | Round | Baseline before us | Candidate us | Baseline after us | Latency reduction |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1 | 4239.288 | 3882.246 | 4366.985 | 8.422% |
| 1024 | 2 | 4240.702 | 3876.824 | 4240.886 | 8.581% |
| 1024 | 3 | 4236.278 | 3879.435 | 4248.421 | 8.423% |
| 4096 | 1 | 15726.550 | 14199.110 | 15759.691 | 9.712% |
| 4096 | 2 | 15696.767 | 14308.423 | 15802.565 | 8.845% |
| 4096 | 3 | 15782.460 | 14178.686 | 15846.198 | 10.162% |

Both final tensors are byte-identical between variants at both lengths. Peak
Torch allocated-byte deltas are identical: 72,351,744 at T1024 and 286,261,248
at T4096. These allocator measurements do not measure all native runtime memory.
The preceding temporary-driver sandwich also passed all three rounds at each
length: T1024 reduction 7.788..8.986%, T4096 8.725..8.814%. It is retained along
with the checked-in-runner replay; neither run is discarded in favor of the other.

Raw receipts are retained in ignored `tmp/gdn2-opt/`; current identities are:

| Receipt | SHA-256 |
| --- | --- |
| `repro-sandwich.json` (checked-in runner) | `b2d4a6f88f1ea499b66ec52bf1a6ef100dc5be3e76ba048da10f530f4fd75fce` |
| `sandwich.json` (initial same-process driver) | `7805ba7cde3d79e3d5d9c6092ebdc3b5ca56553dba7aade1efd9c06f0b152253` |
| `preweight-grid/bd1.json` | `f5abfee687b618a72a901325fad51ae4c8eb236262a1bde211bbfd4fd887dc44` |
| `preweight-grid/bd8.json` | `28ec977be0c10ec475b16291668450414549dd4c31044ad3fc2992b9e235f7ad` |

The accepted baseline source digest is
`82dd3fec13edee5c358a336721bd8d45ef7238fb1a62a05eca668b5fbae47e66`.
Its benchmark-only entry renaming yields
`da38f63331c2b59743023dc061eb4730642bdda8fa3629fc204c7b80fa3bfc37`.
Candidate scan/output artifact signatures are `10b7bac21c287097` and
`be946134e965d78a`; renamed baseline signatures are `064926fb0099bc1b`
and `3b51409c544f3a7a`. Full reports retain all 50 samples for every phase.

```sh
mkdir -p tmp/GD2-03
git show 0397aa1:kernels/projects/a5/gdn2_chunk_fwd/kernels/stages.py > tmp/GD2-03/stages-baseline.py
python kernels/projects/a5/gdn2_chunk_fwd/benchmark.py --baseline tmp/GD2-03/stages-baseline.py --output tmp/GD2-03/sandwich.json
```

Use the assigned device environment and hardware lock. Timing compares native
chunk implementations; it is separate from the earlier Torch NPU recurrence
comparison. WY remains the largest stage. This round tests one candidate and
makes no Cube, reduced-precision or whole-model performance claim.

## WY row-pair reuse (GD2-04)

The next candidate starts from the qualified preweight source `690f420` and
changes only WY. A fresh full-workload profile was completed before editing.
Two adjacent solve rows retain independent FP32 accumulators but share loads
of each previously solved U/W row. After the ordered `j<i` contributions, the
second row consumes completed row i directly from registers. Each row keeps
the same sequence of FP32 products and subtractions; the fixed C64 inner loop
and explicit guard retain the native dependent-bound workaround.

For a full C64/D128 tile, previous U/W row loads fall from 2,064,384 to
1,015,808 requested UB bytes, a saving of 1 MiB. Coefficient loads and ordered
arithmetic contributions are unchanged. Row-publication STORE-to-LOAD barriers
fall from 64 to 32, plus the unchanged initialization barrier. WY UB remains
176 KiB, one slot per allocation. The emitted VF declares 18 FP32 registers
and separate `vmul`/`vsub` instructions; physical register allocation and HBM
traffic were not measured. GM layout, workspace, core ownership and launches
are unchanged. This is intra-VF reuse, with no mixed pipeline or new lookahead.

The loop visits even rows, so its partner always fits in the C64 allocation.
For odd valid counts, the partner is a zero-padded row: b/v/lower were already
initialized to zero. Both rows are published before the next pair reads them;
existing auto_sync still orders the enclosing DMA and repeated-tile UB reuse.
All padded stage values remain observable to the verification harness.

Discovery measurements use the same 950PR_9589 V100, CANN compiler/OPP 9.2.0,
Torch CPU goldens and synchronized Torch NPU timing described above, with
FP32 B1/H16/K=V128/bd8, warmup10/repeat50. Every stage and final composition
passes CPU goldens before timing. Isolated stage times are not additive.

| Stage | T1024 preweight us | T1024 row-pair us | T4096 preweight us | T4096 row-pair us |
| --- | --- | --- | --- | --- |
| prepare | 854.920 | 976.199 | 2936.543 | 2659.515 |
| scores | 893.042 | 895.911 | 3211.199 | 3210.159 |
| WY | 1361.942 | 799.822 | 5094.977 | 2837.223 |
| scan | 641.859 | 647.291 | 2172.654 | 2165.914 |
| output | 459.198 | 463.292 | 1488.119 | 1491.655 |
| public call | 3862.470 | 3314.700 | 14657.458 | 11968.898 |

Unchanged-stage variation is retained and is not attributed to WY reuse.
The current scores stage is the largest isolated stage in this profile.

The row-pair kernel source SHA-256 is
`803c45093dd21b7d5af196ae76e1df207fde7d7261e3fa574934b8cb90519bb4`.
The current benchmark accepts the identified `0397aa1` and `690f420` sources.
It compares complete VF/kernel source blocks, shares identical stage artifacts
and gives each changed baseline kernel a distinct operator name. The incremental
comparison changes only WY; the cumulative comparison changes WY/scan/output.
The benchmark checks final tensor bytes, including signed-zero bits, before
timing. All precompile work and input transfers remain outside timed samples.

All 12 contract cases at bd1/bd8 pass all 13 CPU-golden stage comparisons.
Every stage tensor is byte-identical across those block counts and versus
`690f420`, including zero padding in odd tails. Output/state relative L2 maxima
remain 3.348794618e-6/5.677315394e-6. Static checks report zero errors/warnings
for all five entries (WY: 199 surface operations). Fresh sim T1/H1 and pipesim
T2/H1 plus T65/H16 at bd1 pass; the host suite passes 223 tests with 5 skips.
The fresh canonical board source check also passes T4096/H16/FP32/bd8:
all 13 stage comparisons and separate composition pass. Composition output/state
relative L2 are 2.400727462e-6/4.140812156e-6; maximum absolute errors are
1.345761120e-7/4.950910807e-6. The redacted source-bound board receipt SHA-256 is
`b9360522b23b69405ff804a0faacb9bfb6af9bbf3d4b1abe3f34008d1e8e6fa0`.

Incremental same-process sandwich against `690f420`, with three rounds,
warmup10/repeat50 and the same FP32 shape/device/software above:

| T | Round | Preweight before us | Row-pair us | Preweight after us | Latency reduction |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1 | 3866.532 | 3299.593 | 3867.014 | 14.663% |
| 1024 | 2 | 3864.672 | 3297.731 | 3862.171 | 14.615% |
| 1024 | 3 | 3859.590 | 3299.446 | 3860.119 | 14.513% |
| 4096 | 1 | 14473.173 | 12108.251 | 14389.744 | 15.855% |
| 4096 | 2 | 14401.214 | 12116.276 | 14388.028 | 15.789% |
| 4096 | 3 | 14373.337 | 12201.671 | 14535.752 | 15.109% |

Each reduction uses the faster same-round baseline. Both final tensors have
identical bytes before timing. Peak Torch allocated-byte deltas match the
preweight candidate: 72,351,744 at T1024 and 286,261,248 at T4096.
The incremental receipt SHA-256 is
`5cc541d699c84ef2cb86d2baadd5ff369e8da1d1ffacf279299bf399676497ba`.
Candidate WY artifact signature is `5e29506610a2ccf7`; the renamed preweight
WY baseline signature is `baa1ad7bed3e9a45`. The other four artifacts are shared.

An independent cumulative sandwich directly compares the final candidate with
accepted GD2-01 `0397aa1`, under the same shape/software/timing conditions:

| T | Round | GD2-01 before us | Final candidate us | GD2-01 after us | Latency reduction |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1 | 4215.212 | 3294.670 | 4260.013 | 21.839% |
| 1024 | 2 | 4220.965 | 3293.888 | 4225.445 | 21.964% |
| 1024 | 3 | 4234.540 | 3288.779 | 4210.145 | 21.884% |
| 4096 | 1 | 15780.261 | 12022.261 | 15854.296 | 23.815% |
| 4096 | 2 | 15803.773 | 11995.092 | 15786.295 | 24.016% |
| 4096 | 3 | 15850.708 | 12134.457 | 15810.924 | 23.253% |

These are direct measurements, not a sum or product of earlier speedups.
Final output/state bytes and peak Torch allocated-byte deltas match GD2-01
as well. Raw receipts retained in ignored `tmp/gdn2-opt/`:

| Receipt | SHA-256 |
| --- | --- |
| `wy-baseline.json` | `5f51921fcd02d218254e93a2d6ab3820429f7693cde089ecd2fd6faa586cd81d` |
| `wy-pair-v1.json` | `eb3281b3fd630b08d3fce5dac14643bfcc98a20b23b4158dbed1a1a455f55b37` |
| `wy-pair-grid/bd1.json` | `a4ad747902abadde699fe67e12a15f5d19a005ffaf393ded95db10cab5a43874` |
| `wy-pair-grid/bd8.json` | `272923c81eea1292e2bf2192cad0ca675056e26eb6977514a7e36776e90aff3d` |
| `wy-sandwich.json` | `5cc541d699c84ef2cb86d2baadd5ff369e8da1d1ffacf279299bf399676497ba` |
| `combined-sandwich.json` | `595c2c24d85a6d99d8b13772f5f56d4d40b4b33140093b023ab51409cceb4e4d` |

The cumulative baseline's benchmark-only WY/scan/output entry renaming yields
source digest `b6e8a895dbe2e9e5a5858bbc5f97afff1ac41ecfade10b33a3f5b0658623f314`.
Its artifact signatures are `fe0956b2edcc7bb8`, `5971b10c525db587` and
`ae5cdbc1f6a7baaf`; prepare/scores are shared. Reports retain all raw samples.

Reproduce the incremental comparison from the final source checkout, with the
assigned environment and hardware lock:

```sh
mkdir -p tmp/gdn2-wy
git show 690f420:kernels/projects/a5/gdn2_chunk_fwd/kernels/stages.py > tmp/gdn2-wy/stages-preweight.py
python kernels/projects/a5/gdn2_chunk_fwd/benchmark.py --baseline tmp/gdn2-wy/stages-preweight.py --output tmp/gdn2-wy/wy-sandwich.json
```

For the cumulative comparison, export the same path from `0397aa1` instead and
use a separate output receipt. The WY round tested one candidate and retained
every result; no precision or native loop-workaround relaxation was needed.
