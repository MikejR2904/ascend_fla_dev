# GDA-03 grouped GDN backward ABI and range contract

Status: FP32-only delivery per D-PM-35. Full native grid, final-source byte
closeout and refreshed same-device FP32 measurements passed.
This task is standalone backward. The accepted grouped forward and GDN-2 are
unchanged. The authority is FLA pin `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`,
`fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule`.

## Frozen public ABI

`chunk_gdn_bwd(q,k,v,g,beta,do=None,dht=None,...) -> (dq,dk,dv,dg,dbeta)`.

| Item | Supported contract |
|---|---|
| q, k | Contiguous token-major `[B,T,H,128]`, FP32 only |
| v | Contiguous `[B,T,HV,128]`, same dtype as q/k |
| g, beta | FP32 `[B,T,HV]`, finite `g<=0`, `0<=beta<=1` |
| Heads | Positive `H,HV`, `HV%H==0`, consecutive groups |
| B, T | Positive B; T multiple of 64, from 64 through 4096 |
| do | Optional `[B,T,HV,128]`, same storage dtype as output v |
| dht | Optional FP32 `[B,HV,128,128]`; at least one cotangent |
| dq, dk, dv | Input shapes, FP32 |
| dg, dbeta | FP32 `[B,T,HV]` |
| Initial state | Only None (zero); no dh0 output |
| Normalization | None; scale fixed `128**-0.5`, including dq's chain factor |
| Execution | A5 CCE, block_dim 1 or 2, explicit inprocess/aclnn/board launcher |
| Unsupported | Tails/decode, varlen, CP, head-first, transposed state, normalization, other scale/dtypes/devices |

CPU launchers check finite/range preconditions. The in-process NPU path inherits
forward's caller preconditions, with shape/dtype/layout checked before dispatch.
The API computes a first derivative explicitly; it does not wire autograd or
provide higher derivatives. BF16 q/k/v/do explicitly reject before preparation
or dispatch, with an error pointing to BF-02. Native BF16 backward is outside
GDA-03. Runtime preparation is allocation and optional-cotangent zeros on the
input device; no dtype conversion, host recurrence or reference fallback is permitted.
All preparation and checkpoint recomputation are included in public-path timing.

## Mathematical reference and fixed budgets

For each value head (sharing its group's raw key/query):

```
a = exp(g); D = a*Sprev; r = v - k^T D; z = beta*r
S = D + k*z^T; o = scale*q^T S

dq = scale*S*do; G += scale*q*do^T
dz = G^T*k; dr = beta*dz
dk = G*z - D*dr; dv = dr; dbeta = dot(dz,r)
GD = G - k*dr^T; dg = sum(GD*D); Gprev = a*GD
```

Start G from dht (or zero); absent do contributes zero. Sum dq/dk over the
consecutive value-head group. A is CPU FP32 autograd of the exact hash-checked
pinned naive function; B is a separately derived adjoint without FLA imports
or autograd. Both recompute from runtime-generated inputs.

A itself forces FP32, so double-epsilon finite differences cannot qualify it
literally. The FP64 qualification lift replaces only its two `torch.float32`
AST attributes by `torch.float64` in memory, preserving every equation/order.
The pinned file and actual A oracle stay unchanged. B is dtype-preserving and
its analytical backward is independently checked against FP64 scalar-loss finite
differences. Reports distinguish this precision lift from the literal A.

Before kernels, A/B must agree to relative L2 <=1e-5 per gradient. Native FP32
acceptance is <=1e-4 per gradient against **both** A and B. Historical BF16
quality records are excluded from GDA-03 delivery by D-PM-35. Negative controls zero, negated, and
1.25-times gradients must each fail the FP32 budget. No tolerance adaptation.
Executed calibration and native evidence are recorded below.

## Complete exponent inventory

| Place | Expression | Range for finite g<=0 |
|---|---|---|
| Boundary checkpoint forward recurrence | exp(g) | [0,1]; underflow tends to zero |
| Within-chunk primal replay | exp(g) | [0,1]; same operation/order as checkpoint |
| Reconstruct decayed state during adjoint | exp(g) | [0,1] |
| Recurrent adjoint propagation | exp(g) | [0,1] |

There is no gate cumsum, exp(-g), exp(g_last-g), state inversion, or division by
`1-beta*||k||²`. No additional gate-span restriction is introduced. The inherited
finite input domain is not a claim that arbitrary huge finite q/k/v cannot
overflow the recurrence; observed results are scoped to executed workloads.

## Checkpoint ABI and ownership

Three launches: boundary recurrence → replay/adjoint → grouped dq/dk reduction.
The first writes FP32 `[B,N,HV,128,128]` chunk-start states, where N=T/64,
and final state for leaf validation. The second owns one `(B,HV)` at a time,
replays 64 tokens into FP32 `[B,HV,64,128,128]` scratch, then reverses them.
It emits per-value-head dq/dk and final dv/dg/dbeta. The third deterministically
sums dq/dk in ascending group order, with one owner per `(B,T,H)` and no atomics.
There are no checkpoints supplied by or modifications to the forward operator.

Each vector owner has a single UB slot per tensor. State and state adjoint
require 128 KiB together; row/scalar buffers fit in the selected profile's remaining
UB. State rows are 128 FP32 elements, exactly two vector registers. DMA and VF
lifetimes use local autosync; explicit VF STORE→LOAD barriers protect dependent
local reloads. A local MTE3→MTE2 event publishes the replay tape; a local drain
precedes tape reuse. Stream-ordered launches publish inter-stage GM edges.
The tape has one independent slice per value head and costs 32 MiB at B1/HV8,
independent of T. All outputs are fresh, fully written, and inputs are read-only.

## Reference and diagnostic qualification

Before writing device kernels, 58 completed calibration cases passed A/B <=1e-5;
maximum per-gradient relative L2 was dq 2.47549645e-7, dk 0, dv 0,
dg 4.84878352e-7, dbeta 0. Both FP64 checks passed at ratios 1/2. The preserved
[pre-kernel report](../../kernels/projects/a5/gdn_chunk_bwd/evidence/pre-kernel-calibration.json)
contains individual case numbers. The completed extended calibration has 69
cases, including all four ratios at T4096 and all three cotangents, with unchanged
maxima; see [full calibration](../../kernels/projects/a5/gdn_chunk_bwd/evidence/calibration.json).

The first complete native workload B1/T4096/H=HV8/K=V128, both cotangents, bd2,
passed FP32 checks against A/B and all 10 composition/10 independent leaf arrays,
with inputs unchanged. The original [raw native output](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-bd2.log)
and [source manifest](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-source-manifest.json)
are retained. Their BF16 rows describe historical execution and are excluded
from current delivery acceptance. The complete FP32 grid is summarized below.

Initial vendor compilation rejected the internal parameter name `do`, a C++
keyword, before device computation. Renaming the native argument to `dout`
fixed compilation; the public parameter remains `do`. No equation or budget
changed. Full failed compiler logs are retained in the failed-run evidence linked below.

The canonical CPU reference sweep passed all 138 cases. A bounded functional
model case (C1/equal heads/bd1) passed all ten independent leaf arrays and five
composition gradients. The first larger pipesim (C2/grouped/multiple head items)
timed out in the interpreter with an empty blocked-lane summary. Two smaller source diagnostics isolate tape reuse and uneven head ownership.
The CPU-only host selection passed 567 tests with 4 skips (full FLA cache
adapters). It used `--noconftest` and excluded the five KDA NPU test modules:
`test_kda_fwd_npu`, `test_kda_bwd_npu`, `test_kda_bwd_deep_npu`,
`test_kda_caches_npu`, and `test_kda_layer_npu`. After four additional shape
rejection cases, the dedicated backward test file passed all 55 tests. These
counts are separate historical runs. The final FP32-only wrapper test file
passed **67 tests** in 8.88s, including 15 BF16 rejection cases (q/k/v/do individually
and together, across all three launchers). A TorchDispatchMode rejects any tensor
operation before the BF-02 error, and kernel imports are blocked in these cases.
See [host-test receipt](../../kernels/projects/a5/gdn_chunk_bwd/evidence/host-test-receipt.json).
These counts are not a claim that the entire repository's NPU suite ran.

The reduced B1/T128/H=HV1/bd1 pipe diagnostic has now passed all three stages:
empty event balance, no hazards/deadlock, all 10 arrays <=2.46e-7. This case retains
the GM tape overwrite between chunks. The larger timeout remains recorded and
is not relabeled a pass. The separate B1/T64/H1/HV3/bd1 uneven head-reuse diagnostic also passed all
three stages, with no hazards/deadlock and all 10 arrays <=2.48e-7. One vector
worker processes two heads while the other processes one. The reverse stage
finished in 212.96s within its 240s diagnostic bound, derived from the earlier
measured model cost. These are bounded model results, not device timings. See
[tape-reuse result](../../kernels/projects/a5/gdn_chunk_bwd/evidence/pipe-tape-reuse.json)
and [head-reuse result](../../kernels/projects/a5/gdn_chunk_bwd/evidence/pipe-head-reuse.json).

## Complete FP32 native grid and block_dim comparison

Each block_dim passed 69 FP32 cases: all four head ratios, C1/2/3 and T4096,
three cotangents, multi-batch, gate/beta endpoints, underflow, spikes and zero
q/k. All 10 composition arrays, 10 independent leaf arrays and 5 public gradients
passed, with inputs unchanged. The following maxima use only FP32 rows:

| Gradient | Against literal A | Against independent B |
|---|---:|---:|
| dq | 4.855229916e-07 | 4.851761855e-07 |
| dk | 6.724300833e-07 | 6.724300833e-07 |
| dv | 6.876718134e-07 | 6.876718134e-07 |
| dg | 8.329418537e-07 | 8.319868937e-07 |
| dbeta | 7.043351236e-07 | 7.043351236e-07 |

There are **138 accepted FP32 records** and **69 case pairs ×15 arrays =1035
byte-identical array pairs**. Every FP32 per-case numerical record is also
identical between block_dim 1/2. Both jobs began with the full T4096 workload
and finished Healthy. The [FP32-only summary](../../kernels/projects/a5/gdn_chunk_bwd/evidence/fp32-grid-summary.json)
can be recomputed by selecting exactly `dtype == "torch.float32"` from the
original [bd1 numbers](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd1.json)
and [bd2 numbers](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2.json).
The original raw logs and their line indexes are retained alongside these files.
Their BF16 rows and the old all-dtype byte comparison are historical records,
not evidence for the current public ABI. Receipts identify Ascend950PR_9589V100,
CANN9.2.0 and the built-in ascend950 operator package. Validation walltime is
not a performance measurement.

Core kernel and pipeline bytes are unchanged. The prior bd2 snapshot used three
earlier validation files; the current revision additionally changes the public
wrapper and FP32-only runners. The source identity review records these exact
revisions. Final-source T4096 closeout passed five records: three cotangent modes
at bd2, then the exact final Python snapshot at bd2 and bd1 for both cotangents.
All 75 old/new stage/public array hashes and every per-gradient numerical record
match the earlier FP32 grid; final bd1/bd2 also match all 15 arrays. The earlier
three-mode closeout differs only in the subsequently updated comparison utility;
its public wrapper/kernel bytes are final. Both final closeouts and refreshed
timing match all 190 Python files in the delivered snapshot. See
[closeout records](../../kernels/projects/a5/gdn_chunk_bwd/evidence/fp32-final-closeout.json)
and [source identity review](../../kernels/projects/a5/gdn_chunk_bwd/evidence/source-identity-review.json).
Old manifests remain historical; they are not relabeled as current. The older
bd1 snapshot predates the comparison utility, and bd2 predates both comparison
and timing utilities; the source review explicitly lists those absent files.

The first closeout launch failed before kernel compilation/execution because its
source manifest upload had not finished. The full
[failure log](../../kernels/projects/a5/gdn_chunk_bwd/evidence/failures/closeout-manifest-race.log)
is retained. A transfer marker now blocks launch during upload; the completed
snapshot was verified before retrying with a fresh run label.

## Build and runtime diagnostics

[Six selected build receipts](../../kernels/projects/a5/gdn_chunk_bwd/evidence/compiler/build-artifact-manifest.json)
retain hashes for 264 emitted/compiled files and every vendor build-log line.
Build directories are selected from the current kernel source signature, not
from stale directory timestamps. SDK header deprecations, an unused CMake
cross-compiler parameter, and a Python escape SyntaxWarning in the SDK's
non-raw script template are preserved. All six selected builds completed.
The original C++ keyword compile failure and larger bounded pipe timeout remain
in [failed-run evidence](../../kernels/projects/a5/gdn_chunk_bwd/evidence/failures/).

Two runtime startup warning classes were investigated without changing shared
software: plugin-policy query fallback at `plugin_version_manager.cpp:38`, and
TensorFlow framework preload failure at `ae_kernel_lib_fwk.cc:298`. A fresh
process importing only Torch/Torch NPU, with basic NPU transfer/multiply/add and
synchronization, reproduced both warnings while returning exact results; the
subsequent device scheduler initialization succeeded. The installed driver
header maps query return 3 to `DRV_ERROR_INVALID_VALUE`; its logged fallback is
`PLUGIN_NOT_FORCE_UPDATE`. The exact missing TensorFlow library remains
unidentified, and no TensorFlow backend support is claimed. The qualified path
uses task CCE and Torch NPU operations, whose actual outputs are checked above.

The [runtime review](../../kernels/projects/a5/gdn_chunk_bwd/evidence/runtime/runtime-review.json)
retains all 3410 lines from 56 logs: 28 startup warnings and
0 ERROR/FATAL/CRITICAL. Machine identifiers are redacted without removing lines.
This includes the completed device validation, controls and accepted timing run.

## Same-device measurements

The candidate is the complete public API, including allocations, all three launches and checkpoint recomputation.
All public inputs/outputs are FP32, with no dtype conversions.
The baseline is the same task-owned adjoint supplied with cached boundary
checkpoints; it excludes checkpoint generation. It is a cost baseline, not a
separate complete backward implementation. Neither timing path runs a host
recurrence. Inputs and independent B references are generated during the run.

The final FP32 scope is B1/H=HV8/K=V128, block_dim 2, T1024/4096. Each
workload used three baseline/candidate/baseline rounds, with 10 warmups and
50 samples per segment, synchronized before/after every call (900 total).
Independent B checks and baseline/candidate byte identity passed before/after.
The table reports ranges of segment medians; no speed threshold was imposed.

| T | Saved-checkpoint baseline median (ms) | Complete FP32 API median (ms) |
|---:|---:|---:|
| 1024 | 136.499–136.564 | 160.342–160.386 |
| 4096 | 545.793–546.337 | 645.316–645.520 |

[Raw samples](../../kernels/projects/a5/gdn_chunk_bwd/evidence/fp32-final-sandwich-bd2.json),
[unfiltered log](../../kernels/projects/a5/gdn_chunk_bwd/evidence/fp32-final-sandwich-bd2.log),
[round-to-log index and identity](../../kernels/projects/a5/gdn_chunk_bwd/evidence/timing-index.json),
and [summary](../../kernels/projects/a5/gdn_chunk_bwd/evidence/timing-summary.json)
record every number. Earlier BF16 timing is retained history and excluded from
GDA-03 delivery. No CUDA/Triton or pretrained-weight comparison is claimed.

The shared lock was held through context cleanup, with fresh environment/source
checks and Healthy status before/after. Read-only SVM/TRS registries were
sampled before/during/after the job and matched to the exact compute child.
The [occupancy receipt](../../kernels/projects/a5/gdn_chunk_bwd/evidence/timing-occupancy.json)
reports zero foreign contexts; the [positive control](../../kernels/projects/a5/gdn_chunk_bwd/evidence/occupancy-positive-control.json)
qualifies this method. Raw machine identities remain private.

An earlier complete measurement is [retained as unqualified](../../kernels/projects/a5/gdn_chunk_bwd/evidence/failures/timing-first-unqualified.json):
its per-device FD sampler missed a known live process. It is not used for the
accepted timing table. The corrected monitor and complete repeated measurement
supply the evidence above.

The canonical `board`/`aclnn` launchers remain untested for this task. Actual
vendor compilation and native device acceptance used the in-process CCE path;
functional simulation and the two reduced pipe diagnostics are separately
scoped evidence. No other backend, device family, input domain or weight-level
validation is claimed.
