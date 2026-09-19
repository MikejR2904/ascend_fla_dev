# GDA-03 grouped GDN backward ABI and range contract

Status: complete native grid, block_dim byte identity and same-device measurement passed.
This task is standalone backward. The accepted grouped forward and GDN-2 are
unchanged. The authority is FLA pin `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`,
`fla/ops/gated_delta_rule/naive.py::naive_recurrent_gated_delta_rule`.

## Frozen public ABI

`chunk_gdn_bwd(q,k,v,g,beta,do=None,dht=None,...) -> (dq,dk,dv,dg,dbeta)`.

| Item | Supported contract |
|---|---|
| q, k | Contiguous token-major `[B,T,H,128]`, matching FP32/BF16 |
| v | Contiguous `[B,T,HV,128]`, same dtype as q/k |
| g, beta | FP32 `[B,T,HV]`, finite `g<=0`, `0<=beta<=1` |
| Heads | Positive `H,HV`, `HV%H==0`, consecutive groups |
| B, T | Positive B; T multiple of 64, from 64 through 4096 |
| do | Optional `[B,T,HV,128]`, same storage dtype as output v |
| dht | Optional FP32 `[B,HV,128,128]`; at least one cotangent |
| dq, dk, dv | Input shapes/storage dtypes; internal gradients remain FP32 |
| dg, dbeta | FP32 `[B,T,HV]` |
| Initial state | Only None (zero); no dh0 output |
| Normalization | None; scale fixed `128**-0.5`, including dq's chain factor |
| Execution | A5 CCE, block_dim 1 or 2, explicit inprocess/aclnn/board launcher |
| Unsupported | Tails/decode, varlen, CP, head-first, transposed state, normalization, other scale/dtypes/devices |

CPU launchers check finite/range preconditions. The in-process NPU path inherits
forward's caller preconditions, with shape/dtype/layout checked before dispatch.
The API computes a first derivative explicitly; it does not wire autograd or
provide higher derivatives. Storage rounding is not differentiated. BF16 is
promoted exactly before computation and dq/dk/dv are rounded only on return.
Runtime preparation is allocation, optional-cotangent zeros, promotion and return
casts on the input device; no host recurrence or reference fallback is permitted.
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
acceptance is <=1e-4 per gradient against **both** A and B. BF16 public storage
quality is <=5e-3, reported separately. Negative controls zero, negated, and
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
passed FP32 and BF16 public-output checks. FP32 core maximum against both
references (including BF16-valued inputs) was 2.71993077e-7. Public BF16 dq/dk/dv
relative L2 was 0.00165841/0.00167523/0.00163092 against A. All 10 composition and
all 10 independently supplied leaf outputs passed; inputs were unchanged.
[Raw native output](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-bd2.log),
[numbers](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-bd2.json), and
[verified source manifest](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-source-manifest.json)
identify this run. The complete grid and block_dim comparison are recorded below.

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
counts are separate runs; they are not a claim that the entire repository's
NPU suite ran.

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

## Complete block_dim=2 native grid

All 69 canonical cases ran with FP32 and BF16 public inputs (138records): all four
head ratios, C1/2/3 and T4096, three cotangents, multi-batch, gate/beta endpoints,
underflow, spikes and zero q/k. All 10 composition arrays, 10 independent-leaf
arrays and 5 public gradients passed, with inputs unchanged. Health was checked
before and after the locked Docker job. Core FP32 maxima over both input dtypes:

| Gradient | Against literal A | Against independent B |
|---|---:|---:|
| dq | 4.855229916e-07 | 4.851761855e-07 |
| dk | 6.737140044e-07 | 6.737140044e-07 |
| dv | 6.876718134e-07 | 6.876718134e-07 |
| dg | 8.833763473e-07 | 8.841634564e-07 |
| dbeta | 7.043351236e-07 | 7.043351236e-07 |

Public BF16 dq/dk/dv maxima against A are 0.00168075/0.00184963/0.00175118.
The actual dg/dbeta outputs remain FP32.
[Raw log](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2.log),
[per-case numbers](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2.json),
[environment receipt](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2-receipt.json),
and [case-to-log-line index](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2-index.json)
preserve each number and its source. The receipt identifies A5
Ascend950PR_9589V100, CANN9.2.0, and built-in ascend950 operator package.
Validation walltime is not a performance measurement.

## Complete block_dim pair qualification

The block_dim=1 grid also passed all 69 cases x both public dtypes. Across the
combined 276 records, all original budgets and input-immutability checks passed;
per-gradient maxima are unchanged from the table above. Comparing the exact
returned byte hashes gives 138 case/dtype pairs x (10 stage + 5 public) arrays =
**2070 byte-identical array pairs**. Both jobs finished with Healthy device status.
The second grid began with the original B1/T4096/H=HV8 workload before smaller
cases. [bd1 raw log](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd1.log),
[bd1 numbers](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd1.json),
[bd1 line index](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd1-index.json),
and [complete byte comparison](../../kernels/projects/a5/gdn_chunk_bwd/evidence/bd1-bd2-byte-comparison.json)
make this qualification reproducible with `compare_runs.py`.

The delivered production Python files match both device snapshots. The bd1 run
uses the final validation harness; the bd2 manifest identifies earlier versions
of three validation files. Every per-case metric against A, B and stage
references is also identical between runs, in addition to the returned-array
byte identity. The [source identity review](../../kernels/projects/a5/gdn_chunk_bwd/evidence/source-identity-review.json)
records these harness revisions explicitly.

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
retains all 2438 lines from 39 logs: 20 startup warnings and
0 ERROR/FATAL/CRITICAL. Machine identifiers are redacted without removing lines.
This includes the completed device validation, controls and accepted timing run.

## Same-device measurements

The candidate is the complete public API, including device promotions,
allocations, all three launches, checkpoint recomputation and storage casts.
The baseline is the same task-owned adjoint supplied with cached boundary
checkpoints; it excludes checkpoint generation. It is a cost baseline, not a
separate complete backward implementation. Neither timing path runs a host
recurrence. Inputs and independent B references are generated during the run.

For B1/H=HV8/K=V128 and block_dim=2, each shape/dtype used three
baseline/candidate/baseline rounds. Every segment had 10 warmups and 50 measured
calls, synchronized before and after each call: 1800 raw wall-time samples.
The table reports the range of segment medians across the three rounds.

| T | Public dtype | Baseline median range (ms) | Complete API median range (ms) |
|---:|---|---:|---:|
| 1024 | float32 | 135.578–135.710 | 159.210–159.324 |
| 1024 | bfloat16 | 135.690–135.866 | 160.210–160.324 |
| 4096 | float32 | 545.436–545.460 | 645.160–645.225 |
| 4096 | bfloat16 | 545.308–545.508 | 644.387–644.625 |

Mathematical checks against independent B and byte identity between baseline
and candidate passed before and after each workload. FP32 dg/dbeta retain the
1e-4 budget for both public dtypes. No speed threshold was imposed.
[Raw samples](../../kernels/projects/a5/gdn_chunk_bwd/evidence/sandwich-context-bd2.json),
[unfiltered native log](../../kernels/projects/a5/gdn_chunk_bwd/evidence/sandwich-context-bd2.log),
[round-to-log index and target identity](../../kernels/projects/a5/gdn_chunk_bwd/evidence/timing-index.json),
and [summary](../../kernels/projects/a5/gdn_chunk_bwd/evidence/timing-summary.json)
record every measured number. These measurements do not compare CUDA/Triton or
pretrained-weight execution.

The job held the shared device lock in Docker, with fresh source/environment
checks and Healthy device status before and after. A known live control first
qualified the installed driver's read-only SVM and TRS process registries.
The timing audit retained 409 snapshots, matched contexts to the exact
compute child, found zero foreign contexts, and waited for its contexts to drain
before releasing the lock. See the [occupancy receipt](../../kernels/projects/a5/gdn_chunk_bwd/evidence/timing-occupancy.json)
and [positive control](../../kernels/projects/a5/gdn_chunk_bwd/evidence/occupancy-positive-control.json).
Raw machine identities remain private.

An earlier complete measurement is [retained as unqualified](../../kernels/projects/a5/gdn_chunk_bwd/evidence/failures/timing-first-unqualified.json):
its per-device FD sampler missed a known live process. It is not used for the
accepted timing table. The corrected monitor and complete repeated measurement
supply the evidence above.

The canonical `board`/`aclnn` launchers remain untested for this task. Actual
vendor compilation and native device acceptance used the in-process CCE path;
functional simulation and the two reduced pipe diagnostics are separately
scoped evidence. No other backend, device family, input domain or weight-level
validation is claimed.
