# GDA-03 grouped GDN backward ABI and range contract

Status: native block_dim=2 grid passed; block_dim=1 and timing remain in progress.
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
replays64 tokens into FP32 `[B,HV,64,128,128]` scratch, then reverses them.
It emits per-value-head dq/dk and final dv/dg/dbeta. The third deterministically
sums dq/dk in ascending group order, with one owner per `(B,T,H)` and no atomics.
There are no checkpoints supplied by or modifications to the forward operator.

Each vector owner has a single UB slot per tensor. State and state adjoint
require128KiB together; row/scalar buffers fit in the selected profile's remaining
UB. State rows are128 FP32 elements, exactly two vector registers. DMA and VF
lifetimes use local autosync; explicit VF STORE→LOAD barriers protect dependent
local reloads. A local MTE3→MTE2 event publishes the replay tape; a local drain
precedes tape reuse. Stream-ordered launches publish inter-stage GM edges.
The tape has one independent slice per value head and costs32MiB at B1/HV8,
independent of T. All outputs are fresh, fully written, and inputs are read-only.

## Initial measured evidence (whole-task validation remains in progress)

Before writing device kernels,58 completed calibration cases passed A/B <=1e-5;
maximum per-gradient relative L2 was dq2.47549645e-7, dk0, dv0,
dg4.84878352e-7, dbeta0. Both FP64 checks passed at ratios1/2. The preserved
[pre-kernel report](../../kernels/projects/a5/gdn_chunk_bwd/evidence/pre-kernel-calibration.json)
contains individual case numbers. The completed extended calibration now has69
cases, including all four ratios atT4096 and all three cotangents, with unchanged
maxima; see [full calibration](../../kernels/projects/a5/gdn_chunk_bwd/evidence/calibration.json).

The first complete native workload B1/T4096/H=HV8/K=V128, both cotangents, bd2,
passed FP32 and BF16 public-output checks. FP32 core maximum against both
references (including BF16-valued inputs) was2.71993077e-7. Public BF16 dq/dk/dv
relative L2 was0.00165841/0.00167523/0.00163092 against A. All10 composition and
all10 independently supplied leaf outputs passed; inputs were unchanged.
[Raw native output](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-bd2.log),
[numbers](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-bd2.json), and
[verified source manifest](../../kernels/projects/a5/gdn_chunk_bwd/evidence/first-full-source-manifest.json)
identify this run. This is not yet evidence for the remaining grid, bd1/bd2
identity or the required timing sandwiches. Bounded synchronization evidence is below.

Initial vendor compilation rejected the internal parameter name `do`, a C++
keyword, before device computation. Renaming the native argument to `dout`
fixed compilation; the public parameter remains `do`. No equation or budget
changed. Full failed compiler logs remain in task scratch.

The canonical CPU reference sweep passed all138cases. A bounded functional
model case (C1/equal heads/bd1) passed all ten independent leaf arrays and five
composition gradients. The first larger pipesim (C2/grouped/multiple head items)
timed out in the interpreter with an empty blocked-lane summary. Two smaller source diagnostics isolate tape reuse and uneven head ownership.
Host regression:567passed/4skipped (full FLA cache adapters).

The reduced B1/T128/H=HV1/bd1 pipe diagnostic has now passed all three stages:
empty event balance, no hazards/deadlock, all10arrays<=2.46e-7. This case retains
the GM tape overwrite between chunks. The larger timeout remains recorded and
is not relabeled a pass. The separate B1/T64/H1/HV3/bd1 uneven head-reuse diagnostic also passed all
three stages, with no hazards/deadlock and all10arrays<=2.48e-7. One vector
worker processes two heads while the other processes one. The reverse stage
finished in212.96s within its240s diagnostic bound, derived from the earlier
measured model cost. These are bounded model results, not device timings. See
[tape-reuse result](../../kernels/projects/a5/gdn_chunk_bwd/evidence/pipe-tape-reuse.json)
and [head-reuse result](../../kernels/projects/a5/gdn_chunk_bwd/evidence/pipe-head-reuse.json).

## Complete block_dim=2 native grid

All69canonical cases ran with FP32 and BF16 public inputs (138records): all four
head ratios, C1/2/3 and T4096, three cotangents, multi-batch, gate/beta endpoints,
underflow, spikes and zero q/k. Every10composition array,10independent-leaf
array and5public gradient passed, with inputs unchanged. Health was checked
before and after the locked Docker job. Core FP32 maxima over both input dtypes:

| Gradient | Against literal A | Against independent B |
|---|---:|---:|
| dq | 4.855229916e-07 | 4.851761855e-07 |
| dk | 6.737140044e-07 | 6.737140044e-07 |
| dv | 6.876718134e-07 | 6.876718134e-07 |
| dg | 8.833763473e-07 | 8.841634564e-07 |
| dbeta | 7.043351236e-07 | 7.043351236e-07 |

Public BF16 dq/dk/dv maxima against A are0.00168075/0.00184963/0.00175118.
The actual dg/dbeta outputs remain FP32.
[Raw log](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2.log),
[per-case numbers](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2.json),
[environment receipt](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2-receipt.json),
and [case-to-log-line index](../../kernels/projects/a5/gdn_chunk_bwd/evidence/grid-bd2-index.json)
preserve each number and its source. The receipt identifies A5
Ascend950PR_9589V100, CANN9.2.0, and built-in ascend950 operator package.
Validation walltime is not a performance measurement.
