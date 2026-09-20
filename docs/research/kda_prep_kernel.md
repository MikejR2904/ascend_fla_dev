# KDA raw preprocessing kernels (BF-07 stage 1)

BF-07 moves enabled raw q/k normalization, gate transform and beta sigmoid into
a new `kernels/projects/a5/kda_prep` unit on A5. The selected source baseline is
`c09a80ed8dbb96fdde7642eee21bec00d8f38e3a`; three byte-preserved public wrappers
and their SHA256 identities are in `baseline/`. Existing stable forward,
backward and layout kernels remain read-only.

## Current evidence

The first budget commit, `c9409a2`, contains **100 CPU calibration cases** and
**66 rejected corrupt-output controls**, Python3.11.15 / Torch2.10.0+cpu, before
any candidate kernel source. The candidate now emits all28 typed chunk/decode
entries to CCE. All50 vendors compile at bd1,2,3,4. The complete Kimi T4096 all-flags
workload passes at each of bd1,2,3,4. All17 returned outputs, caches and gradients
are bitwise identical across those four runs. After that hardware
run, all14 bounded source cases pass sim and pipesim at bd1. Remaining native
grids, decode block dimensions, endpoints, repeatability and performance are pending;
this is not completed BF-07 acceptance.
FP64 is used only to measure preprocessing numerical error; the independent
and pinned FLA end-to-end KDA goldens remain **Torch CPU FP32**.

The seed7007 generator covers K128, B1/B2, C1/C3 and full Kimi B1/T4096/H32,
all four norm input/output dtype combinations and eight independent raw gate
input/A_log/dt_bias dtype tuples. It includes zero/near-zero/large rows,
softplus threshold20 neighbors, sigmoid saturation, exponent over/underflow,
NaN/Inf and signed zero. Inputs are rounded to their declared BF16/FP32 dtype
before either reference path. Log/exp/sigmoid outputs stay FP32; normalized
q/k use BF16 for chunk or v.dtype for decode. Disabled flags preserve identity.

## Frozen ordinary finite numerical budgets

`budgets.json` is the machine-readable authority, fixed in this first commit
before kernel implementation. F is the maximum measured predecessor error
against independent FP64 math for each operation/dtype tuple. Relative-L2
budgets are3F (norm additionally capped at1e-2); elementwise relative budgets
are3 times that group's measured maximum relative error. There is **no additive
absolute tolerance**; zero references must match exactly. Against the old host
path we also report all differences, without replacing the FP64 comparison.

| Operation / dtype tuple | Relative-L2 floor F | Relative-L2 limit | Elementwise relative limit |
|---|---:|---:|---:|
| norm:bf16_bf16 | 0.00168262802843 | 0.00504788408528 | 0.0116729698285 |
| norm:bf16_f32 | 6.29763341467e-08 | 1.8892900244e-07 | 5.42259363867e-07 |
| norm:f32_bf16 | 0.00166844413381 | 0.00500533240143 | 0.0116731448133 |
| norm:f32_f32 | 5.3027837224e-08 | 1.59083511672e-07 | 5.29366160389e-07 |
| gate:bf16_bf16_bf16 | 4.61923600487e-08 | 1.38577080146e-07 | 5.10637936927e-07 |
| gate:bf16_bf16_f32 | 4.88042180112e-08 | 1.46412654034e-07 | 8.14464019558e-07 |
| gate:bf16_f32_bf16 | 4.33976365207e-08 | 1.30192909562e-07 | 5.83186411394e-07 |
| gate:bf16_f32_f32 | 4.57815534948e-08 | 1.37344660484e-07 | 7.29829239364e-07 |
| gate:f32_bf16_bf16 | 4.61968317111e-08 | 1.38590495133e-07 | 8.41324823171e-07 |
| gate:f32_bf16_f32 | 4.76932359888e-08 | 1.43079707967e-07 | 8.00884112086e-07 |
| gate:f32_f32_bf16 | 4.46002123875e-08 | 1.33800637163e-07 | 8.57793647117e-07 |
| gate:f32_f32_f32 | 4.64723427901e-08 | 1.3941702837e-07 | 7.16692450484e-07 |
| beta:bf16 | 3.89619498087e-08 | 1.16885849426e-07 | 3.54302567418e-07 |
| beta:f32 | 3.82180445751e-08 | 1.14654133725e-07 | 3.46300092717e-07 |

BF16 normalized outputs must also differ by at most1ULP from the correctly
rounded FP64 reference and the predecessor FP32 host result, separately. The predecessor's FP32 normalized outputs themselves reach
2–3ULP against correctly rounded FP64 (BF16-input maximum3, FP32-input maximum2).
This was reported to PM **before implementation**. FP32 differences above1ULP
are reported with a located reduction/sqrt/division explanation. PM clarified
that FP32 has no ULP pass line; its frozen relative-L2 and elementwise3F limits
are the numerical criteria. The frozen budget file is unchanged.

## Range and nonfinite semantics

These observations do not inflate ordinary budgets or remove input cases. The
native verifier must compare actual predecessor and candidate endpoint masks
and finite values, report FP64 ideal differences, and retain any discrepancy.
No new range gate or silently narrower input domain is authorized.

| Boundary | Predecessor FP32 behavior observed / required comparison |
|---|---|
| K128 zero / near-zero row | Additive epsilon1e-6; zero remains zero, small rows divide by approximately1e-3; compare signed zero |
| x squared or summed squares overflow | At scale1e20 the predecessor returns finite zero norm outputs although FP64 ideal is nonzero; this is a retained FP32 semantic endpoint, not accuracy against FP64 |
| exp(A_log) overflow | Can overflow before multiplication even where ideal FP64 final value would fit; compare NaN/Inf masks, do not reassociate or clip |
| exp(A_log) underflow | Subnormal/zero multiplier; report quantization and signed-zero consequences separately |
| softplus threshold20 | Strict u>20 branch returns u; its derivative is exactly1 on that branch, not an unconditionally evaluated sigmoid |
| large negative softplus | Preserve/report subnormal and zero behavior; no ln(1+exp(u)) cancellation shortcut justified by this CPU study |
| sigmoid saturation | FP32 rounds some large positive outputs to1 and negative outputs to subnormal/zero; no broad relative-error budget from those endpoints |
| NaN / Inf inputs | Compare propagation masks and finite siblings; a comparison over an empty finite subset never counts as numeric success |

End-to-end limits stay o/state/dq/dv/dbeta/dh0=.05, dk=.15, dg=.25 relative-L2
against both CPU FP32 references, with existing elementwise checks. Gate limits
stay forward155/backward105 and require fresh tests at0,100.8,105,155. Block
sizes must produce identical bytes. D-PM-42 arithmetic exceptions and D-PM-44
inherited scan uncertainty retain their original scope.

## Reproduce

From the selected accepted CPU environment, with this unit directory as cwd:

```bash
python ref/calibrate.py --output evidence/calibration
python verify_calibration.py
```

The standalone study imports only Torch and the verified predecessor snapshot;
it does not import candidate kernel code or access devices/network. Text raw
records are `evidence/calibration/pre-kernel.json`; summary and all-file hashes
are adjacent. Inputs and references regenerate at run time from the seed.
PM split the custom preprocessing gradient chain into BF-08. That task must
separately calibrate and freeze raw/parameter-gradient budgets before writing
its backward kernels.


## Candidate architecture and dispatch boundary

There are14 typed variants per namespace: four norm input/output dtype pairs,
eight independent gate input/A_log/dt_bias tuples and two sigmoid inputs. Chunk
and decode use separate operator names so their block dimensions can differ in
one process. Existing public `prepare` reaches the compilation hooks; all50
vendors (28 prep,6 layout,5 forward,9 backward,2 decode) register before the first
custom execution. Each block dimension still requires a separate process.

The three vector kernels own disjoint4096-element tiles, with a single UB slot
and explicit MTE2/VF/MTE3 dependencies under auto_sync. Gate tiles contain up
to32 token rows for one head. Norm reduces each K128 row as two64-lane cadds,
then adds the partial sums, additive1e-6, sqrt and division in that fixed order.
BF16 stores round to nearest even. Gate implements compensated log1p(exp(u))
and the strict u>20 branch without clipping exp(A_log). Sigmoid uses FP32
1/(1+exp(-x)). Native endpoint behavior remains to be measured, not inferred.

All flags enabled add four preparation launches: gate, q, k and beta. Disabled
routes retain the input object and launch nothing. Host work is metadata,
validation, allocation and dispatch. Chunk compiles all nine backward vendors
before preparation; decode compiles both decode dtypes and its14 prep vendors.

Chunk chooses the retained training graph before preparation when grad mode is
on and any q/k/v/g/beta/initial_state requires grad, or when enabled gate
parameters require grad. This includes parameter-only training. `no_grad`
always selects kernel preparation; decode always selects the inference path.
The original `_prepare_inputs` remains differentiable and is a **noncompliant
training exception pending BF-08**. D-PM-42 and D-PM-44 retain their scope.

## Host checks and pending native evidence

The new117 host cases check typed launch arguments, eight flag combinations,
BF16/FP32 input and output pairs, both namespaces, disabled identity, unchanged
inputs, compilation order, dtype rejection, and training/no_grad routing.
The approved domain suite has71 passing cases; the combined layer/decode/prep
boundary suite has146. AST checks confirm only the five approved existing test
functions changed and only the public module docstring changed in `__init__`.
In the isolated three-entry base swap, the52 public dtype/routing checks yield
44 expected failures and8 retained training passes. These are host ABI evidence,
not numerical kernel validation. All stable/layout kernel sources remain intact.

Remaining native completion requires the full dtype/flag/shape and gate endpoint
grids, repeatability and training checks, decode block dimensions, cross-bd
comparison of the entire grid, and three same-card old/new/old timing rounds
with Torch NPU baseline,
and a verified fresh restore of the final evidence archive.


## First complete native workload

New evidence: `evidence/native/full-v1-bd4/`, CANN9.1.0-beta.1 compiler timestamp
20260509_173000235; Python3.12.14, Torch2.12.0+cu130, torch_npu2.12.0. Every
native JSON embeds raw compiler/OPP version lines and version.info hashes.
The complete public all-flags forward, cached forward and backward execute49
custom launches after all50 vendors have registered. Plain/cached o and state
are bitwise equal; all nine caches and six gradients are finite; all ten inputs
are unchanged; actual TorchDispatch has no unregistered arithmetic/copy entry.

Prep relative-L2 errors against FP64 are q0.00165520, k0.00165529,
g6.16113e-8, beta4.07918e-8, all below frozen budgets. BF16 q/k differ by at
most1ULP from each comparison object. FP32 gate reaches4ULP versus roundedFP64
and6 versus host; beta reaches2 and3. These distributions require the retained
operation-boundary explanation, not a looser budget.

Against each CPU FP32 oracle using predecessor CPU preparation, gradient
relative-L2 is dq0.0280442, dk0.0330546, dv0.0035786, dbeta0.0038455,
dg0.0629151, dh0 0.0023648. Both goldens also pass using actual native preparation
as their input. Per-head/per-chunk forward checks pass. This single full case
does not establish the D-PM-44 device-repeatability boundary or the remaining grid.

The complete host suite is1001 passed /5 NPU-module skips, with1001 collected
cases and per-file counts in `evidence/host/summary.json`. The declared14 bounded
reference cases pass; sim and pipesim each pass14 only after the full hardware
run, with pipe evidence retaining balance, hazard and deadlock fields.

## Native endpoint investigation (not yet qualified)

Two leaf runs each execute100 cases in both chunk/decode namespaces at bd4.
All120 ordinary finite cases meet the frozen budgets; all200 input hashes and
output guard regions remain unchanged. Of80 endpoint cases,28 exactly match
the CPU predecessor and52 have finite-value differences. NaN/Inf masks and
finite signs agree throughout. The exactness flag is a diagnostic trigger,
not a replacement for the agreed error criteria. Both raw runs are retained.

The second run also executes the byte-pinned predecessor on the same NPU.
It flushes the same tested CPU subnormal gate/beta values to zero. For beta
saturation the candidate and predecessor NPU outputs are identical. At
softplus input−87 and A_log−0.2, however, the predecessor NPU returns−0 while
the candidate returns−1.3474764119981045e−38; CPU FP32 returns
−1.347476552127951e−38 and FP64−1.3474764283785483e−38. That nonzero
candidate result is normal FP32. This requires a located operation comparison
and explicit endpoint review, not a wider tolerance, new gate, or replacement
of the CPU reference. Large finite norm rows show the already declared fixed
reduction rounding difference; exponent overflow masks agree with both actual
predecessors. Full range qualification remains open.

The retained allocator notice comes from the package-reported
[AddPadSize owner source](https://github.com/Ascend/pytorch/blob/fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L179).
It reports compatibility padding, not a requested tensor-arithmetic change.
Every native input/output guard passed; the notice was not suppressed.
Runtime/source line numbers differ, so the reported git version alone is not
asserted to prove an unmodified binary. See native/allocator-notice.json.
