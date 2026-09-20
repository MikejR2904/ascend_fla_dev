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
run, all14 bounded source cases pass sim and pipesim at bd1. Ordinary chunk grids, regression and training checks, fresh scan repeatability, and
synchronized performance measurements have also completed. All required grid populations have now executed. Nearzero state and output
failures, their unresolved acceptance boundary, and the final delivery archive
still block BF-07 acceptance. Endpoint reachability, high-bd leaf checks and
metric-comparison reruns have completed. The task and PR remain incomplete.
FP64 is used for preprocessing error studies and reliable norm accumulation; the independent
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
1/(1+exp(-x)). Native endpoint behavior is measured below; expanded nonfinite
cross-products are diagnostic classifications without a numerical pass line.

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

## Host checks

The new117 host cases check typed launch arguments, eight flag combinations,
BF16/FP32 input and output pairs, both namespaces, disabled identity, unchanged
inputs, compilation order, dtype rejection, and training/no_grad routing.
The approved domain suite has71 passing cases; the combined layer/decode/prep
boundary suite has146. AST checks confirm only the five approved existing test
functions changed and only the public module docstring changed in `__init__`.
In the isolated three-entry base swap, the52 public dtype/routing checks yield
44 expected failures and8 retained training passes. These are host ABI evidence,
not numerical kernel validation. All stable/layout kernel sources remain intact.

Remaining completion requires resolution of retained nearzero failures
and a verified fresh restore of the final source and evidence archive. Completed
regression, repeatability and timing stages are recorded separately below.


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

## Native endpoint investigation and remaining failures

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
reduction rounding difference. Exponent overflow masks agree with both actual
predecessors in this initial population only; the expanded cross-products below
expose disagreements. Full range qualification remains open because of the composed nearzero failures below.

The retained allocator notice comes from the package-reported
[AddPadSize owner source](https://github.com/Ascend/pytorch/blob/fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d/torch_npu/csrc/core/npu/NPUCachingAllocator.cpp#L179).
It reports compatibility padding, not a requested tensor-arithmetic change.
Every native input/output guard passed; the notice was not suppressed.
Runtime/source line numbers differ, so the reported git version alone is not
asserted to prove an unmodified binary. See native/allocator-notice.json.

The18-value native primitive trace now locates two distinct boundaries
(`native/endpoint-v1-bd4/stages.json`). At−100/−90/−88, native `exp`
returns0 where CPU FP32 returns subnormal values. At−87/−86/−40/−20,
native `exp` and native `log1p(exp(x))` retain nonzero normal values, but
native `softplus` returns0; at−16 it returns1.1920928244535389e−7 versus
CPU1.1253516873921399e−7. The observed softplus values match the cancellation
boundary of `ln(1+exp(x))`; this is an inference from operation results, not
a claim about unavailable vendor implementation source. The candidate's
compensated branch retains the independent CPU/FP64 value within the frozen
ordinary threshold-case budgets. The owner subsequently clarified that native
underflow/saturation behavior is recorded explicitly, with no CPU-subnormal
bitwise requirement. Ordinary budgets and domains remain fixed. The expanded cross-products below use the subsequent explicit diagnostic
classification decision.


## Completed ordinary grids and regressions

The ordinary public Cartesian grid covers C1/2/3, HV/H1/2/4/8, all eight flags
and independent supported input dtype combinations. Chunk has1620 cases per
block dimension; bd1/2/3/4 all pass and every output, all nine caches and every
preparation tensor has identical hashes across those runs. Decode has3240 cases
per dimension, including BF16 and FP32 v. All bd1/2/4/8/16/28 runs pass and every returned tensor and preparation
value is bitwise identical across the six dimensions. The full ten-run ordinary
grid contains25,920 native cases and51,870 self-contained original JSON receipts;
all have been losslessly archived and restored with matching original hashes. Decode uses legal16-token public
calls and carries the returned FP32 state through each C*64-token sequence.
Every process registers all50 vendors, including all nine backward entries,
before executing a custom kernel. These counts exclude failed boundary cases.

`native/regression-v1-bd{1,2,3,4}` records10 backward cases per dimension:
the five original cases, T1024 with H32/HV32 and H16/HV32, and gate spans
0/100.8/105. All retain the existing six gradient budgets. The six gate tests
at0/100.8/105/155/105.001/155.001 preserve the public forward155 and cached
backward105 boundaries. Thirty training combinations per dimension include
both raw dtypes, all flags, and A_log-only / bias-only gradients. Retained host
training outputs and gradients are bitwise identical to the predecessor. This
verifies preserved training semantics, not custom preprocessing backward support;
that noncompliant host graph remains assigned to BF-08.

`native/repeat-v1-bd4` is a fresh D-PM-44 population qualification:
B2/H2/HV4/C3/span46/seed2026,12 predecessor calls and12 candidate calls with
the original poison pattern and no inserted internal barriers. All eight
returned tensors have one hash across24 calls, with dh0 relative-L2
0.0023368734400719404 against both CPU FP32 references. Eight separately
captured scan inputs match, and12 direct scan replays are identical. This is
limited to the measured device/input window; the inherited scan defect is not
claimed fixed.

The third leaf series runs200 cases at each of bd1/2/3/4. Its120 ordinary
numerical cases and80 qualified range cases pass the clarified criteria.
High decode bd8/16/28 each also complete200 cases. All100 input populations
are byte-identical across14 combinations of run and chunk/decode namespace
(including repeated chunk-bd4 controls). This leaf result does not qualify
the entire composed input range.

## Expanded range failures retained for review

The48 actual cross-products in `native/endpoint-v2-bd4/downstream.json` retain
candidate, predecessor NPU, CPU FP32 and FP64 gate values, plus actual NPU
exp(g). Explicit IEEE bits distinguish JSON null values representing NaN/Inf.
Representative downstream observations are:

| A_log | u | Candidate NPU exp(g) | Predecessor NPU exp(g) | Difference |
|---:|---:|---:|---:|---:|
| -0.2 | -87 | 1 | 1 | 0 ULP |
| -0.2 | -16 | 0.9999999403953552 | 0.9999998807907104 | 1 ULP |
| 2.7 | -16 | 0.9999983310699463 | 0.9999982118606567 | 2 ULP |
| 80 | -87 | 0.9990885257720947 | 1 | 15292 ULP |
| 80 | -20 | 0 | 1 | Different limiting decay |

The last rows show why softplus cancellation cannot be called universally
negligible. The public chunk span check rejects the very large processed-gate
case; a leaf result does not override that existing gate.

For A_log89/100 and u=-100/-90/-88, candidate and predecessor NPU produce
NaN while CPU FP32 produces -Inf. For those A_log values and u=-87/-40/-20,
candidate agrees with CPU FP32 (-Inf) while the predecessor NPU produces NaN.
The independent FP64 gate remains finite. Matching both predecessor masks is
impossible for these combinations. PM5750678047 explicitly removes that
impossible equality requirement for this diagnostic endpoint population: no
numerical pass line is assigned. The48 points classify as16 finite ties to
FP64,16 finite candidate improvements over old NPU,4 shared nonfinite classes,
6 candidate/CPU agreements differing from old NPU, and6 candidate/old-NPU
agreements differing from CPU. All16 nonfinite points have A_log89/100, outside
the FLA default log(U(1,16)) initialization range. This classification does not
change the ordinary domain, frozen budgets, kernel or155/105 gates.

The A_log80/u=-88 control is candidate/old-NPU zero while CPU FP32 gives
-0.000335462624207139. The normal final product would be representable, so
this result locates the underflow before the final multiplication. Together
with the native exp(-88)=0 trace and the retained kernel operation order, it
explains the later Inf*0=NaN class at A_log89/100. Public entry rejection or
propagation was measured separately:96 actual public-entry calls cover all48
points in chunk and decode. Chunk yields29 finite/19 rejected candidate cases
versus31/17 old-NPU cases. All16 nonfinite processed-gate cases are rejected
by the existing default span check. Decode has no such span gate:40 finite/8 NaN
candidate returns versus34 finite/14 NaN predecessor returns. These are observed
endpoint classes, not accuracy passes; every input remains unchanged. See
`native/reachability-v1-bd4`.

Two public boundary failures are retained in `native/boundary-diagnosis`:

- Chunk `nearzero_c2_g2_rbf16_vbf16_flags010`: q/k scaled by1e-20;
  second chunk/head1 CPU FP32 o has magnitude at most approximately7.85e-44,
  below BF16's smallest subnormal. Returned BF16 o is numerically zero. The
  saved tensor inspection finds3991/3670 signed-zero differences against the
  two correctly rounded CPU goldens: numeric ULP distance is0, but byte equality
  is false. The earlier torch.equal-based wording was corrected. Relative-L2 is1,
  above0.05, although global o/state pass. The verifier accumulates norms in
  FP64 to avoid FP32 underflow masking this failure; the golden is CPU FP32.
- Decode `nearzero_c2_g1_rbf16_vbf16_flags110`: all enabled preparations meet
  their frozen budgets, but final_state relative-L2 is0.0004308748 against the
  predecessor-CPU-prepared golden, above the retained BF-06 limit1e-5. The
  state is normal FP32, around1e-17. The located rerun uses actual-native-preparation inputs for both CPU
  references: state relative-L2 is7.23693e-8 /3.96414e-8. Raw and directly
  prepared public calls are bitwise identical. This locates the discrepancy
  before the recurrence, at the preparation/composition comparison boundary.
  The original1e-5 failure is retained; neither the limit nor its acceptance
  reference has been changed. See `boundary-probe-v2-decode-bd1` alongside
  the original failures.

The original numerical failures remain available in
[the located boundary RISK](https://github.com/ddddwee1/ascend_fla_dev/issues/106#issuecomment-5750600450).
PM5750678047 separately qualifies BF16 slices whose correctly rounded CPU
golden is entirely zero using at most1ULP from numeric zero; raw relative-L2
and signed-zero differences remain. This does not cover the wider failures below.

## Complete special-value collection and unresolved nearzero failures

Zero, threshold20 and beta-saturation populations pass576 chunk cases per
block dimension and1152 decode cases per dimension:9216 native cases across
the required four/six block dimensions, respectively. Their returned outputs,
caches and preparations match bitwise across dimensions. They are independent
partitions; nearzero remains a required population and is reported separately.

All nearzero runs are complete. Each chunk bd1/2/3/4 has192 cases:176 numerical
passes and16 output failures. Each decode bd1/2/4/8/16/28 has384 cases:324 passes,
16 state failures and44 output failures. All returned/preparation hashes,
failure locations and metrics are identical across the respective dimensions.
`--observe-all-cases` only keeps collecting after a numerical failure: its
zero exit status means collection finished, while the receipt retains
`passed=false` and `diagnostic_collection_only=true`. No kernel exception is
converted to a pass. Every failed case retains the actual tensors privately,
and their hashes are verified against the original public receipts before analysis.

The16 decode state failures are C2/3 × group1/2/4/8 × flags110/111, with BF16
raw q/k and BF16 v. Maximum old-CPU-prep state relative-L2 is
0.0009824887023388708 at `nearzero_c3_g1_rbf16_vbf16_flags111`, FLA reference.
The unchanged1e-5 comparison remains failed, retained as an observation after the user-approved D-PM-51 change. Fresh
all-bd reruns against actual native-prep CPU FP32 references are in progress;
state stays limited to1e-5, and prep budgets plus raw/prepared byte equality
are required. This permission applies only to the named nearzero decode group.
Ordinary decode still compares against **old CPU preparation**: across all six
dimensions its median/p99/max state errors are8.092760291e-8 /
2.983920105e-7 /3.266389653118004e-7. The maximum occurs at
`c2_g1_vf32_flags100_bf16_bf16_f32_f32_f32_f32`, independent reference.

The48 native normalization probes in `native/norm-rounding-v1-bd4` distinguish
FP32 arithmetic from the final BF16 cast. In all48 groups, casting the actual
FP32-output variant reproduces the actual typed output bytes. Ordinary BF16
outputs match the old CPU values in this population. Nearzero BF16→BF16 C2
q/k differ at173/185 of16384 elements, relative-L2
0.0005294714281 /0.0005775565286. All48 saved differing BF16 samples have an
FP64 ideal result exactly at a BF16 midpoint, with candidate and old FP32
quotients on opposite sides. Epsilon-dominated denominators and discrete BF16
inputs produce these ties. Candidate internal denominator values were not
captured, so this evidence does not attribute the difference to sqrt versus
division individually. The formula and BF16 RNE boundary are unchanged.

PM5750896563 permits explicit classification of CPU FP32 subnormal outputs
that the device flushes to signed zero, without calling that a CPU correctness
pass. True numeric zeros are counted separately. The first wider chunk slice,
`nearzero_c2_g4_rbf16_vbf16_flags010`, chunk1/head1, has CPU maximum
1.14804698e-38, below FP32 minimum normal. Candidate and old NPU are both
bitwise zero;955 CPU values remain nonzero when rounded to BF16, with up to125
ULP from zero and raw L2=1. Thus it is not the all-rounded-zero class.

Applying the classification to every failed slice also exposes **unqualified
normal and nonzero-subnormal failures**:

- Eight of the16 failing chunk cases contain CPU-normal elements whose subset
  error exceeds0.05. For `nearzero_c3_g8_rbf16_vbf16_flags011`, chunk1/head2,
  the28 normal golden elements have relative-L2=0.9754624318055064; the entire
  slice is0.9872856378563912. Candidate and old NPU are bitwise equal in this
  slice. This is an inherited numerical failure, not CPU accuracy acceptance.
- For `nearzero_c2_g8_rbf16_vbf16_flags010`, chunk1/head1,480 normal elements
  have relative-L2=0.18840924222990063. There are7376 subnormal→zero elements
  and33 CPU-subnormal elements with nonzero candidate values. Candidate and
  old NPU are not bitwise equal for the entire slice. Across the differing chunk
  failure slices, the largest candidate/old difference is1 BF16 ULP; maximum
  relative-L2 is0.0002745665475.
- All44 decode output-failure cases have results outside the zero-only flush
  class. Example `nearzero_c2_g2_rbf16_vf32_flags010`, chunk1/head1:613 CPU
  subnormals become zero,6866 remain nonzero (1181 numerically equal), and528
  CPU zeros become nonzero. Relative-L2=0.29826944892132695, max absolute
  error1.6815581571897805e-44. Candidate and old NPU are bitwise equal in this
  example; other slices differ by up to4 FP32 ULP. Their maximum
  candidate/old slice relative-L2 is0.001312404898. These do not erase
  the larger CPU-reference errors.

Counts above are per run and per identified oracle/slice; two oracles are not
distinct hardware samples. All four/six block dimensions reproduce these
classifications. No accepted gate, domain, stable kernel or tolerance is
changed. The first flushing instruction in the composed path remains unlocated.
See `native/saved-endpoint-classification-v2`, the raw nearzero shards, and
[the complete-classification RISK](https://github.com/ddddwee1/ascend_fla_dev/issues/106#issuecomment-5750969353).

## Metric implementation verification

CPU FP32 goldens are retained, while relative-L2 accumulation uses FP64 without
a denominator clamp. FP32 norm underflow previously made some chunk slice
errors read0; the old decode clamp at1e-30 similarly hid extremely small
slice errors. Original and corrected values are both retained. Ordinary
acceptance requires **both original and corrected checks**, preventing this
metric correction from converting an ordinary failure into a pass.

Fresh native runs repeat1620 chunk and3240 decode ordinary cases. All pass;
all returned and preparation bytes match the original runs. Among30780 chunk
metric comparisons, the maximum absolute change is1.0916166891857676e-8;
25455 match at six significant digits and30730 at six decimal places. These
are small differences, not universal digit equality. All61560 ordinary decode
metric comparisons are unchanged. The now-visible nearzero failures above
remain failed. See `native/metric-implementation-proof.json` and `native/metrics-v1`.

## Synchronized performance: slower than the predecessor

`native/perf-v1-bd4` records three same-device predecessor/candidate/predecessor
rounds, with synchronization, one warmup per phase and raw preparation included.
The selected medians are end-to-end wall times; compilation is excluded.

| T | Public path | Predecessor ms | Candidate ms | Candidate / predecessor |
|---:|---|---:|---:|---:|
| 1024 | plain | 6.054806 | 6.792983 | 1.121916 |
| 1024 | cached | 11.158003 | 11.889210 | 1.065532 |
| 4096 | plain | 24.696192 | 27.605673 | 1.117811 |
| 4096 | cached | 45.938957 | 48.759425 | 1.061396 |

All flags add four custom preparation launches: plain11 to15, cached19 to23.
The clean measurements show approximately12% slower plain forward and6% slower
cached forward. There is no speed acceptance threshold, and no speedup over
the predecessor is claimed.

Actual Torch NPU vectorized baselines take9.835303/30.416977ms at
T1024/T4096, versus candidate6.841035/27.622619ms in those corresponding
sandwiches. Actual Torch NPU recurrent baselines take122.81136/492.48823ms,
versus candidate7.380494/27.878302ms. Every measured output passes its comparison
and inputs remain unchanged. Separate instrumented device-event and host-dispatch
records are diagnostic attribution; they are not pure kernel time and are not
added together to explain clean wall time.

## Lossless text evidence and restoration

`evidence_archive.py` packs each completed grid run into bounded JSONL text
shards. Each line contains a complete original receipt, including its own raw
environment; no numerical, dispatch or provenance fields are dropped. The
adjacent manifest maps original filenames to shard/line, byte length and SHA256.
Verification reconstructs the original pretty-printed JSON bytes and checks
those hashes. Fresh restoration additionally compares all restored files to
all originals. No archived source is executed by this utility.

```bash
python evidence_archive.py verify evidence/native/grid-v2/grid-v2-chunk-bd1 --restore tmp/restored-grid
```

A completed evidence-only round trip does not substitute for the still-pending
final delivery archive and source restoration, or any remaining hardware gate.
