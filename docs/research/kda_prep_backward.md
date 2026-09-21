# KDA raw preparation backward — BF-08

BF-08 implements custom backward kernels for raw KDA preparation. Final-source
full Kimi training passes bd1/2/3/4 with 19 matching output/gradient/cache hashes;
12 reduced model checks pass. Native performance is measured and slower.
All required grids and native regressions have executed on the final production
source. Native flushes and the PM-designated unqualified gate observations
remain explicit disclosures, subject to the user’s final merge disposition. BF-07 forward results do not qualify BF-08.

The first commit froze comparison criteria and calibration before any candidate
kernel implementation; historical failures below remain part of the evidence.

The target is the training raw-flag path of `chunk_kda`: normalization of raw q/k,
raw gate transformation, and raw beta sigmoid must have custom-kernel backward
implementations. Raw gradients retain their input dtype; disabled flags preserve
the original inputs and gradient paths. Gate parameter gradients require a
fixed, deterministic two-stage GM reduction, without atomics. Stable KDA kernels,
layout kernels, gate spans155/105 and the existing end-to-end budgets stay unchanged.

## Derivatives and comparison authority

With S=sum(x*x)+1e-6, normalization has dx=gy/sqrt(S) -
x*sum(x*gy)/(S*sqrt(S)). Incoming gy is BF16 at the chunk prepared-q/k boundary.
For u=g+dt_bias and a=-exp(A_log), softplus uses a strict u>20 branch; dg=
gy*a*(1 if u>20 else sigmoid(u)), dA_log=a*sum(gy*softplus(u)) over B,T,K,
and ddt_bias=sum(dg) over B,T. Gate sensitivities are FP32.
Beta has dbeta=gy*s*(1-s), with the predecessor FP32 sigmoid saturation semantics.
The independent FP64 beta reference evaluates exp(-abs(x))/(1+exp(-abs(x)))²
before multiplying gy, avoiding premature rounding to1 before subtraction.

Isolated derivative accuracy uses analytic FP64 math; ULP uses its correctly
rounded raw-output value. Twenty-eight comparisons with independent FP64
Torch autograd have relative L2 below1e-12. End-to-end goldens remain the two
Torch CPU FP32 references, never these FP64 precision studies. The byte-pinned
predecessor in `baseline_backward/` is a comparison object, not the semantic
authority. Baseline source identities and all calibration driver hashes are
embedded in the raw records.

## Frozen budgets

`kernels/projects/a5/kda_prep/backward_budgets.json` is authoritative. Each dtype
combination has its own measured ordinary floor F. L2 is min(0.01,3F), and the
maximum elementwise relative budget is3 times the measured maximum. Reference
zeros must remain zero; no additive absolute tolerance is introduced. These
limits cannot be widened after implementation. Large finite rows and near-null
sensitivities remain subject to ordinary criteria unless an explicitly applicable
owner endpoint rule says otherwise; leaving a case out of floor estimation
never grants an acceptance exemption.

The CPU study uses Python3.11.15/Torch2.10.0+cpu, seeds8008/8009/8010, K128,
B1/B2, small multi-head/multi-chunk shapes and full Kimi B1/T4096/H32. It produces
144 output records across28 budget groups. Immutable records preserve their
original pending-policy labels; the later D-PM-54 disposition below resolves
that ULP-policy question without rewriting historical measurements.

| Output and dtype tuple | L2 floor | L2 limit | Elementwise relative limit | BF16 ULP line |
|---|---:|---:|---:|---|
| norm:bf16:dx | 0.00169599157753 | 0.0050879747326 | 0.0588298001954 | report only |
| norm:f32:dx | 6.27963496771e-08 | 1.88389049031e-07 | 0.122967909905 | report only |
| gate:bf16_bf16_bf16:dg | 0.00165754876397 | 0.00497264629192 | 0.0116731659834 | 1 to both |
| gate:bf16_bf16_bf16:dA_log | 0.001815227609 | 0.00544568282701 | 0.00992912796931 | 1 to both |
| gate:bf16_bf16_bf16:ddt_bias | 0.00168226944614 | 0.00504680833842 | 0.0113807777915 | 1 to both |
| gate:bf16_bf16_f32:dg | 0.00166949587136 | 0.00500848761407 | 0.0116735601324 | 1 to both |
| gate:bf16_bf16_f32:dA_log | 0.00202895524108 | 0.00608686572325 | 0.00917538749397 | 1 to both |
| gate:bf16_bf16_f32:ddt_bias | 1.44368258883e-07 | 4.33104776649e-07 | 0.000533673387147 | report only |
| gate:bf16_f32_bf16:dg | 0.00167133343491 | 0.00501400030472 | 0.0116732475821 | 1 to both |
| gate:bf16_f32_bf16:dA_log | 1.02061283009e-06 | 3.06183849027e-06 | 5.70104550948e-05 | report only |
| gate:bf16_f32_bf16:ddt_bias | 0.00168887966671 | 0.00506663900013 | 0.011655737366 | 1 to both |
| gate:bf16_f32_f32:dg | 0.00166888284579 | 0.00500664853736 | 0.0116734791376 | 1 to both |
| gate:bf16_f32_f32:dA_log | 8.61497054253e-07 | 2.58449116276e-06 | 0.000479829030575 | report only |
| gate:bf16_f32_f32:ddt_bias | 1.40517150986e-07 | 4.21551452958e-07 | 0.00216684542515 | report only |
| gate:f32_bf16_bf16:dg | 5.85020733064e-08 | 1.75506219919e-07 | 1.17668306384e-06 | report only |
| gate:f32_bf16_bf16:dA_log | 0.00377329919304 | 0.01 | 0.0114293124719 | 1 to both |
| gate:f32_bf16_bf16:ddt_bias | 0.00174971939829 | 0.00524915819488 | 0.0116420571085 | 1 to both |
| gate:f32_bf16_f32:dg | 5.88009125358e-08 | 1.76402737607e-07 | 1.33141870714e-06 | report only |
| gate:f32_bf16_f32:dA_log | 0.00173176746605 | 0.00519530239815 | 0.00997526087771 | 1 to both |
| gate:f32_bf16_f32:ddt_bias | 1.58767725332e-07 | 4.76303175996e-07 | 0.0016550567562 | report only |
| gate:f32_f32_bf16:dg | 5.50255726416e-08 | 1.65076717925e-07 | 1.2397318975e-06 | report only |
| gate:f32_f32_bf16:dA_log | 1.04431368609e-06 | 3.13294105827e-06 | 0.000189560092076 | report only |
| gate:f32_f32_bf16:ddt_bias | 0.00167470539833 | 0.00502411619498 | 0.0115644113282 | 1 to both |
| gate:f32_f32_f32:dg | 5.52765130215e-08 | 1.65829539065e-07 | 1.09892957788e-06 | report only |
| gate:f32_f32_f32:dA_log | 5.94061481199e-07 | 1.7821844436e-06 | 0.000119609584947 | report only |
| gate:f32_f32_f32:ddt_bias | 1.4719301474e-07 | 4.4157904422e-07 | 0.00219387346617 | report only |
| beta:bf16:dbeta | 0.00166350212933 | 0.00499050638798 | 0.0116383981361 | 1 to both |
| beta:f32:dbeta | 6.52193081221e-08 | 1.95657924366e-07 | 9.85543024981e-06 | report only |

D-PM-54 ([PM decision](https://github.com/ddddwee1/ascend_fla_dev/issues/117#issuecomment-5754228432))
selects the ULP policy from the ordinary calibration by output and dtype.
BF16 outputs whose predecessor already exceeds1ULP use the frozen relative
metrics; other BF16 outputs retain ≤1ULP to both correctly rounded FP64 and the
actual predecessor, subject to the later pointwise D-PM-56 rule below. FP32 outputs retain the BF-07 rule with no ULP pass line.
Both-reference ULP distributions must be reported. Every BF16 element farther
than1ULP from either comparator must be individually listed with its three-way
bits and sensitivity row or cancellation condition. No3ULP acceptance limit is
introduced. PM will separately disclose D-PM-54 at merge authorization; it is
subject to the user's final disposition.

## Located numerical observations

The predecessor BF16 norm backward has7 elements beyond1ULP out of16,777,216
on the seed8008 full Kimi population; maximum distance3, relative L2
0.0016560061688898119. Three locations have disjoint dual1ULP neighborhoods:
`[0,289,8,96]` (old/rounded bits46025/46022), `[0,1894,7,61]`
(45872/45875), `[0,3675,6,89]` (45988/45985). These are normal FP32 values,
with subtraction condition numbers approximately1.5e5–5.9e5. They are not a
new kernel regression or a D-PM-52 tiny-output endpoint. PM independently
reproduced these measurements. All seven complete input/sensitivity rows and
term values are retained in `evidence/backward/calibration/norm-bf16-ulp-contract-gap.json`.

The additional valid finite near-null sensitivity gy=x is much more sensitive:
BF16 predecessor-to-FP64 relative L2 is10.926449904724377; FP32 raw x with
BF16-rounded gy gives5.5010933221446365e-5. These observations do **not** enlarge
the ordinary frozen budgets or authorize an exception. They remain explicit
validation cases. A candidate exceeding a required budget still fails; the
predecessor's error alone is not acceptance evidence.

Range observations separately retain norm sum-of-squares overflow, exp(A_log)
over/underflow, beta sigmoid saturation, softplus threshold neighbors and zero /
near-zero rows. For example, scale1e20 norm gradients become zero in the old
FP32 graph while the FP64 derivative is nonzero. Existing D-PM-48/50/52 rules
apply only within their written definitions; backward outputs gain no new
endpoint category automatically. Numerical failures and nonfinite masks must
be retained, with CPU FP32 / FP64 / actual old NPU / candidate values where
applicable. No domain gate or tolerance is relaxed.

## Detection and reproduction

All168 zero/negated/scaled1.25 controls fail both their own3F line and the final
group L2 limits. Another32 structural controls test omitted normalization projection /
r³ / dot-product element, omitted gate sigmoid / final32 BT rows of reduction,
and omitted beta (1-s). There is one rounded, just-above-budget perturbation
per group (28 additional controls, separate from the32 structural faults); these perturbations change the wrong output only, never
the limit. Their measured L2/limit range is 1.00263–1.6384.
The BF16 norm missing-one-dot-product-element structural error is 1.5652 times
the frozen L2 limit. All228 controls are rejected. These are comparator
validation results, not functional qualification of a candidate.

From the repository root, with the accepted CPU environment selected:

```bash
python kernels/projects/a5/kda_prep/ref/backward_calibrate.py --section norm --output tmp/bf08-calibration-norm
python kernels/projects/a5/kda_prep/ref/backward_calibrate.py --section gate --output tmp/bf08-calibration-gate
python kernels/projects/a5/kda_prep/ref/backward_calibrate.py --section beta --output tmp/bf08-calibration-beta
python kernels/projects/a5/kda_prep/ref/backward_controls.py --output tmp/bf08-controls.json
python kernels/projects/a5/kda_prep/ref/check_backward_freeze.py
```

The final command uses only the standard library to recompute every frozen
floor/limit/classification from the committed raw JSON, verify hashes and check
all negative-control decisions. Raw calibration, located proof and controls are
small text artifacts. Native acceptance, host operator audit, cross-bd hashes,
model diagnostics and synchronized performance measurements are still pending.


## Candidate implementation and current validation

The candidate uses per-operation autograd Functions only on enabled raw flags.
It retains `chunk._prepare_inputs` for predecessor comparisons. The chunk compile
chain installs all new backward vendors before the first custom launch, including
when inference precedes a later training call. Existing D-PM-42 host exceptions
remain outside this preparation change.

Norm uses the forward kernel's two64-lane K128 sum order. Gate stage1 owns fixed
(head,32 BT-row) work items, forms raw dg plus FP32 parameter contributions, and
writes a fixed binary-tree reduction into GM. Stage2 merges these partials in
ascending order with compensated FP32 summation; head groups give each32-byte
A_log output block one owner. There are no atomics or bd-dependent reduction
orders. Beta saves the raw input and actual FP32 sigmoid. D-PM-55 selects the analytic
derivative for nonsaturated values and zero for saved sigmoid exactly0 or1.
Raw outputs round only at their declared BF16/FP32 gradient boundary.

Sixteen new dtype-specialized entries emit CCE successfully. Host verification
covers1136 cases across the full suite and a focused optional-oracle follow-up;
only5 NPU-only modules remain skipped on the CPU host. The19-case base-entry
negative control produces the expected11 failures and preserves8 no-grad passes.
The three approved test function bodies preserve all decorators/signatures and
other AST nodes. Detailed counts and the16 source-emission receipts are under
`evidence/backward/host/`. These results do not establish native acceptance.

All50 vendors (including all9 inherited backward entries) compiled for bd1..4.
The first full Kimi BF16 training run at bd4 completed but failed isolated norm
elementwise comparison: q0.0773618 and k2.1657375 exceed the frozen0.0588298
limit. Both norm relative-L2 results pass. Gate, beta, parameter-gradient checks,
all six end-to-end gradients against both CPU FP32 references, all2048
head/chunk output checks per reference, input immutability, nine finite caches,
and exact plain/cached outputs pass. The actual training audit records no
unexpected or old host preparation operations. Raw results are retained under
`evidence/backward/native-v1/`; this numerical failure blocks native acceptance.
Grid, boundary, host audit, exact cross-bd equality, reduced model diagnostics
and synchronized performance acceptance remain pending.


## Compensated norm findings

The unscaled compensated candidate subsequently passes the original full Kimi
training workload at bd1,2,3,4. All19 returned output/gradient/cache hashes match
bitwise across the four independent processes. Its primary row sum retains the
forward two64-cadd order; a fixed compensated tree supplies a low component.
The numerator preserves product residuals and the epsilon term through cancellation.
No frozen budget changes. Raw full receipts, the executed source and cross-bd
proof are in `evidence/backward/compensated-v1/`.

The original candidate's full workload is also a useful predecessor finding:
on exactly those returned sensitivities, old CPU norm dq/dk have maximum relative
errors0.046371962552015374 and2.165737451646475 against FP64, with10 and14
elements beyond one BF16 ULP. The old k result itself exceeds the frozen0.0588298
limit; the compensated candidate passes that stricter limit. This observation
does not revise the frozen comparison.

In the near-null B1/T64/H2 case, compensated BF16 results all16384 equal FP64
correct rounding. Candidate/old CPU/old NPU relative-L2 are0.0016607078/12.6931546/
10.8792216. FP32 raw also passes the original limits. Native-generated inputs
and the old CPU graph use Torch2.12; the initial calibration used Torch2.10, so
these numbers are a new observation, not a bytewise replay or recalibration.

The unscaled candidate fails scales1e18/1e20 with intermediate NaNs. A subsequent
range candidate uses device-local powers of two and the algebraically equivalent
scaled derivative, keeping the input/output ABI and fixed reduction order. Its
complete qualification, same-input forward observations, and performance remain
pending. Historical numerical failures and metadata clarifications remain explicit.


## Range repair and newly located grid failures

The source at `297c80baaf6e055b924579f1d6909cd698cefaf2` completed the full
Kimi training workload at bd1/2/3/4. All50 vendors, including9 existing backward
entries, preceded the first custom launch in each process. All19 returned
output/cache/gradient hashes agree across the four processes. The22 norm native
cases pass the original frozen gradient criteria, and12 reduced sim/pipesim
configurations pass their numerical and synchronization checks. Raw receipts,
executed source and the manifest are in
`kernels/projects/a5/kda_prep/evidence/backward/range-and-grid-v1/`.

These results do not qualify the complete task. At scale1e20, the unchanged
BF07 forward and both predecessor paths return finite zero normalized values,
while FP64 forward is nonzero. The repaired analytic gradient is finite/nonzero
and close to FP64; predecessor gradients are zero. This forward/backward
endpoint consistency question is disclosed to PM in issue117 comment5754878405.
No endpoint exemption, forward change or new tolerance is assumed.

All four complete training grids at the earlier source each retained54 failed cases outof1620.
The failures are in gate parameter cancellation (including one BF16 bias
position at2ULP to both references) and FP32 beta's frozen relative-L2 limit.
The latter is bitwise the correct product of the actual saved FP32 sigmoid;
the same oldCPU gradient and alternate multiplication associations also exceed
the frozen analytic limit on the located population. Comment5754936474 requests
the owning derivative-semantics decision before changing this boundary.
Numerical budgets remain byte-identical to the initial freeze. Additional gate
precision work is an unqualified candidate until native workload and grid reruns.


## D-PM-55 and current compensated candidate

[PM decision D-PM-55](https://github.com/ddddwee1/ascend_fla_dev/issues/117#issuecomment-5755012001)
resolves the two preceding semantic questions. Norm backward follows the unchanged
forward: a square-sum overflow in the same two64-cadd FP32 order gives a zero
gradient. Binary scaling protects intermediate products only. Native norm checks
retain all FP64 discrepancies and require the candidate, CPU FP32 and old NPU
forward/gradient endpoint observations; ordinary rows retain frozen limits.

Beta stores the original raw tensor object and the actual forward sigmoid, with
no host copy or cast. Its nonsaturated derivative uses compensated arithmetic
for gy*exp(-abs(beta))/(1+exp(-abs(beta)))²; only the saved sigmoid being exactly
zero or one selects a zero derivative. On the located failed population, the old
CPU graph itself has relative-L2 2.41418e-7, above the frozen 1.95658e-7 limit.
That predecessor observation does not change the criterion. D-PM-54 and D-PM-55
must both be disclosed when the user authorizes merge and remain subject to the
user's final decision.

Gate uses two FP32 components for softplus/sigmoid contributions and both levels
of reduction. Each (head,32 BT-row) partial contains 512 FP32 elements: A high/low
and bias high/low, each with128 channels. Fixed tree/ascending merge order and
output ownership stay independent of block_dim. The exp(A_log) factor is applied
after each parameter reduction. Polynomial coefficients are immediate operands;
the initial vector-coefficient version exceeded the CANN VF stack limit and did
not reach device execution. That failure remains retained; no limit was raised.

The new candidate passes full Kimi training independently at bd1/2/3/4 with all50
vendors ready first, including9 existing backward entries. The seven previously
located failures now pass all9 gradient selections. Native 22-case norm checks
and12 reduced sim/pipesim diagnostics pass. Public explicit prepare and an actual
inference-then-training process both pass without manual precompilation, as does
the additional T1024 full training check against both CPU FP32 references. These
are completed stages; whole-task qualification remains open until the final
1620-case grid at every bd, endpoint investigation, performance and evidence
closeout complete. Raw receipts and their executed production hashes are published under
`evidence/backward/compensated-gate-v1/`; this is an intermediate candidate,
not final task qualification.


The additional gate/beta range run retains eight failed populations: gate A_log
88/89/100 in both raw dtypes, and both beta saturation populations. At beta16
with BF16 sensitivity -1.3141944408416748, the actual saved sigmoid is
0.9999998807907104 (not saturated). Candidate/rounded-FP64 BF16 bits are46111;
old CPU and old NPU bits are46120. Their one-ULP neighborhoods are disjoint.
This is an ordinary dual-reference feasibility gap under the frozen criterion,
not a new permitted endpoint. At beta-88, the actual saved sigmoid and old NPU
are zero while CPU sigmoid/gradient are subnormal; that existing native
underflow observation is retained separately. Gate compensated products at
large A_log also introduce NaNs in places where predecessor gradients are
infinite; those are retained failures requiring investigation. No criteria have
been changed to make these runs pass.


## Final source measurements and D-PM-56

[PM decision D-PM-56](https://github.com/ddddwee1/ascend_fla_dev/issues/117#issuecomment-5755378864)
interprets retained BF16 dual-ULP criteria pointwise. Correctly rounded FP64
remains within one ULP at every ordinary element. The old-host line applies
where that host itself is within one ULP of FP64; other locations are counted
and listed with all three bit patterns and sensitivity/condition data. Norm
BF16 remains D-PM-54 report-only. Frozen budget bytes remain unchanged.
Negative controls reject a candidate two ULP from FP64, disagreement by two ULP
where the host line still applies, and a normal value mislabeled as native flush.

Compensated add/multiply now preserve a primary overflowing FP32 infinity with
a zero low component, avoiding an artificial Inf-minus-Inf residual NaN. Ordinary
arithmetic, loop structure, event credits, buffer allocations and gradient ABI
remain unchanged. Final full-v7 native execution passes all four block dimensions;
all 19 result hashes agree. Full source/environment/build/audit/numerical receipts,
12 model results and raw performance samples are losslessly stored under
`evidence/backward/final-native-v1/original-receipts/`. To restore every original
JSON and verify its original-byte SHA (including its own environment):

```bash
python kernels/projects/a5/kda_prep/evidence_archive.py verify kernels/projects/a5/kda_prep/evidence/backward/final-native-v1/original-receipts --restore tmp/bf08-final-native-restored
```

Boundary-v3 retains two non-passing beta populations solely for native flush:
one element per dtype at beta=-88 has CPU FP32 subnormal gradient and candidate
zero. All ordinary elements pass the frozen criteria; the classification is
valid under D-PM-56(2), **not a CPU correctness pass**. Candidate +0 and old NPU
-0 have zero numeric ULP distance but different bytes. No bytewise-equality or
CPU-pass claim is made. Both public beta paths accept -88,16,20 in both dtypes
with finite output/state/gradients. BF16 beta16 now passes the authoritative
FP64 line and discloses its nine-ULP distance from the old graph.

Gate A_log89/100 no longer has candidate-only NaNs. A_log88 still has positions
where candidate and FP64 are finite while CPU FP32 is infinite:
2 BF16 and13 FP32 dg positions, also present in corresponding dt_bias gradients.
For each output, old NPU is infinite at the2 BF16 and5 FP32 positions;
at the other8 FP32 positions it is finite with a rounding difference from candidate.
These do not fit the literal three equality labels in D-PM-48(1); they remain
unqualified pending owner interpretation, rather than inventing another endpoint
class or forcing an old arithmetic order. `located-endpoints.json` publishes all
30 output positions and four located beta rows; the archive has every point.
Measured public default-gate checks reject A_log80/88/89/100 in both dtypes;
-120/-100 are reachable. Rejection does not qualify the isolated leaf.
D-PM-54/55/56 must all be separately disclosed at user merge authorization.

Full training performance uses one card, bd4, all raw flags and eight gradients,
three synchronized baseline/candidate/baseline rounds, with old-host and Torch
NPU baselines separately. Raw samples, clean whole-stream device events, host
wall time and separate launch-attribution measurements are retained. There is
no speed acceptance threshold. Candidate medians are about106.8ms at T1024 and
425.7ms at T4096:

| Tokens | Baseline | Baseline median ms | Candidate median ms | Candidate / baseline |
|---|---|---:|---:|---:|
| 1024 | Previous host preparation graph | 9.574992 | 106.751083 | 11.148947 |
| 1024 | Torch NPU training | 39.937813 | 106.819345 | 2.674642 |
| 4096 | Previous host preparation graph | 37.599414 | 425.724631 | 11.322640 |
| 4096 | Torch NPU training | 140.536072 | 425.778071 | 3.029671 |

The final host suite passes1139 tests with5 NPU-only skips. Base-entry negative
controls still produce11 expected failures and8 no-grad passes. These host
checks and measured performance do not resolve the remaining endpoint scope.


## Final grid, native disclosures and private restoration

The final production completes 1,620 cases × nine gradient selections at each
of bd1/2/3/4: 58,320 public training calls, including parameter-only gradients.
All returned outputs, gradients and nine cache hashes agree bitwise across bd.
The four original summary files have the identical SHA256
`219b24724ddfb1170c5afac67eb2e81894e57d386814abdf08eaef31eb5c59c6`.
`evidence/backward/final-grid-v1` publishes a bounded representation of every
returned hash,7992 original numerical records,123 distinct original host audit
tables and all invocation mappings. Six full worst-case receipts are retained.
The public verifier reconstructs the original summary bytes and recomputes all
28 frozen-budget maxima/ratios and the complete host operator union:

```bash
python kernels/projects/a5/kda_prep/verify_backward_review.py kernels/projects/a5/kda_prep/evidence/backward/final-grid-v1
```

Final norm22, explicit prepare, inference then training, T1024 and original
backward/gate regressions have all been rerun on the final production. Their
summary/hash records are under `evidence/backward/final-closeout-v1`. All32,775
individual BF16 norm discrepancy rows were compared against the previously
published detail: actual/reference/input hashes and each recorded point agree.
This links identical detail values after the new execution; it does not use
historical execution to qualify new source.

The final four-bd endpoint run measures identical four-column values and bits.
Per bd, the D-PM-56 native-flush count is482:239 BF16 gate gradients and241 FP32
gate gradients at A_log=-100, plus one beta=-88 gradient in each dtype. These
are **not CPU correctness passes**. CPU-zero/candidate-zero points are counted
separately. The signed-zero observation is retained, and D-PM-52 is not applied.
Gate ordinary finite subsets, including A_log80 and the finite portions of88,
meet the original frozen analytic criteria. Candidate-only NaN count is zero.

[PM disposition](https://github.com/ddddwee1/ascend_fla_dev/issues/117#issuecomment-5755620410)
leaves the30 A_log88 output records per bd as **unqualified observations**:
14 have both old paths infinite;16 have CPU infinite and old NPU finite with a
rounding difference from candidate. They are neither passes nor ordinary-budget
failures. The four columns, FP64 distances and default public gate rejection are
retained. No fourth endpoint class is invented. Full unchecked public execution
at these extreme gate values is not claimed qualified. PM will present this
section, D-PM-54/55/56 and the measured slowdown together for user merge approval.

The complete private archive `bf08-final-receipts.private.tar` is75,786,240 bytes,
SHA256 `30dca16aa8b3de77ebcb7f4e8b42071435bd23d8993917c8547fe22374bc4b01`.
It contains all6560 original grid JSON and910 other final validation JSON,
including per-receipt environments. Both component archives were freshly
restored with every original-byte hash checked; all282 bundled members were
then checked byte-for-byte against those verified components. Source identity,
archive manifests, restoration results and public recomputation results are in
`final-closeout-v1/archive-restoration.json`. The complete archive stays private;
machine coordinates are excluded. After extracting it into an ignored scratch
directory, restore its compressed receipt shards with:

```bash
python kernels/projects/a5/kda_prep/backward_evidence_archive.py verify tmp/bf08-private-receipts/grid-v4/grid-v4-bd1 --restore tmp/bf08-restored-bd1
python kernels/projects/a5/kda_prep/backward_evidence_archive.py verify tmp/bf08-private-receipts/validation --restore tmp/bf08-restored-validation
```

Repeat the first command for bd2/3/4 using distinct fresh output directories.
Re-run `summarize_backward_grid.py` on those four restored directories to
recompute the bounded public package from complete originals. Later publication
adds summaries/hashes only, following PM’s evidence-size limit.


## Device kernel time and dispatch attribution

The37.6→425.7ms result measures the **complete forward/backward training step**,
including preparation; it is not preparation alone. Its three-round clean
measurements remain the headline result. A separate NPU Level1 profiler capture
uses one warmup and one active step per path, with explicit `profiler.step()`.
All50 vendors precede the first custom launch, and candidate result hashes match
the clean T4096 run. The first profiling attempt warned that stopping in RECORD
could leave incomplete data; it is retained privately and not used below. The
scheduled rerun removes that warning and the initial level/AIC-metric warning.

| Separate event-instrumented run | Old path | Candidate |
|---|---:|---:|
| Synchronized host wall (ms) | 37.704827 | 425.361666 |
| Whole-stream NPU event (ms) | 37.566223 | 425.249359 |
| Custom host dispatch sum (ms) | 3.518247 | 3.605901 |
| Custom launches | 34.000000 | 43.000000 |

Host dispatch overlaps device execution. Per-launch event windows can include
host idle gaps, so the event sums are not an additive decomposition of clean
wall time. The profiler's device durations exclude task queue wait, but Level1
hardware-counter collection perturbs durations: summed kernel durations are
81.949ms old/476.127ms candidate, rather than the clean37.6/425.7ms medians.
Both sources identify the same dominant gate first stage. The profiler observes
34 old/43 candidate custom launches and812 old/761 candidate total NPU kernels;
unchanged registered host operations remain in that total.

Every candidate custom kernel is listed below, aggregated over identical kernel
names within the one active training step. The public JSON also keeps each
individual invocation's device duration, the source CSV hash and environment.

| Kernel | Calls | Profiled device ms | Separate event window ms | Host dispatch ms |
|---|---:|---:|---:|---:|
| `kda_prep_backward_gate_bf16_f32_f32_kernel` | 1 | 375.466117 | 370.907471 | 0.112889 |
| `kda_prep_backward_norm_bf16_kernel` | 2 | 19.566360 | 17.761840 | 0.151778 |
| `kda_layout_bf16_bf16_kernel` | 12 | 16.360361 | 10.143459 | 0.383406 |
| `inverse_epilogue_kernel` | 1 | 5.993875 | 2.262993 | 0.199139 |
| `finalize_post_stable_kernel` | 1 | 4.932353 | 1.912493 | 0.181552 |
| `inverse_mm_bounded_kernel` | 1 | 4.424511 | 0.717511 | 0.142484 |
| `kda_sub45_aqk_repaired_kernel` | 1 | 3.926023 | 1.290965 | 0.046851 |
| `kda_sub2_score_stable_kernel` | 1 | 3.888422 | 1.285671 | 0.042995 |
| `scan_fused_kernel` | 1 | 3.514776 | 2.000056 | 0.161843 |
| `kda_layout_f32_bf16_kernel` | 3 | 3.397746 | 2.321267 | 0.900762 |
| `finalize_pair_kernel` | 1 | 3.305203 | 0.656380 | 0.138718 |
| `finalize_pre_stable_kernel` | 1 | 3.112445 | 0.997151 | 0.147110 |
| `tril_inverse64_v2_strict_bf16_kernel` | 1 | 3.103281 | 1.467564 | 0.025608 |
| `kda_sub3_wy_stable_kernel` | 1 | 2.761051 | 0.617966 | 0.051698 |
| `kda_prep_chunk_norm_bf16_bf16_kernel` | 2 | 2.537441 | 1.644463 | 0.054463 |
| `kda_layout_bf16_f32_kernel` | 3 | 1.933004 | 0.924175 | 0.225988 |
| `inverse_dainv_kernel` | 1 | 1.838224 | 0.339910 | 0.099789 |
| `kda_prep_chunk_gate_bf16_f32_f32_kernel` | 1 | 1.833099 | 1.078656 | 0.069975 |
| `kda_layout_f32_f32_kernel` | 2 | 1.781218 | 1.170068 | 0.052048 |
| `finalize_reduce_kernel` | 1 | 1.296735 | 0.576625 | 0.080911 |
| `inverse_dakk_fused_kernel` | 1 | 1.267772 | 0.337246 | 0.074011 |
| `kda_sub1_gate_stable_kernel` | 1 | 0.566155 | 0.193082 | 0.033740 |
| `kda_prep_backward_beta_bf16_kernel` | 1 | 0.440268 | 0.429601 | 0.112349 |
| `kda_prep_backward_gate_reduce_f32_f32_kernel` | 1 | 0.151922 | 0.117413 | 0.085679 |
| `kda_prep_chunk_beta_bf16_kernel` | 1 | 0.014990 | 0.010045 | 0.030115 |

The gate first stage spends375.466ms in the profiler and
370.907ms in the separate event window. Its
profiled vector/scalar activity ratios are0.994/
0.001; these are hardware-counter observations,
not an independent speed model. Source inspection explains the likely cost:
for every32-BT-row×128-channel work item, each64-lane half evaluates compensated
exp, log1p/softplus and sigmoid, multiple product residuals and a fixed five-level
partial reduction. Exp evaluates its polynomial and eight compensated squarings;
all high/low components are materialized through the fixed reduction. These are
inferences from the measured dominant stage and source, not isolated per-operation
microbenchmarks. The second global reduction is small in the actual measurements;
it does not account for the several-hundred-millisecond regression.

A later performance task could reduce the compensated transcendental/product
instruction count while proving the same frozen error bounds, reduce repeated
64-lane temporary loads/stores, improve work-item pipelining and tile reuse, or
reduce redundant preparation launches. Selective compensation would require a
proved sensitivity/error bound; no threshold or approximate fallback is approved
here. Any pipeline/tile change must preserve fixed reduction order, cross-bd
bitwise results and full native validation. The16,777,216-element gate first stage
is the first measurement target; merely removing one small final-reduction launch
would not address the dominant cost. **No such optimization is made in BF-08.**

The profiler archive is private. Public `performance-attribution-summary.json`
contains only aggregate kernel measurements and per-kernel times; machine/device
coordinates and raw tracing paths are omitted. `summarize_backward_profile.py`
recomputes it from the archived CSV and clean measurement receipts.
