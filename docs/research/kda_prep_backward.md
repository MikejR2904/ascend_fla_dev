# KDA raw preparation backward — BF-08

This first commit freezes the backward comparison criteria before any candidate
kernel implementation. It contains CPU calibration, independent derivative
references and corrupt-output controls. **No BF-08 native or simulator stage has
passed yet.** BF-07 forward qualification does not qualify new backward kernels.

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
actual predecessor. FP32 outputs retain the BF-07 rule with no ULP pass line.
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
orders. Beta differentiates the actual FP32 sigmoid output, preserving saturation.
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

The complete bd1 andbd3 training grids each retained54 failed cases outof1620.
The failures are in gate parameter cancellation (including one BF16 bias
position at2ULP to both references) and FP32 beta's frozen relative-L2 limit.
The latter is bitwise the correct product of the actual saved FP32 sigmoid;
the same oldCPU gradient and alternate multiplication associations also exceed
the frozen analytic limit on the located population. Comment5754936474 requests
the owning derivative-semantics decision before changing this boundary.
Numerical budgets remain byte-identical to the initial freeze. Additional gate
precision work is an unqualified candidate until native workload and grid reruns.
