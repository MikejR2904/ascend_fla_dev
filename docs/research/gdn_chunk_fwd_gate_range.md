# GDN chunk forward: ABI, range and validation contract

This unit implements scalar-gated DeltaNet (`gated_delta_rule`), not GDN-2.
Scope is inference-only A5 CCE, non-GQA, zero initial state. It does not
establish Qwen3-Next compatibility or A2/A3 support.

## Frozen public ABI

Inputs q/k/v are contiguous token-major `[B,T,H,128]`, all BF16 or all FP32.
The activated scalar beta and log-decay g are contiguous FP32 `[B,T,H]`.
B and H are positive, T is a positive multiple of 64 up to 4096. Value heads
must equal key/query heads. The only scale is `128**-0.5`; q/k normalization
is not part of this operator. Input values must be finite, beta in [0,1],
and g<=0. Inputs requiring autograd are rejected.

Only `initial_state=None` is accepted as the zero-state representation;
explicit initial-state tensors, including zero tensors, are rejected. This
keeps the unsupported nonzero-state case explicit without a device-to-host
read in the timed path. Output has the input q dtype. Optional final state
is fresh FP32 `[B,H,128,128]`, K-major. Noncontiguous/head-first inputs,
GQA, tails, unsupported scale, backend or block dimension raise errors.
Initial core scope is block_dim in {1,2}.

The upstream `a5.gdn_fwd` unit is an ABI comparison source, not a runtime
dependency. Its blocked input layout, unscaled query and BF16 state differ
from this new public ABI. Those differences are intentional and observable:
the new unit scales q in FP32 and keeps all intermediate edges/state FP32.
BF16 input conversion to FP32 is exact; only final output is rounded back.
No upstream numerical or hardware receipt certifies this new unit.

## Equations and independent references

For each head, with zero S initially:

`D_t = exp(g_t) S_(t-1)`

`r_t = beta_t (v_t - k_t^T D_t)`

`S_t = D_t + k_t r_t^T`, `o_t = (q_t / sqrt(128))^T S_t`.

The authoritative reference is FLA's CPU FP32
`naive_recurrent_gated_delta_rule`. A separately authored CPU block reference
solves a unit-lower triangular system for r within each 64-token chunk;
it does not call FLA or repeat FLA's row-wise inverse expansion.

Let p be the FP32 chunk prefix sum of g and
`L_ij = beta_i (k_i^T k_j) exp(p_i-p_j)` for j<i.
Then `(I+L) R = beta * (V - exp(p) * K S_in)`.
Outputs follow from inter-chunk q*S and the causal q*k score matrix times R.
The state transition is
`S_out = exp(p_last) S_in + (K * exp(p_last-p))^T R`.

CPU reference preparation covered (T,H)=(64,3),(192,3),(1024,16),(4096,16)
and (128,2), with weak, ordinary and strong negative gates. Both references
computed in FP32 on the same BF16-valued inputs. Maximum output/state
relative L2 was 1.87742e-6, below a predeclared 1e-5 oracle agreement budget.
This is a CPU result, not a kernel execution result.

## Range and precision boundaries

| Expression | Range / constraint |
| --- | --- |
| p=cumsum(g) | Nonpositive; FP32 accumulation error must be measured |
| exp(p_i) | [0,1]; underflow correctly removes old-state influence |
| exp(p_i-p_j), j<=i | [0,1]; acausal entries masked before exponentiation |
| exp(p_last-p_i) | [0,1]; no reciprocal decay or positive exponent |
| Triangular solve | Unit diagonal; no data-dependent diagonal division |
| beta*k and beta*v | beta in [0,1]; operand/product range remains a precondition |
| Dot products and state accumulation | FP32 order/rounding are explicit; finite exponentials alone do not prove accuracy |
| scale | Fixed positive constant 128**-0.5, applied in FP32 |
| final BF16 output cast | Sole lossy storage boundary for BF16 callers |

The generated validation domain uses q/k/v normal draws scaled by
0.05, with additional normalized-key and zero/strong-gate cases.
Device comparison budgets must be fixed before the first candidate run;
initial FP32 output/state relative-L2 budget is 1e-4 and BF16 output budget
is 5e-3, together with elementwise atol=2e-5/rtol=2e-4 for FP32 and
atol=2e-5/rtol=1e-2 for BF16. State always uses the FP32 budget. These
budgets do not declare arbitrary finite inputs numerically qualified.

## Launch and evidence plan

Five ordered stages prepare, form causal scores, solve WY factors, scan
chunk states, and form outputs. GM edges are private to the invocation;
each producer completes before its next launch consumer. Parallel stages
own complete (batch,chunk,head) items; scan owns complete (batch,head)
sequences. Stage implementations must establish DMA/VF buffer retirement
before reusing local storage. No cross-core state reduction is needed.

Full T=1024/4096 workloads run on hardware before any reduced simulator
diagnostics. Cover C=1/2/3 and repeated heads under block_dim 1/2. Compare
all outputs against both CPU references, include zero/perturbation negative
controls, and record cross-block-dimension equality independently.

Profile the baseline before choosing changes. Use synchronized same-device
baseline/candidate/baseline, separate processes for different builds, fixed
warmup/repeat, and exact source/toolchain identities. In-process CCE/aclnn
must be validated separately from the SSH board harness. The following
measurements qualify only the recorded generated inputs and device.

## Original-device baseline and selected optimization

The initial per-token prepare baseline (`fb1e0de`, stage-source SHA256
`1c022aa953e77c37661c4283bb470741c4aa8ece1a3b6ff2641b9ce16c2ddfe8`)
passed native CCE/aclnn T4096/H16/block_dim2. Output max_abs was 2.153684e-9,
relative L2 6.135233e-7; FP32 state max_abs 2.980232e-8, relative L2
9.675854e-7 against the independent block solve. The separate in-process
bridge passed all ten cases against both CPU references and public BF16/FP32
entry checks. The fixed runtime is Python 3.12.13, Torch 2.12.0+cpu with
Torch NPU 2.12.0; the device reports Ascend950PR_9589, compiler/OPP 9.2.0,
OPP kernel directories ascend950, ascend910b and ascend910_93.

The first synchronized T4096 baseline profile (warmup10/repeat50, block_dim2)
measured public median 54969.98 us and prepare median 9901.86 us. Scores,
WY, scan and output were respectively 12768.92, 11244.81, 8524.70 and
5871.34 us. A subsequent grid run measured 46464.14 us public latency;
this observed drift is why acceptance uses paired sandwiches rather than
comparing isolated historical medians.

The selected candidate batches prepare's transfers into complete 64-row
head tiles. At T4096/H16 the source-level DMA invocation count falls from
655360 to 10240, and VF invocations from 65536 to 1024. Requested logical
input/output bytes are unchanged (101187584 / 167772160); these are analytic
requested bytes, not measured HBM traffic or a bandwidth utilization claim.
The same four vector participants are active at block_dim2; no cube pipeline
is introduced. The sequential FP32 prefix and per-row multiply order remain.

Single-slot ownership uses five 64x128 FP32 matrices (160 KiB) and two
64x8 scalar staging matrices (4 KiB). Every MTE2 writer precedes its VF
reader, and all MTE3 readers retire before the next item reuses the slot.
The explicit q/k/v row gap is `(H-1)*128` elements. Two full slots would
require 328 KiB, exceeding the 256 KiB profile, so this candidate deliberately
keeps one item in flight; it makes no overlap or lookahead claim.

### Rejected implicit-stride candidate

An initial candidate used `qu[:,:] <<= q[bb, tt:tt+64, hh, :]`. At the
accepted library revision, `passes/device_lower.py:81` (`gm_transfer`) and
its two-sliced-dimension branch infer a zero row gap from the contiguous
suffix, omitting the indexed-away H axis. The emitted transfer was
`gm_to_ub_pad(...,64,512,0,0)`; its correct byte gap is `(H-1)*512`.
A reduced T64/H3/block_dim1 pipe-model diagnostic failed 24135/24576 qn
values (max_abs 0.0234730), despite finite results. Native T4096/H16 also
failed: 8240529/8388608 qn values, max_abs 0.0334046. This candidate is rejected;
no threshold was relaxed. The original per-token baseline is unaffected.

The unit uses the supported explicit `gm_to_ub_pad` source-gap argument,
without modifying the upstream library. The corrected reduced diagnostic
passes: qn/kn/bk/wv match the CPU reference exactly; prefix max_abs=1.78814e-7
and relative L2=1.17071e-7 versus torch cumsum. Event balance and hazard lists
are empty and no deadlock is reported. This is specifically a repeated-buffer
and strided-copy model check, not silicon qualification. The failure and
located workaround were reported in GDA-01's RISK thread.

## Replacement-device native qualification

After the original-device fault described below, the user authorized a
device change. The following complete grid and six paired comparisons
were rerun on the healthy replacement; no original-device performance
sample is used in this qualification. The explicit-gap candidate has stage SHA256
`a5d0c7a7b7f62001936d64f06538444afc90f4d2100362f9ceacef4142fbd72d`.
`kernels/projects/a5/gdn_chunk_fwd/validation.json` is the retained numerical
receipt, including source identities, generated-input hashes, both oracle
comparisons, independent leaf checks and all paired samples. The environment
is Ascend950PR_9589, CANN compiler/OPP 9.2.0, kernel packages ascend950,
ascend910b and ascend910_93; Python 3.12.13, Torch 2.12.0+cpu, Torch NPU 2.12.0.
The accepted library revision is `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`;
FLA reference revision is `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`.

All ten cases pass at block_dim1 and block_dim2, including repeated heads,
chunks, batches, normalized keys and zero/weak/strong gates. All 13 stage
outputs are byte-identical across both block dimensions and the baseline.
Every independent leaf and the composed graph pass with NaN-poisoned
outputs; public BF16/FP32 calls pass both relative-L2 and explicit elementwise
checks. Shape and input seeds are recorded per case. FP32 elementwise
atol/rtol=2e-5/2e-4 and relative-L2
1e-4 are unchanged; BF16 output uses atol/rtol=2e-5/1e-2 and relative-L2 5e-3.

Across the complete grid, maxima (max_abs / relative-L2; maxima may arise
from different cases) are:

| Comparison | Output | FP32 state |
| --- | --- | --- |
| Independent CPU block solve | 5.84987e-09 / 1.779822e-06 | 3.352761e-08 / 9.675616e-07 |
| FLA CPU recurrence | 2.779416e-09 / 9.826487e-07 | 4.097819e-08 / 7.502357e-07 |
| Public BF16 vs CPU block solve | 7.605646e-06 / 0.001696268 | 3.72529e-08 / 9.675616e-07 |

### Three-round paired performance

Same reserved device, fresh process per sample, block_dim2, B1/H16/K128/V128,
FP32 inputs, warmup10/repeat50. Every timed public call is synchronized;
latency includes host launch and allocation overhead. The conservative
speedup divides the faster of the surrounding baseline medians by the
candidate median. These measurements are not device-only kernel timings.

| T | Round | Baseline before (us) | Candidate (us) | Baseline after (us) | Conservative speedup |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1 | 163980.472 | 152230.666 | 164150.194 | 1.0772x |
| 4096 | 1 | 652253.193 | 603352.131 | 651657.716 | 1.0801x |
| 1024 | 2 | 164143.023 | 152390.867 | 163971.726 | 1.0760x |
| 4096 | 2 | 651921.704 | 603404.083 | 651448.616 | 1.0796x |
| 1024 | 3 | 163218.400 | 152464.804 | 164594.333 | 1.0705x |
| 4096 | 3 | 654349.508 | 603469.789 | 651153.854 | 1.0790x |

All six paired comparisons improve. All 13 checkpoint hashes remain equal
in every baseline/candidate/baseline triplet. T4096 retained-stage workspace
is 369098752 bytes and measured public peak allocation increment is
287309824 bytes, unchanged between candidates. The receipt also retains
per-stage profile medians to distinguish prepare gains from runtime drift.

The native in-process CCE bridge passes on the replacement device. All 45
health checks (before/after each of 22 benchmark processes, plus completion)
return health=0 and no error codes. The full T4096 workload ran first. The
replacement has much higher absolute latency than the original device; the
cause is not established, and neither latency nor speedup is transferred
between devices.
The initial baseline additionally passed the standalone native aclnn harness; the final
revision does not claim a separate SSH board-harness or full-unit simulator
qualification. The prepare-only pipesim regression remains a diagnostic.
GQA, nonzero initial state, arbitrary finite input magnitudes, backward and
A2/A3 remain outside this qualification.

### Retained original-device postflight failure

On the original device, after all six paired comparisons completed, the additional strict-grid
preflight returned DSMI health rc=0, health=2, error_count=1 and code
`0x80f78009`. The driver describes it as "node type=HWTS/Stars-TS, sensor
type=RAS State, event state=bus error, probably caused by software". Repeated
read-only queries returned the same status. No remaining GDN benchmark or
compiler process, or device-node owner, was found. No reset or process kill
was attempted. The onset and cause are not established by the completed
numerical receipts. The original-device NaN-poisoned grid never launched.

Original-device receipts remain in Git history at `40b5d9d`. That device
was not reset and its fault is not declared resolved. User-authorized
replacement-device verification completed the strict grid and paired
benchmarks above, with healthy pre/postflight checks throughout. The final
qualification is limited to the replacement device and recorded workloads.
