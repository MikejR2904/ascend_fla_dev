# BF-01 GDN forward: native input/output types and grouped indexing

BF-01 ASSIGN5743791210; base f8f9eb89. The comparison contract below was
frozen before kernel implementation.
D-PM-37 requires the derived unit to serve both FP32 and BF16. The old FP32
unit is an unchanged actual-output baseline. No backward/decode/optimization.

## Frozen ABI and arithmetic

Contiguous token-major q/k[B,T,H,128], v/o[B,T,HV,128], g/beta[B,T,HV];
B/H/HV positive, HV%H=0, T a multiple64 <=4096, a5/cce block_dim1/2.
q/k/v share FP32 or BF16 storage; o matches them; gates/state remain FP32.
Zero initial state only, scale128**-.5, raw q/k with no normalization.
Finite values, g<=0 and beta in[0,1] are NPU caller preconditions.
Consecutive groups share q/k head value_head//(HV/H).

The five-stage Vector plan retains the FP32 score/WY/scan/output operation
order, widens BF16 loads inside prepare, initializes state inside scan and
narrows final output inside output. All stage edges/state remain FP32.
No host conversion, group copying, extra conversion launch or Cube claim.
Public NPU host work is allocations, metadata checks and launch only.
PM provisionally allowed the inherited explicit CPU board/aclnn diagnostic
input-value checks in issue #100 comment 5743858792, pending user confirmation.
These checks remain unchanged; the actual NPU paths must pass the strict audit.

## Calibration before kernel implementation

A is the literal SHA-checked FLA e52dbc0e naive recurrent rule, CPU FP32 on
BF16-rounded inputs (or FP32 inputs in the FP32 control). Its test-only grouped
adapter uses consecutive repeat_interleave. B independently selects heads and
solves the chunk triangular system in CPU FP32; it never imports the kernel.

Completed 28 cases x2input dtypes =56 records before creating kernel source.
Includes ratios1/2/4/8 xC1/2/3/64, multibatch, uneven3heads, gate0/.03/30/1000,
beta0/1, zero-qk and spikes. A/B maxrelativeL2: o2.03599722099332e-6,
final_state1.7792152098930612e-6 (both below the frozen1e-4 calibration gate).
Zero/sign/1.25x wrong-output controls were rejected for nonzero references.

For each case/output/oracle R, define F_R=relL2(BF16(R).FP32,R). This is a
hypothetical storage-rounding floor also for FP32 final_state; the state itself
is never narrowed. Nonzero BF16-input floors across A/B: o
[0.0016268728623421865,0.001675833931353194], state
[0.0016333126533481393,0.001673150026218268].

Budget is fixed now: BF16 public outputs, checked in FP32, each satisfy
relL2<=min(1e-2,3*F_R) against A and B separately. Zero reference/F means exact
zero/exact result, with no epsilon budget inflation. FP32 outputs and all core
FP32 stage arrays use1e-4, and FP32 old/new actual public outputs must be
byte-identical. No later tolerance increases. Report max_abs with every metric.

Raw [calibration](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/calibration-pre-kernel.json),
[log](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/calibration-pre-kernel.log),
and [summary](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/calibration-summary.json).
These are CPU calibration results, not native BF16 acceptance.

## Native qualification

Ascend950PR_9589V100, CANN9.2.0, built-in package ascend950; native Python3.12.13,
Torch2.12.0+cpu, TorchNPU2.12.0, Ascriptor0.1.0 at90cfcdc720bb. Both builds start
with the complete B1/T4096/H=HV8 workload.28cases x2dtypes xbd1/2 =112records.
Every composition and independent leaf is NaN-poisoned; all13stage arrays are
checked in FP32, and actual public returns are separately checked against A/B.

| Input/output storage | Oracle | max o relative L2 | max FP32 state relative L2 |
| --- | --- | ---: | ---: |
| bfloat16 | A | 0.00167582162223 | 1.01935628127e-06 |
| bfloat16 | B | 0.00167581840789 | 2.27452031291e-06 |
| float32 | A | 1.47586613923e-06 | 1.03535485716e-06 |
| float32 | B | 2.17497187351e-06 | 2.27315854415e-06 |

All fixed per-case budgets passed.728stage-array pairs are byte-identical across
bd1/2.112actual-public o/state pairs also match acrossbd; those arrays already
occur among the stage outputs and are not counted again as distinct stages.
112FP32 old/new public-array pairs match the freshly executed original baseline.
Every input hash is unchanged. The actual public-call audit records only
aten.empty.memory_format:13percall,1456total, zero forbidden operators.

This qualifies the recorded generated inputs: q/k/v normal draws scaled0.05,
normalized-key/beta1 cases, gate0/.03/30/constant-1000, zero and spike cases.
As in GDA-02, the numerical budget does not qualify arbitrary finite magnitudes.
There is no new public magnitude gate or ABI expansion. Nonpositive causal
exponents and the same FP32 accumulation order are retained.

Raw [bd1](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/grid-bd1.json),
[bd2](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/grid-bd2.json) and
[recomputed summary](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/native-grid-summary.json)
include every metric, max_abs, fixed budget, input/output hash and host operator.
Each report has an unfiltered log, source manifest, environment and occupancy receipt.

## Storage, synchronization and compiler warnings

Five Vector launches, seven typed entries. Actual lowered UB address endpoints
are164/212KiB for FP32/BF16 prepare,160KiB scores,176KiB WY,224KiB scan,
208/192KiB for FP32/BF16 output. Each buffer has one slot. Mutex ownership retires
MTE3 readers before reuse, orders MTE2 loads before VF consumption, and publishes
scan's VF-initialized state before its first MTE3 store. Explicit VST/VLD barriers
remain inside VF helpers. GM stage edges have unique producers and launch-ordered
consumers. There is no Cube arithmetic or overlap-performance claim.

All7static checks have zero errors/warnings and lowered event balance is empty.
All14selected vendor builds completed. Vendor build warnings remain in the raw
logs:24vector-loop condition type sites, CMake unused cross-platform parameter,
GE header deprecations and a Python outer-template escape warning.

For the vector warning, count=min(64,T-64*cc)=64 throughout the frozen ABI;
i+1 lies1..64, steps are positive1/2, and no uint16_t wrap or negative bound is
reachable. The four shared VF helper ASTs match the old FP32 helpers. Every bound
value/row is covered by bothbd actual poisoned independent-leaf checks; all24
compiled header hashes match the locally emitted headers. This is a scoped
resolution for complete64-token chunks, not permission for arbitrary count/tails.
The vendor Python warning was located by fresh source tokenization: line194 is
inside a non-raw outer template string spanning154-211. No warning was suppressed.
Full evidence is in [compiler analysis](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/compiler-warning-analysis.json).

## Same-card measurement

Complete public APIs, B1/H=HV8/K=V128/block_dim2, identical BF16-rounded values:
original baseline stored FP32, candidate stored BF16. Three baseline/candidate/
baseline rounds per length,10warmups and50samples per segment, synchronize before
and after every call.900raw samples. Independent B checks before/after; candidate
o bytes equal the rounded baseline and state bytes equal the original FP32 state.
All input hashes remain unchanged. Both source trees were verified before launch.

| T | Original FP32 segment median range (ms) | BF16 candidate median range (ms) |
| ---: | ---: | ---: |
| 1024 | 76.639338–76.841495 | 76.193974–76.286289 |
| 4096 | 302.3228475–302.713015 | 301.1417625–301.3799925 |

No speed threshold; these Vector paths have similar measured latency. The baseline
is the unchanged original complete FP32 API, not CUDA/Triton or a BF16 Cube path.
[Raw samples](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/sandwich-bd2.json)
and [recomputed rounds](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/timing-summary.json).

## Host validation and runtime closeout

Dedicated tests38passed, unchanged legacy forward tests55passed; full explicit
CPU selection704passed/4skipped. All56canonical reference cases passed. The first
unfiltered host-suite attempt inadvertently selected KDA NPU tests and is retained
as4failed/275passed/4skipped interrupted, excluded from acceptance. Corrected CPU
selection excludes native-only files; no failed CPU case was removed.

Fresh task-free Torch NPU startup control passed exact transfer/multiply/add checks
and reproduced both observed startup notices: plugin-policy query fallback and
TensorFlow preload failure. Scheduler initialization succeeded. All20runtime log
files (1200lines) are retained:10known startup warnings,0error/fatal/critical. The
exact TensorFlow missing-library cause is not identified; no TensorFlow backend
support is claimed. [Runtime review](../../kernels/projects/a5/gdn_chunk_fwd_bf16/evidence/runtime/runtime-review.json)
includes the scoped disposition and actual control source/hash.

All209Python files in the native snapshot still match delivered source bytes; later
changes only document qualification and add evidence. Canonical
board/aclnn execution, functional/pipe simulation, CUDA/Triton and weights were
not executed for this candidate. Models were not used to qualify vendor lowering.
