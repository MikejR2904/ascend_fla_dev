# GDN chunk forward: ABI, range and validation contract

This unit implements scalar-gated DeltaNet (`gated_delta_rule`), not GDN-2.
Scope is inference-only A5 CCE, grouped heads, zero initial state. It does not
establish Qwen3-Next compatibility or A2/A3 support.

## Frozen public ABI

Inputs q/k are contiguous token-major `[B,T,H,128]`; v is `[B,T,HV,128]`,
all BF16 or all FP32. Activated scalar beta and log-decay g are contiguous
FP32 `[B,T,HV]`.
B and H are positive, T is a positive multiple of 64 up to 4096. Value heads
must be a positive multiple of key/query heads. Value head j uses key/query
head `floor(j/(HV/H))`; the groups are consecutive. The only scale is `128**-0.5`; q/k normalization
is not part of this operator. Input values must be finite, beta in [0,1],
and g<=0. Inputs requiring autograd are rejected.

Only `initial_state=None` is accepted as the zero-state representation;
explicit initial-state tensors, including zero tensors, are rejected. This
keeps the unsupported nonzero-state case explicit without a device-to-host
read in the timed path. Output has the input q dtype. Optional final state
is fresh FP32 `[B,HV,128,128]`, K-major. Noncontiguous/head-first inputs,
invalid head ratios, tails, unsupported scale, backend or block dimension raise errors.
Initial core scope is block_dim in {1,2}.

The upstream `a5.gdn_fwd` unit is an ABI comparison source, not a runtime
dependency. Its blocked input layout, unscaled query and BF16 state differ
from this new public ABI. Those differences are intentional and observable:
the new unit scales q in FP32 and keeps all intermediate edges/state FP32.
BF16 input conversion to FP32 is exact; only final output is rounded back.
No upstream numerical or hardware receipt certifies this new unit.

## GDA-02 implementation and qualification scope

The shared launch graph replicates q/k with device-local `repeat_interleave`
only for HV>H, after exact BF16-to-FP32 conversion. The existing five CCE
kernels receive HV equal-sized heads; their arithmetic, synchronization and
local storage are unchanged. Each value head retains its own g, beta and
state. Replication uses two FP32 buffers of B*T*HV*128 elements each, in
addition to existing stage storage. This depends on the target OPP tensor
replication operator. HV==H returns the original input dictionary and does
not allocate replication buffers. It must remain byte-identical to GDA-01.
This is an ABI extension, with no performance optimization claim.

The pinned FLA GDN naive at e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2
accepts equal heads only. PM clarified this in the GDA-02 assignment:
PGDN naive defines the grouping convention. Grouped CPU correctness is
therefore checked against an independently indexed, head-local recurrence
and a block solve. FLA GDN naive is an explicitly expanded-input comparison,
not a claim that upstream GDN naive natively supports GVA. The public
zero-state-only restriction is unchanged; internal zero state uses HV.

GDA-02 native qualification and same-device regression measurements are
recorded below and in the current unit validation.json.

## Equations and independent references

For each head, with zero S initially:

`D_t = exp(g_t) S_(t-1)`

`r_t = beta_t (v_t - k_t^T D_t)`

`S_t = D_t + k_t r_t^T`, `o_t = (q_t / sqrt(128))^T S_t`.

For equal heads, FLA's CPU FP32 `naive_recurrent_gated_delta_rule`
provides the semantic reference; grouped authority is described above. A separately authored CPU block reference
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

## Inherited GDA-01 preparation and synchronization

The unchanged GDA-01 CCE implementation retains the following ownership
analysis and explicit-stride workaround. Historical baseline and performance
receipts remain at commit `6b6048592bd1fa2a49de203c40346a21b0343f9f`.

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

## GDA-02 native qualification

The same healthy replacement device used for GDA-01 reports Ascend950PR_9589,
CANN compiler/OPP 9.2.0, kernel packages ascend950/ascend910b/ascend910_93.
Python is 3.12.13, Torch 2.12.0+cpu, Torch NPU 2.12.0; library revision is
`90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`. The five-stage source remains
`a5d0c7a7b7f62001936d64f06538444afc90f4d2100362f9ceacef4142fbd72d`.
The current unit `validation.json` retains exact source identities, seeds,
shapes, numerical maxima and paired samples. Baseline is the final GDA-01
revision `6b6048592bd1fa2a49de203c40346a21b0343f9f`, not its older per-token
implementation. Machine details and full raw logs stay in ignored scratch.

Baseline T4096/H16 profile preceded candidate execution. The first candidate
run was the full B1/T4096/H4/HV16/block_dim2 grouped workload. Against the
independent head-local recurrence, output relative L2 was 4.891699e-7 and
state 8.529894e-7. This was followed by all 22 cases at both block dimensions:
ratios 1/2/4/8 crossed with chunks 1/2/3, repeated batches/heads, long T1024
and T4096, plus inherited normalized-key and zero/weak/strong-gate cases.
All 13 composed checkpoints and independently supplied leaf outputs passed
with NaN-poisoned allocations. Checkpoints are byte-identical between
block dimensions; all ten equal-head cases also match GDA-01 exactly. A separate four-process
check additionally confirms public FP32/BF16 output and state bytes are
identical between GDA-01/GDA-02 at both block dimensions.

Public FP32 maximum relative L2 is 1.779822e-6 for output and 1.379258e-6
for state; BF16 output maximum is 0.001696268, with FP32 state using the
same unchanged 1e-4 budget. Elementwise tolerances remain unchanged.
Canonical CPU reference: 22 cases passed. Host tests: 284 passed, 5 skipped
(the five NPU-only KDA modules); no GDN oracle check was skipped.

### Same-device performance regression

FP32 public latency is synchronized and host-inclusive, warmup10/repeat50,
block_dim2, B1/H=HV=16. Three fresh-process baseline/candidate/baseline
rounds preserve every stage hash. No speedup is claimed for this ABI change.

| T | Round | Baseline before (us) | Candidate (us) | Baseline after (us) | Ratio |
| --- | --- | --- | --- | --- | --- |
| 1024 | 1 | 152300.446 | 152065.442 | 152282.791 | 1.001429x |
| 4096 | 1 | 603322.731 | 603376.514 | 603477.619 | 0.999911x |
| 1024 | 2 | 152185.945 | 152556.232 | 152257.815 | 0.997573x |
| 4096 | 2 | 603369.926 | 603403.910 | 603561.410 | 0.999944x |
| 1024 | 3 | 152302.413 | 152242.034 | 152303.066 | 1.000397x |
| 4096 | 3 | 603536.311 | 603409.512 | 603588.433 | 1.000210x |

Grouped B1/T4096/H4/HV16 public median is 604216.626 us including q/k
replication. Its extra FP32 replication buffers occupy 67108864 bytes;
measured public peak allocation increment is 354418688 bytes. This is a
grouped profile, not a grouped speedup comparison. The equal-head path
retains its no-copy behavior. No cross-device performance inference is made.

The baseline pre/postflight and six candidate orchestration phases all
returned health=0 without errors (14 checks). The four public-byte
comparison processes add eight healthy pre/postflight checks (22 total). The sandwich holds the same
exclusive device lock across all 18 child processes; health checks bracket
that whole phase, not each timing child. No full-unit simulator or SSH
board-harness qualification is claimed. Public nonzero state, backward,
decode, arbitrary finite magnitudes and A2/A3 remain outside this task.

The original GDA-01 device fault (health=2, code 0x80f78009, driver-described
HWTS/Stars-TS RAS bus error) remains unresolved and that device was not used
or reset. Its onset/cause were not established; historical evidence is at
`40b5d9d`. This qualification belongs only to the healthy replacement.
