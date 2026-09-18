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

The initial generated validation domain uses q/k/v normal draws scaled by
0.05, with additional normalized-key and zero/strong-gate cases planned.
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
must be validated separately from the SSH board harness. Hardware, latency,
workspace and optimization conclusions are pending; no speedup is claimed.

## Measured baseline and selected optimization

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
