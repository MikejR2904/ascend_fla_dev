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
