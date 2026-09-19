# BF-01 GDN forward: native input/output types and grouped indexing

Pre-implementation contract. BF-01 ASSIGN5743791210; base f8f9eb89.
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
The inherited explicit CPU board/aclnn value-validation boundary awaits PM
clarification; no exception or compatibility change is inferred.

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

## Validation still required

Fresh original-FP32 baseline, full-workload-first native candidate, complete
ratio/chunk grid and both block dimensions, NaN poisoning, actual returned
outputs against A/B, input immutability, both dtype host-operator audits and
same-card three-round BF16/oldFP32 timing sandwiches. No device result is yet
claimed for this new unit. Canonical board/aclnn, CUDA/Triton and weights are
not claimed as executed.
