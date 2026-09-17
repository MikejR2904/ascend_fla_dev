# GDN-2 chunk forward

Status: FP32 implementation, CPU references and reduced simulator diagnostics pass;
no hardware or performance qualification yet.
Task: GD2-01 (#62), originating from #61. Only the standalone forward unit and
opt-in sibling API are in scope. Existing model/operator files and PM-owned
board/matrix files remain unchanged in this task branch.

## Semantic contract

Source: FLA `fla/ops/gdn2/{naive,chunk_fwd,chunk_intra,wy_fast}.py`, revision
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`. The independent recurrence is

    D_t = exp(g_t) * S_(t-1)
    r_t = w_t * v_t - (b_t * k_t)^T D_t
    S_t = D_t + k_t r_t^T
    o_t = (scale * q_t)^T S_t

Inputs are token-major; state is K-major FP32. q/k normalization uses FP32,
eps=1e-6 and scale=128^-0.5 at the CCE boundary. Normalized keys and queries
are materialized in FP32. BF16 public inputs are widened before this boundary.
Finite g<=0, b in [0,2], w in [0,1]; no clipping. Public scope: B=1,
H=1 or 16, K=V=128, 1<=T<=4096, continuous storage, no autograd.

## Decomposition

Within each 64-token chunk, let G_i=sum_(t<=i) g_t and E_ij=exp(G_i-G_j).

    L_ij = sum_d (b_i*k_i)_d * k_jd * E_ijd, j<i
    Q_ij = sum_d (scale*q_i)_d * k_jd * E_ijd, j<=i
    (I+L) U = w*v
    (I+L) W = (b*k)*exp(G)
    Vnew = U-W*S_in
    S_out = exp(G_last)*S_in + (k*exp(G_last-G))^T Vnew
    O = (scale*q*exp(G))*S_in + Q*Vnew

Five launches: prepare normalized/padded chunk inputs and G; causal scores;
exact forward substitution for U/W; ordered state scan per head; independent
chunk outputs. Every workspace edge has a unique producer. Chunk buffers are
[B,C,H,64,128], score buffers [B,C,H,64,64], carried states [B,C,H,128,128].
C=ceil(T/64). Padding has zero q/k/b/wv and repeats the final G; it is an
identity update. Workspaces are initialized completely by their producer.
No backward caches survive the public call; no input/output aliasing is allowed.

FP32 Vector arithmetic is the initial correctness baseline. The score phase
uses causal differences directly, avoiding positive exponents even with the
observed span 1461. This also avoids multiplying a masked infinity by zero.
Forward substitution is exact triangular solving, not a truncated inverse.
FLA's `safe_gate=True` selects an anchored matrix path whose range assumptions
must not be inferred from its name. Later Cube work may use 16-token diagonal
blocks with direct differences and boundary-anchored off-diagonal blocks.
Any conversion or changed reduction tree must pass the original error gate.

## Ownership and synchronization

Prepare, score, WY and output assign complete (batch, chunk, head) work items to
vector participants. Scan assigns complete heads and visits chunks in order.
A launch boundary publishes all GM edges. The baseline uses one UB slot per
buffer and `auto_sync`; VF store/load barriers publish local scalar reductions
and solved rows. No mixed Cube/Vector pipeline or multi-buffer overlap is claimed.
Physical rows are 128 FP32 elements (512 bytes); score rows are 64 (256 bytes).
The scan working set must stay below the A5 256 KiB UB profile capacity.

## Validation and remaining gates

Compare the CPU chunk composition and CCE output/state against an independent
FP32 recurrence, including nonzero state, odd chunk counts, tails, strong decay,
and split-prefill/decode. Output and state relative L2 must be <=1e-4; report
maximum absolute error too. Real-model budgets remain 1e-3 FP32 and 1e-2 BF16.

The user requires Torch CPU mode. CPU references and CPU-tensor aclnn/board
harness execution are supported separately from the in-process NPU bridge.
CPU timings are not NPU performance. SSH authentication initially failed, so reduced simulator checks were used only
to diagnose the newly authored stages. The assigned directory and device health
are now confirmed; native full-workload validation is in progress. NPU model
and torch_npu baseline qualification remain untested under the CPU-only request.

## Host observations

Five CCE stages emit successfully at the accepted library pin. After the native
WY workaround, functional sim passes T=1/H=1 and pipesim passes T=2/H=1 at
block_dim=1. Revision `9ba48cd` also passed T=65/H=1, T=2/H=16 and T=65/H=16
pipesim diagnostics; those larger diagnostic claims require renewal after the
source change. CPU tests cover both zero and strong decay through T=4096/H=16;
FLA recurrent is loaded explicitly by file path in the optional oracle test.

The public graph retires each workspace after its last consumer launch. The
standalone stage checker intentionally retains intermediates for comparison.
This reduces retained allocations, without changing arithmetic or claiming a
measured device speedup. No tensor identities or private machine paths are
stored in the tracked contract.

## GD2-01 gate-range survey status

This derivation predates the arithmetic port into the scoped task branch.
The requested real-95B random-token replay and natural-text run are **pending**:
no checkpoint or tokenizer is available in the inspected checkout or assigned
remote user directory. Asset locations have been requested. The historical 1461.214
number is provenance from `gdn2-chunk-gate-range`, not a reproduced measurement.
Synthetic gate sweeps must be labeled synthetic; they cannot close this gate.

All exponential sites of the selected forward chain:

| Site | Exponent | Sign for g<=0 | Failure avoided |
| --- | --- | --- | --- |
| causal score Q/L | G_i-G_j, j<=i | nonpositive | no noncausal exp overflow before masking |
| WY right-hand side | G_i | nonpositive | no inverse cumulative decay |
| state carry | G_last | nonpositive | valid decay to zero |
| state update key | G_last-G_i | nonpositive | no reciprocal decay |
| output carry | G_i | nonpositive | no positive rescaling |

The representation is a per-chunk logarithmic prefix plus directly evaluated
causal pair differences. It is not the KDA endpoint/midpoint product of a
large positive and negative exponential. No division by exp(G) occurs.
Consequently exp(1461/2) is never evaluated. Precision, independently of
finiteness, must still be measured for prefix subtraction and triangular
solves; no universal gate-domain qualification follows from the sign proof.
The chosen baseline has 64-token chunks, with no anchored 16-token fast path.
The five-size survey will compare reset intervals {4,8,16,32,64} before any
measured gate-range claim or later blocked Cube implementation is accepted.

## Synthetic five-size observation (not checkpoint evidence)

CPU Torch 2.14.0, seed 20260914, B=1/T=4096/H=16/K=V=128, FP32;
each 64-token gate block is calibrated to cumulative decay 1461.214.
All outputs and states are finite. No performance timing is inferred.

| Prefix reset size | Maximum block decay | Output relative L2 | State relative L2 | Output max abs | State max abs |
| --- | --- | --- | --- | --- | --- |
| 4 | 207.451 | 2.304e-7 | 2.522e-7 | 5.122e-8 | 1.013e-6 |
| 8 | 354.658 | 4.076e-7 | 8.151e-7 | 1.378e-7 | 5.156e-6 |
| 16 | 604.186 | 7.970e-7 | 1.430e-6 | 2.086e-7 | 4.858e-6 |
| 32 | 970.827 | 1.620e-6 | 3.413e-6 | 5.178e-7 | 1.550e-5 |
| 64 | 1461.214 | 3.284e-6 | 5.908e-6 | 1.068e-6 | 1.562e-5 |

This supports continuing with the 64-token FP32 baseline on synthetic inputs.
It does not establish the real-checkpoint gate-domain limit or satisfy the
required random-token and natural-text replay. Reproduce with:

```sh
python kernels/projects/a5/gdn2_chunk_fwd/ref/survey.py --synthetic --output tmp/GD2-01/synthetic-survey.json
```

For actual model evidence, use the same script with `--checkpoint` and
`--tokenizer` pointing to local assets. It verifies the recorded checkpoint
SHA-256, uses CPU Torch only, and reports each layer and both sample types.

## Native WY dependent-loop diagnosis

The first full native run of revision `9ba48cd`, B=1/T=4096/H=16,
FP32/block_dim=8, failed at stage `u` (max absolute error 0.05106529966).
All five stages compiled and executed; successful native exits did not establish
numerical correctness. Environment: Ascend 950PR_9589 V100, CANN compiler and OPP
9.2.0, Python 3.12.13, Torch 2.12.0+cpu. The OPP inventory contains ascend950,
ascend910b and ascend910_93. These observations do not apply to other devices.

Re-solving WY from its actual input blobs gave max absolute error 0.05105925724
and relative L2 0.009164325893. A diagnostic recurrence dropping the immediately
preceding row matched native output within 5.960464478e-8, locating the missing
contribution without accepting that altered recurrence as the reference.

A B=1/T=4/H=1/block_dim=1 control uses a generated strictly lower matrix with
only its first subdiagonal set to 0.25. Native artifact controls:

| WY VF form | U max absolute error | W max absolute error |
| --- | --- | --- |
| original dependent inner bound | 0.2538757324 | 0.25 |
| native float load/store overloads | 0.2538757324 | 0.25 |
| VV_ALL barriers | 0.2538757324 | 0.25 |
| fixed 64 iterations, explicit `j < i` guard | 0 | 0 |

The scoped source workaround uses the last form, preserving all terms and their
order. This establishes a failing native dependent-bound form, not whether the
vendor compiler or hardware implements it incorrectly. Artifact controls alone
are not source-kernel qualification. The source change passes 44 targeted host
tests, T=2/H=1 pipesim at block_dim=1, and static checking with zero errors and
warnings. Full native source regression remains pending. Redacted failure and
control receipts are retained in ignored `tmp/GD2-02/`.
The tokenizer must be the recorded local revision; vocabulary size alone is
not proof of tokenizer provenance. The natural-text sample is repeated to
4096 tokens, and that construction is identified in the report.

The scoped task branch passes 223 host tests (5 device-dependent skips),
including 44 chunk tests with the supplied FLA oracle. All five entry checks
report zero errors and warnings. The combined T=65/H=16/block_dim=1 pipesim
case compares every intermediate and both final outputs, with no hazard or
deadlock reported. This covers head reuse together with a cross-chunk tail.
