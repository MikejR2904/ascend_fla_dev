# PKDA FP32 chunk forward

This unit follows FLA `precond_kda/naive.py` at
`e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`. It implements a five-launch FP32
forward with separate main state and ATK diagonal state. PK-04 adds real
NPU in-process acceptance and same-card timing to PK-02's CPU references and
bounded functional/pipe models. The qualification applies to the exact
950PR_9589 V100 / CANN9.2.0 environment and executed cases below. SSH board/aclnn
launchers, training/backward and real-checkpoint logits remain untested.
FLA Triton was not executed. There are no published model weights
used by this task.

## ABI comparison

The premise that FLA PKDA and the existing KDA wrapper differ only by ATK
arguments is not literally true. The pinned public FLA chunk function also
normalizes q/k and has options absent from this repository's KDA wrapper.
The authoritative naive function itself **does not normalize q/k**: it
widens inputs to FP32, scales q, then starts the recurrence. The new operator
consumes q/k as supplied. Normalized contract inputs are generated explicitly
by the CPU input generator, never silently by the runtime.

| Argument/behavior | Pinned FLA public PKDA chunk | Existing `ops/kda/chunk.py` | New PKDA entry |
|---|---|---|---|
| q/k/v | Token-major, equal q/k heads; public chunk normalizes q/k | BF16, K=V128, T multiple64; grouped value heads allowed; no internal normalization | FP32 only, B1/2,T1..4096,H1..32,K=V128, equal heads; inputs unchanged |
| g | Activated per-key log decay or raw gate input | Activated FP32 `[B,T,HV,K]` | Activated FP32 `[B,T,H,128]`, non-positive |
| beta | `[B,T,H]` | FP32 `[B,T,HV]` | FP32 `[B,T,H]` in[0,1] |
| g_atk / beta_atk | `[B,T,H]` | Absent | FP32, non-positive g_atk; beta_atk in[0,1] |
| scale | Optional, default K^-0.5 | Optional f32 runtime scalar, same default | Optional positive finite f32 runtime scalar; same default |
| initial_state | FP32 `[B,H,K,V]` | FP32 `[B,HV,128,128]` | Optional independent FP32 `[B,H,128,128]`; zero default |
| initial_A_state | `[B,H,K]` | Absent | Optional non-negative FP32 `[B,H,128]`; zero default |
| output_final_state | Returns both final states or None | Returns main state or None | Returns `(o,S,A)`; S/A both None when false |
| use_gate_in_kernel | Raw softplus gate, or bounded sigmoid with lower_bound | Not an argument; no gate activation | True explicitly rejected |
| safe_gate / lower_bound | Safe-gate branch and activation bound; see FLA source | Not arguments; cannot infer their semantics from the KDA wrapper | Non-default explicitly rejected; span validation is a distinct check |
| cu_seqlens / cu_seqlens_cpu / chunk_indices | Variable-length support metadata | Absent | Non-default rejected |
| cp_context | Explicitly unsupported in pinned PKDA | Absent | Non-default rejected |
| transpose_state_layout | Optional transposed main-state layout | Absent | True rejected; K-major only |
| x / eps | ATK squash parameters | Absent | Fixed1.5 /1e-6; other values rejected |
| log_atk_scale | Per-head center, default-.2 | Absent | FP32 `[H]`, same default |
| solve_tril_precision | Triton solve option | Absent | Non-default rejected; this solve is FP32 |
| disable_recompute | Training/recompute option | Absent | True rejected; forward only |
| return_intermediate_states | Inference cache option | Separate cached-forward API | True rejected; no cached-training claim |
| Extra kwargs (A_log,dt_bias,…) | Used for raw gate activation | Absent | Explicitly rejected |
| device / block_dim | Not this Ascend launch ABI | a5,bd1..4 | a5/cce,bd1..4; separate-process verification |
| launcher | Triton | In-process NPU bridge | Explicit sim/pipesim; native inprocess qualified below; SSH board/aclnn untested |

All tensors must be contiguous and on one device. Key row L2 norm must be
<=1+1e-5; the caller must normalize if needed. Wrong dtype/geometry/state
shape, nonfinite inputs, invalid gates, excessive forward span and autograd
requests fail before kernel launch. The 340M training geometry H8/K=V128 is
covered by CPU references and native execution; the 1B H14/K120 geometry is explicitly unsupported. Native H14 tests use K128.
Validation on an NPU can synchronize and use Torch NPU checks; the measured
end-to-end wrapper times below include this validation overhead. Host execution performs only validation,
default allocation, argument/view binding and ordered launches, with no
reference-math fallback or implicit dtype conversion.

## Recurrence and reuse

For each token (q includes the attention scale), with original key k:

```
A = exp(g_atk) * A + beta_atk * k²
r = log(A + 1e-6) - log_atk_scale
M = exp(-log(1.5) * r / (1 + abs(r)))
p = k * M
Sdec = exp(g)[...,None] * S
S = Sdec + outer(beta * p, v - kᵀ * Sdec)
o = qᵀ * S
```

The residual predicts with k; only the write uses p. Pinned naive lines74–83
and the `Aqk=q@k_precond.T` / `Akk=k@k_precond.T` comments in FLA chunk.py
establish this asymmetry. A negative control replacing the prediction key
with p is retained in `ref/survey.py`; observing A alone would miss the error.

The unit reuses the **decomposition and equations** of `kda_fwd_stable`,
with FP32 Vector stage code derived locally from the repository's five-stage
GDN-2 implementation. It does not call the old BF16 KDA kernels, inherit
A5K-01/A5K-02 validation or modify either upstream source:

| Stage | Reused KDA logic | PKDA realization |
|---|---|---|
| prepare | `gate.py` chunk-local cumulative log gate | Fuse sequential ATK A/M/p; emit scaled q, p, beta*k, beta*v and cumulative g |
| scores | `intra.py` causal Aqk and strict-lower Akk | Left Aqk key is q; left Akk key is beta*k; both right keys are p |
| WY | inverse + `wy.py` triangular transformation | Direct FP32 solve `(I+L)U=beta*v`, `(I+L)W=beta*k*exp(gc)`; no BF16 inverse materialization |
| scan | repaired `recurrent.py` residual and state update | delta=U-W*S_in; tail key is p*exp(gc_last-gc); save incoming S per chunk |
| output | repaired `recurrent.py` state/output composition | `(q*exp(gc))*S_in + Aqk*delta`, in a separate Vector launch |

ATK is fused into prepare because both consume each input key and because A
must carry chronologically across chunks. Each prepare/scan worker owns whole
heads; other stages own independent chunk/head tiles. A is `[B,H,128]`, S is
`[B,H,128,128]`; neither is reset at a chunk boundary, aliased, or mutated in
place. The five launches exchange fully written FP32 GM buffers. Tail rows
zero q/p/beta*k/beta*v while carrying cumulative g; only valid tokens update A.
No saved state for backward is advertised.

Every local allocation has one slot. Prepare owns4,736 bytes in13 buffers;
scores/WY/scan/output own160/176/224/208 KiB respectively. FP32 rows have a
512-byte pitch, scalar carriers32 bytes. Same-side auto_sync protects DMA/VF
ownership and reuse, and explicit VF STORE→LOAD barriers protect local
recurrences. This implementation has no cross-side Aqk event/slot handoff.
The existing 32-ID limit is unchanged.

## Precision and gate range

All accumulation, stage materialization, outputs and both states are FP32.
The budget is unchanged: relative L2<=1e-4 for **each** of o/S/A, plus finite,
shape/dtype and elementwise atol=rtol=1e-4 checks. Missing outputs, zero-output
substitution and nonfinite values are rejected by the comparator tests.

ATK's M lies between1/1.5 and1.5. The survey measures `log(A+eps)` and M
against FP64 arithmetic on the same FP32 inputs; FP64 is a precision probe,
not the correctness golden. Correctness goldens are Torch CPU FP32.

Across the16 contract cases, log(A+eps) spans[-13.81551,-1.56111].
The FP32-vs-FP64 max absolute errors are7.54527e-7 for log and1.53182e-7
for M. Independent chunk-vs-naive max relative L2 is3.61568e-7 for o,
2.12391e-7 for S and0 for A; these are CPU mathematical results. Actual
T65/H1 repaired pipe-model outputs vs naive are4.19158e-7 /4.12872e-7 /0.

The full chain uses `exp(gc_i-gc_j)` for causal scores, `exp(gc)` for the WY
right-hand side and incoming-state contribution, and `exp(gc_last-gc)` for
state writes. With non-positive g all these exponents are non-positive.
There is no exp/exp division or centered positive exponential. ATK adds
`exp(g_atk)<=1`, the logarithm of the positive quantity `A+eps`, and a bounded
squash exponential.
These facts remove the old overflow mechanism but **do not prove arbitrary
FP32 prefix-difference accuracy**. A large first decay followed by small
increments loses precision when rounded prefixes are subtracted. The survey
retains failed span4096 experiments outside the public domain. The operator
therefore keeps a conservative per-chunk |sum(g)|<=155 forward gate. ATK did
not require tightening155 in the executed CPU domain. KDA's backward105 limit
is not a PKDA backward qualification; backward is absent.

The generated 16-case reference grid covers T1,2,63,64,65,128,192,1024,4096,
H1/8, B2/H3 ownership, zero/nonzero independently initialized states, exact
span105/155 and near-zero rows. The CPU survey compares an independent chunk
triangular solve against the pinned FLA token recurrence for all three outputs.
Reduced model checks also compare actual returned outputs directly to both
CPU oracles. Cross-bd checks compare SHA-256 of every one of14 stage outputs.
See `validation.json` in the unit for executed case/stage receipts and numbers.

## Native failure, repair and acceptance (PK-04)

The original prepare kernel passed the first B1/T4096/H8 native workload,
but failed an **in-domain** strong/weak decay case. In B1/T192/H8, seed71,
each chunk starts with FP32 `g=-154.9992` and continues with `g=-1e-5`.
The span remains below155. Sequential FP32 addition accumulated a
`3.35693359375e-4` prefix error; subtracting rounded prefixes then produced
finite but wrong output/state. Against pinned CPU FP32 naive, relative L2
was `1.2967912146184837e-4` for o and `1.8233854552833124e-4` for S.
The original receipt is retained in `evidence/original-native-failure.json`.

Prepare now uses FP32 compensated prefix addition, resetting a new512-byte
UB compensation row per chunk. It reuses dead temporary registers and the
existing VF STORE→LOAD recurrence barrier. ATK, the other four kernels,
public ABI, raw-input semantics,155 gate and1e-4 budget are unchanged.
Bounded T64/H1 prepare-only sim and pipesim reproduce the old error and
observe zero prefix error after repair, with no pipe hazards/deadlock and
balanced events. `test_prepare_compensates_weak_decay_after_large_jump`
checks every prefix element, including the weak tail.

After repair the first custom execution again used full B1/T4096/H8/K=V128.
All five vendors were built before any custom kernel call; every block_dim
ran in its own process. The native suite executed28 full-chain cases at each
of bd1,2,3,4 (112 total), plus10 public API checks per bd (40 total). Each
case checks all14 actual stage outputs against independent Torch CPU FP32
chunk math, and public `(o,S,A)` against both chunk and pinned CPU FLA naive.
Output poisoning, input immutability and public/staged equality pass. All14
stage byte hashes and all generated input hashes agree across all four bds
for every case. These are actual device observations, not inherited KDA results.

The28 cases include all16 contract cases, B2/T65/H32, B2/T130/H3,
T129/H14/K128, small raw q/k rows, zero q/k, no main decay, multichunk span155,
zero ATK state across chunks, and the adversarial gate at seeds71/72/73.
The additional API checks cover all four independent optional S/A combinations,
default ATK center, state carry splits1/63/64/65/129 and output_final_state=False.
The delivered verifier was independently rerun atbd4:28+10 passed again,
with the same stage hashes. This repeat is separate from the112/40 totals.

| Native public output | Maximum relative L2 vs CPU chunk | Maximum relative L2 vs CPU FLA naive |
|---|---:|---:|
| o | 4.92169573e-7 | 6.20924408e-6 |
| final_state | 4.94926986e-7 | 6.61719892e-6 |
| final_A_state | 4.47174195e-7 | 4.47174195e-7 |

For the first full T4096/H8 case alone, naive relative L2 is
4.45861995e-7 /4.01681474e-7 /7.20234805e-8 for o/S/A. All existing elementwise
checks also pass. Native A is not assumed byte-identical to CPU A.
Full host tests:521 passed,5 skipped; PKDA-specific51 passed, also51 passed
on the board's CPU. Current bounded model checks were rerun on repaired code.
`evidence/host-pk02.json` preserves historical host-only evidence for the old
source; it does not qualify this repair.

Environment: SoC950PR_9589 V100 (Ascriptor a5/950 profile), CANN9.2.0,
OPP directories ascend950, ascend910b, ascend910_93; Python3.12.13,
Torch2.12.0+cpu with torch_npu2.12.0, NumPy2.5.1, Ascriptor0.1.0 at the pin below.
An alternate CANN9.2.0 header installation fails vendor compilation:
`tensorutils_cce.h:143` tests `IMPL_UTILS_SYS_MACROS_H`, while the vendor's
`sys_macros_impl.h:17` declares `IMPL_UTILS_SYS_MACROS_IMPL_H`, causing duplicate
`g_coreType`. That installation is not qualified. No upstream/header/generated
source was patched. A version string alone does not identify a working toolchain.
Raw numerical receipts, vendor hashes and log lines are indexed in the unit's
`evidence/README.md`; private machine coordinates are excluded.

### Vendor warning audit

The vendor emits `-Wcce-compat` because VF induction variables are uint16,
while `count` and `i+1` condition operands are int32. The pinned CCE owner
`emit.py:1462–1476` emits uint16 induction for positive-step VF loops; its
`docs/cce-support.md` documents that form. Here every runtime count is
`min(64,T-cc*64)` in1..64, and every dependent `i+1` is in1..64. WY's
step2 loop reaches at most62 and its paired row at most63. No bound truncation,
integer overflow or negative step is possible in the declared domain.

All four generated VF header files are byte-identical acrossbd1..4. To check
this compiler's handling of **every legal dynamic bound**, the delivered
`tails` mode ran a fresh fullT4096 case followed by64 full-chain cases
B1/T(64+n)/H3 for n=1..64, with nonzero generated inputs/states. All14 actual
stages and public outputs pass both CPU FP32 oracles at the original budget.
This targeted domain check, generated-source audit and unchanged owner form
support retaining the warning for this exact toolchain and bounded ABI.
It is not a general waiver for other CANN versions or loop bounds. Warning
text and original build-log line numbers are retained in
`evidence/vendor-diagnostics.json`; no warning flag was suppressed.

Other build messages concern CMake install-path policy/unused cross-compiler
settings, deprecated CANN headers and a template regex escape. These affect
packaging/tool maintenance; all20 expected vendors were built and loaded.
The audit and all64 tail receipts are separate from the112-case main grid.

## Same-device performance

These synchronized timings compare the public operator to the pinned FLA
naive recurrence running **Torch NPU eager** on the same card, atbd4 and FP32.
They do not compare against Triton or an optimized native baseline. Each of
three rounds runs baseline/candidate/baseline, with one initial warmup per
implementation, two samples on each baseline side and three candidate samples.
Synchronization brackets every sample. Both implementations first pass CPU
FP32 oracle checks. Candidate time includes validation and allocation; compile,
input generation/transfers and CPU goldens are excluded. There is no speed gate.

| Shape B1/H8/K=V128 | Round | Baseline midpoint (ms) | Candidate median (ms) | Candidate/baseline |
|---|---:|---:|---:|---:|
| T1024 | 1 | 613.556 | 46.836 | 0.07634 |
| T1024 | 2 | 606.576 | 47.080 | 0.07762 |
| T1024 | 3 | 604.783 | 46.860 | 0.07748 |
| T4096 | 1 | 4696.891 | 179.922 | 0.03831 |
| T4096 | 2 | 4679.362 | 179.846 | 0.03843 |
| T4096 | 3 | 4690.992 | 179.850 | 0.03834 |

The midpoint averages the before/after baseline medians. All raw samples
are retained in `evidence/perf-t1024.json` and `evidence/perf-t4096.json`.
These measurements establish only this eager-baseline comparison.

## Near-zero normalization divergence

This is not merely an epsilon difference inside naive: naive never normalizes.
For q/k row norms1e-2,1e-4,1e-6 and0, `ref/survey.py` compares raw naive with
(1) explicit caller `F.normalize` and (2) the pinned chunk normalization formula
`x/sqrt(sum(x²)+1e-6)`, each followed by the same CPU recurrence. This is a
formula comparison, **not Triton execution**. For the fixed two-token sample:

| Input row norm | Raw naive output norm | Chunk-formula normalized output norm | Explicit F.normalize output norm |
|---|---:|---:|---:|
| 1e-2 | 6.35024e-4 | 6.87612e-2 | 6.91586e-2 |
| 1e-4 | 6.34987e-6 | 6.32676e-3 | 6.91586e-2 |
| 1e-6 | 6.34986e-8 | 6.34989e-5 | 6.91586e-2 |
| 0 | 0 | 0 | 0 |

For BF16, FLA's public chunk additionally rounds normalized q/k back to their
input dtype and returns output in the input dtype. This unit accepts FP32 only;
BF16 is rejected explicitly. No optional BF16 quality result is used to declare
correctness, and no prior BF16 KDA evidence is reused.

## Reproduction

Use compatibility-selected Ascriptor library90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5,
kernels runnerb3b3f9c16df7c4626ed3c081032a1be5a753d0b1 and FLA revision above.
Set `PYTHONPATH` to the library and repository and `FLA_PKDA_NAIVE` to the
pinned `fla/ops/precond_kda/naive.py`. CPU acceptance used Python3.11.15,
Torch2.10.0+cpu and NumPy1.26.4. Outputs below belong in ignored task scratch.

```bash
unit=kernels/projects/a5/pkda_chunk_fwd
python "$unit/run.py" reference --output tmp/PK-04/reference
python "$unit/ref/survey.py" --output tmp/PK-04/survey.json
python "$unit/run.py" check --launcher sim --case t1_h1 --block-dim 1 --output tmp/PK-04/sim
python "$unit/run.py" check --launcher pipesim --case t2_h1 --block-dim 1 --output tmp/PK-04/pipesim
python "$unit/verify.py" --launcher pipesim --case t65_h1 --case b2_t2_h3 --block-dim 1 --output tmp/PK-04/bd1
# Repeat only b2_t2_h3 in separate processes at block_dim2,3,4 for byte comparison.
python -m pytest tests/test_pkda_chunk_fwd.py -q
python tools/gen_matrix.py --check
python tools/pm_board.py --check
```

Static checks use `ascriptor check kernels/.../prepare.py::pkda_chunk_prepare`
and each `stages.py::pkda_chunk_{scores,wy,scan,output}`. Source emission and
model checks are separate from vendor compilation or native execution.

Native reproduction uses an externally selected healthy idle device and a held
shared device lock. Keep machine values in ignored external configuration.
Verify the Python/library/CANN identities before entering this process:

```bash
# One invocation per process; repeat the suite with block_dim 1,2,3,4.
python "$unit/verify_native.py" --block-dim 4 --mode suite --output tmp/PK-04/native-bd4
python "$unit/verify_native.py" --block-dim 4 --mode perf --output tmp/PK-04/perf-bd4
python "$unit/verify_native.py" --block-dim 4 --mode tails --output tmp/PK-04/tails-bd4
```

Each invocation precompiles all five vendors before its first custom operation.
`full` executes only the first full workload; `compile` performs no custom-kernel
execution. Neither alone substitutes for `suite`. Compare all input and stage
hashes between suite summaries. The `perf` mode also runs the full workload
before its synchronized measurements. The unchanged canonical runner is for
reference/model checks; native acceptance uses `verify_native.py`.
