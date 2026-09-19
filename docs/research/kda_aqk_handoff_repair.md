# KDA public Aqk integration: A5K-02

Public stable KDA forward and cached-forward now select the repaired Aqk recurrent
kernel. The original upstream path rejects odd C with B*HV > block_dim before
launch. The diagnostic baseline explicitly retains the original recurrent.
[Issue81](https://github.com/ddddwee1/ascend_fla_dev/issues/81) and
[PR83](https://github.com/ddddwee1/ascend_fla_dev/pull/83) deliver this integration.

Cached-forward precompiles backward. Under D-PM-24, the local inverse_mm derivative
makes only l0c_dvh and l0c_dvbeta single-slot L0C tensors and updates their views.
Arithmetic, event credits, loops, lookahead/drain, other buffers and ABI stay the
same. Existing mode-zero M/FIX ownership retires the final FIX read before the
next M overwrite. Actual lowering uses cube 32/vector 15; the unchanged upstream
still refuses 34 > 32. No compiler limit or upstream source was changed.

All required current-PR acceptance checks pass. The compact receipt is
[`public_validation.json`](../../kernels/projects/a5/kda_fwd_stable/public_validation.json).
The tested implementation is 888862be37d6f0c4effece69ec4915c22ea8b5d2; the subsequent
closeout commit adds reports/qualification evidence only. Historical A5K-01
results are retained in the separate appendix and `validation.json`.

## Current source and environment

| Component | Identity |
|---|---|
| Library | 90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5 |
| Kernels | b3b3f9c16df7c4626ed3c081032a1be5a753d0b1 |
| FLA | e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2 |
| Native | 950PR_9589 V100 / CANN9.2.0 / Python3.12.13 / Torch2.12.0+cpu / torch_npu2.12.0 |
| OPP | ascend950, ascend910b, ascend910_93 |
| CPU reference/model/host | Python3.11.15 / Torch2.10.0+cpu |

All 481 project/dependency/oracle source files were hash-verified before native
execution. Complete forward/control/all-nine-backward vendor builds passed at
bd 1/2/3/4. Each bd uses a separate process, with every required vendor registered
before the first custom kernel. Device ownership and health were checked under
the configured lock; final health was 0. Foreign processes and devices were not
modified. Private machine values are excluded from public evidence.

## Current numerical and synchronization acceptance

The hardware-first sequence ran full inverse_mm B1/T4096/HV32/bd 4, actual public
plain and cached-forward B1/T4096/H32/HV32/bd 4, and complete backward workloads
before reduced models. Golden computation uses Torch CPU FP32. Forward/leaf
checks retain rtol=atol=.02 and relative-L2 <=.05; backward retains its existing
per-gradient budgets, finite outputs and explicit rejection of zero gradients.

- Full inverse leaf and three reuse/batch/uneven-work cases pass all five outputs;
  relative-L2 ranges approximately .00164–.00168. An independent CPU ABI-slice
  reference agrees exactly and rejects zero-output controls.
- Full public T4096 output relative-L2 is .0032664153, max_abs 1.2901262e-5;
  final-state relative-L2 is .0025471812, max_abs 7.0009381e-5. Both public entries,
  every head/chunk, all nine caches and even-C original bytes pass.
- The 336-case grid covers C1..6, seven (H,HV) pairs, zero/random state, bd 1..4.
  All 9240 head/chunk rows pass both independent CPU and pinned FLA oracles.
  Every output, prefix state and cache is byte-identical across bd 1..4. Worst
  slice relative-L2 is .003904291(output) and .004268510(state). Original-control
  output failures are 36/30/30/12 by bd; timing-sensitive control passes remain
  passes. Both unsafe upstream entries reject before launch.
- All 13 backward regressions pass: five existing unit cases; KimiT1024,
  H16/HV32T1024 and T4096, plus KimiT4096; and spans 46/94/104/exact 105.
  Existing unit-oracle FP32 autograd/BF16 output ABI is retained for the five
  cases; real/wide cases use unrounded CPU FP32 recurrence/autograd.
  The exact 105 fixture uses token 0=0, 60 increments of -1.5, then 3 of -5 per chunk;
  the same input reaches kernel and oracle. Span 106 is rejected by the real
  cached-forward API. No gate threshold or gradient budget was relaxed.

| Gradient | Full KimiT4096 relative-L2 | Worst of 13 cases | Budget(strict <) |
|---|---:|---:|---:|
| dq | 0.03034958 | 0.04781887 | 0.05 |
| dk | 0.03451302 | 0.09298248 | 0.15 |
| dv | 0.00354112 | 0.00382118 | 0.05 |
| dbeta | 0.00331782 | 0.00647579 | 0.05 |
| dg | 0.05706130 | 0.17371406 | 0.25 |
| dh0 | 0.00234284 | 0.00516330 | 0.05 |

Reduced inverse sim/pipesim covers B1/HV1/C1,3,5/bd 1 and C1/bd 2. All 8 checks return
five complete numerical outputs, zero hazards and no deadlock. Aqk controls cover
C1/2/3/5 at HV2/bd 1: original and qg-only retain two hazards and incorrect replay
for odd C; repaired is clean and replay-correct. All 12 expected outcomes pass.
These are reduced-model results; full-backward standalone harness/model
qualification and wider input domains are not inferred.

## Current public-path performance

Each shape uses three synchronized baseline/candidate/baseline rounds, 10 warmups
and 50 samples per measurement. Public gate checks, layout, allocation and launches
are included; compilation and cached-forward are excluded. Torch NPU composition
also passes both CPU golden checks before timing. Every raw sample is archived.

| Shape | Round | Baseline before(ms) | Candidate(ms) | Baseline after(ms) | Faster baseline/candidate |
|---|---:|---:|---:|---:|---:|
| kimi_t1024 | 1 | 19.456404 | 19.506187 | 19.475997 | 0.997448 |
| kimi_t1024 | 2 | 19.440594 | 19.426497 | 19.319314 | 0.994483 |
| kimi_t1024 | 3 | 19.274349 | 19.279193 | 19.313852 | 0.999749 |
| kimi_t4096 | 1 | 74.550863 | 74.562949 | 74.542489 | 0.999726 |
| kimi_t4096 | 2 | 74.559864 | 74.579152 | 74.563674 | 0.999741 |
| kimi_t4096 | 3 | 74.597541 | 74.588059 | 74.601321 | 1.000127 |

Torch NPU composition medians: T1024 31.845413 ms;
T4096 94.630783 ms. The candidate is
slightly slower in all T1024 rounds versus the faster bracket and in two T4096
rounds. These data do not establish a consistent speed gain. Existing compiler
warnings are performance advisories about ND-to-NZ movement and UB bank stride;
no correctness/synchronization warning was suppressed or left unresolved.

## Host checks, evidence and recovery

Host suite: **338 passed / 5 skipped**; canonical CPU reference 8/8. The skips are NPU-only
modules in the CPU environment, not replacements for the native checks above.
Matrix, PM-board, privacy and diff checks pass. The archive has 889
manifest-verified payload files, including all 481 exact tested source files,
raw metrics/digests/timing samples, model diagnostics, sanitized verifier logs
and replay helpers. All hashes were verified after extraction; an ordinary
restored source tree reran 8/8 CPU reference cases. Machine configuration is external.

- Archive: `a5k-02-evidence.tar.gz`
- SHA-256: `4602633ad3456f2e3518f3b0b31d0a3168706c028c3586c4b89828805b5b384e`
- Manifest SHA-256: `8f532c59daaa256ed2afc17a12e3dd5d159a4cd452e5db30a850588ac4bd0aef`

The archive contains the tested implementation snapshot; final report-only changes
are in the PR. No A2/A3 qualification, A5 wave exit, wider gate/block domain or
consistent speedup is claimed. PM owns matrix/board updates.

## Historical appendix: A5K-01 standalone composition

Everything below predates this public integration and qualifies the A5K-01
standalone composition only. Its environment, control failure counts, timings,
host totals and archive are distinct from the current results above.

### Source and hardware scope

| Component | Executed identity |
|---|---|
| Project baseline | `d0ae60f` plus explicit WY `depth=2` compatibility |
| Library | `90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5` |
| Kernels | `b3b3f9c16df7c4626ed3c081032a1be5a753d0b1` |
| FLA naive | `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`, `fla/ops/kda/naive.py` |
| Silicon / CANN | 950PR_9589 V100 / 9.2.0 |
| OPP packages | `ascend950`, `ascend910b`, `ascend910_93` |
| Native Python / Torch / torch_npu | 3.12.13 / 2.12.0+cpu / 2.12.0 |
| Host Python / Torch | 3.11.16 / 2.14.0 |

Every hardware number below belongs to this configuration. Native execution
uses Torch NPU tensors; golden execution uses Torch CPU FP32. No conclusion is
transferred to another device, A2/A3, a wider gate range, or a larger block count.
The initial device failed its pre-run driver health check before any kernel ran.
An unused healthy device was selected under an exclusive task lock; its driver
health was 0 before and after every native invocation. No device was reset.

The accepted library requires an explicit `VcMutex` depth. The old WY source
omitted it and failed compilation. Declaring `depth=2` preserves the previous
default and its two `DBuff` slots. Both baseline and repair include this same
compatibility prerequisite; their arithmetic is identical.

### Repair and lifecycle argument

The accepted upstream `projects/a5/kda_fwd/kernels/recurrent.py` remains read-only.
The local derived source changes its entry name and exactly two slot expressions:

```python
# Original producer and consumer
aqk_slot = Var(c_idx % 2)
# Repaired producer and consumer
aqk_slot = Var(((pair_idx - pair_begin) * C + c_idx) % 2)
```

Each core owns a contiguous interval of `(batch, value-head)` pairs. Each chunk
publishes one BF16 64x64 Aqk tile to L1, then the cube reads it for both V64 output
tiles. The valid channel starts with two credits, permitting one later
publication while an earlier tile is still being consumed. Two slots therefore
must rotate with the publication sequence, including head boundaries.

With chunk-only rotation, odd C ends on slot 0 and the next head starts on slot
0: the two-credit window can contain two publications in the same physical
slot. The repaired per-core sequence alternates slots continuously. Retirement
still follows the final MTE1 read, and the existing event drain still retires
outstanding credits. Both 8192-byte slots, all event operations, arithmetic,
casts, other buffer indices and launch structure are unchanged. No extra buffer,
lookahead or synchronization operation is introduced. For even C, the slot
expression is algebraically identical to the original one.

### Native correctness

Inputs are generated at run time, seed 2026: BF16 normalized q/k, BF16 v scaled
by 0.04, negative FP32 gates calibrated to maximum per-chunk span 46, FP32 beta
in [0.05,0.5], and zero or random FP32 initial state. No recorded golden is used.

The independent oracle stores state in `[V,K]` orientation and uses batched
matrix-vector prediction/update/output. It shares no kernel checkpoints or FLA
implementation code. The second oracle file-loads the pinned FLA naive recurrence
on CPU. Their outputs and all chunk states must agree to relative L2 <= 1e-5
before either judges the kernel. Native outputs require finite values,
`rtol=atol=0.02` and relative L2 <= 0.05; these existing budgets are unchanged.

Every head/chunk has separate output and state metrics against both oracles.
Prefix launches observe each intermediate chunk state through the unchanged
final-state ABI. The full raw metrics remain in the verified archive. All final
states and upstream stage tensors are also byte-identical to baseline, including
odd C. The corrupted baseline output can pass the loose absolute tolerance;
the relative-L2 gate is essential to detect it.

The full selected workload ran before reduced models: B=1, T=4096, H=HV=32,
K=V=128, block_dim=4, nonzero initial state. Repaired output relative L2 is
0.0032664153 (max absolute difference 1.2901262e-5); state relative L2 is
0.0025471812 (max absolute difference 7.0009381e-5), against the independent
CPU oracle. All stages equal baseline byte for byte. Worst head/chunk errors
across both oracles are 0.004008274 for output and 0.004388919 for state.

The fresh pre-repair full-chain baseline also directly reproduced the defect
on silicon at H=HV=32/bd=4: C=1 output relative L2 1.0602632, C=3 0.6897817,
while state relative L2 stayed about 0.00246. C=3 is therefore a silicon
counterexample in this run, not only an extrapolation from A2-04's pipe model.

The grid is C=1..6 x (H,HV)={(1,1),(1,2),(1,4),(2,4),(4,4),(4,8),(32,32)}
x zero/random initial state x block_dim=1..4: 84 input sets, 336 launches of
each variant, plus prefix state checks. It contains single-owner cores, repeated
heads, GQA, uneven core partitions and Kimi head count. There are 9,240
head/chunk rows across the four block counts. All repaired rows pass. Baseline
failures are 36/30/26/12 cases at bd=1/2/3/4 respectively; timing-sensitive
baseline cells that pass are retained as passes, not rewritten as failures.

Worst grid head/chunk relative L2 is 0.003904291 for output and 0.004268510
for state; worst absolute differences are 1.2379838e-5 and 1.0044687e-4.
`compare_repair.py` checks all stage and prefix-state SHA-256 values across
block counts, all even-C baseline bytes, complete case coverage and identical
source identities/budgets. All checks pass.

### Synchronization regression and host checks

Reduced CPU pipesim uses B=1/H=1/HV=2/bd=1, C in {1,2,3,5}, and the actual
shipped recurrent source. The original and qg-only mutation are retained as
negative controls; no intentional fault is introduced on the device.

| C | Original hazards / replay | qg-only hazards / replay | Repaired hazards / replay |
|---|---|---|---|
| 1 | 2 / wrong | 2 / wrong | 0 / exact |
| 2 | 0 / exact | 0 / exact | 0 / exact |
| 3 | 2 / wrong | 2 / wrong | 0 / exact |
| 5 | 2 / wrong | 2 / wrong | 0 / exact |

All 12 diagnostic cases meet their expected verdict, with no deadlock. Even-C
output, state and replay equal baseline byte for byte. This verifies event/slot
behavior in the reduced model; silicon correctness is established separately
above. The canonical standalone runner also passes all 8 CPU reference cases
and the complete five-stage `aqk_c1_gqa` case under both sim and pipesim at bd=1.
Its CPU stage score reference evaluates direct non-positive gate differences.

Host tests: **229 passed, 5 skipped** with both FLA CPU oracle paths supplied.
Matrix and PM board consistency checks pass. Existing numerical/domain guards
are unchanged. Added tests compare the independent CPU recurrence to FLA at
every chunk and ensure a finite, ordinary-scale wrong-head output is rejected.
The existing source-inventory test additionally covers the new recurrent file.

### Same-device performance

Shapes are B=1, H=HV=32, K=V=128, span 46, random initial state, block_dim=4.
Each timing group has 10 warmups and 50 samples with NPU synchronization before
and after every timed call. Three baseline/repair/baseline rounds run on the
same locked device. Compilation, CPU reference generation and validation are
outside timing. Both native variants include allocation, input/output layout
conversion and five launches. The independent Torch NPU vectorized composition
also excludes validation and must pass correctness before timing.

All measured native cases pass both CPU oracles and equal baseline bytewise.
Torch NPU output/state also pass the CPU correctness budget. The table gives
medians in milliseconds; ratio is the faster bracketing baseline divided by
the repair. These are composition wall times, not isolated kernel latencies.

| T | Round | Baseline before ms | Repair ms | Baseline after ms | Baseline/repair |
|---|---|---|---|---|---|
| 1024 | 1 | 18.598 | 18.606 | 18.578 | 0.9985x |
| 1024 | 2 | 18.642 | 18.648 | 18.622 | 0.9986x |
| 1024 | 3 | 18.698 | 18.663 | 18.703 | 1.0019x |
| 4096 | 1 | 72.194 | 76.213 | 72.172 | 0.9470x |
| 4096 | 2 | 72.109 | 72.147 | 72.067 | 0.9989x |
| 4096 | 3 | 72.106 | 72.151 | 72.121 | 0.9994x |

Torch NPU composition is 31.825 ms at T=1024 and 94.451 ms at T=4096. The
median of the three repair medians is 18.648 ms and 72.151 ms, giving 1.707x
and 1.309x versus that independently implemented Torch baseline. These ratios
describe the existing native composition versus Torch, not a gain from this
Aqk change. At T=4096 the first repair round is 5.6% slower than its faster
bracketing baseline; the next two are within 0.12%. Its cause was not isolated,
so the slow round remains in the receipt. There is no consistent repair speedup.

### Reproduction and retained evidence

The [unit README](../../kernels/projects/a5/kda_fwd_stable/README.md) lists native,
model and cross-block verification commands. The current compact, machine-readable
[receipt](../../kernels/projects/a5/kda_fwd_stable/validation.json) contains the
complete 84-row grid projection against both oracles, per-block baseline failure
lists, source digests, full-workload checks and all six timing medians.

Full per-head/per-chunk values, raw timing samples, stage/state digests, model
diagnostics, host receipts, and exact grid/profile source snapshots are in the
ignored `a5k-01-evidence.tar.gz` archive. The receipt records the archive and
manifest SHA-256 and 395 payload files. Recovery into a separate ignored
directory was tested and every payload digest rechecked. Private paths are
redacted. No recorded input/golden tensor or machine configuration is committed.

The grid source snapshot precedes the addition of an optional standalone
launcher adapter in `repair_runtime.py`. The native default path keeps the same
composition; final native profiling executes the adapter-capable source at both
T values and records its separate source identity. The recurrent/WY source
bytes are identical in grid and profile runs. The final profile identity also
records the baseline recurrent, inverse and FLA naive digests.

Public dispatch integration, backward changes, a new gate-span/block_dim domain,
other devices/SoCs and the general A5 wave remain outside this claim.

### Grid projection

Each row applies to **all four block_dims** because repaired stage/state bytes
match. Numbers are the maximum over heads/chunks and both CPU oracles; the
receipt separates the two oracles and the archive retains every individual
slice. B=1, seed=2026, gate span=46, L=64 and K=V=128 throughout.

| H/HV | C | Initial state | Output rel-L2 | Output max-abs | State rel-L2 | State max-abs | Baseline failed bd |
|---|---|---|---|---|---|---|---|
| 1/1 | 1 | zero | 0.00329905 | 4.8193e-06 | 0.0021672 | 3.13534e-05 | none |
| 1/1 | 1 | random | 0.00324873 | 5.44428e-06 | 0.0021672 | 3.13534e-05 | none |
| 1/2 | 1 | zero | 0.00318334 | 4.86447e-06 | 0.00246911 | 4.10849e-05 | 1 |
| 1/2 | 1 | random | 0.00316472 | 5.19422e-06 | 0.00246911 | 4.10849e-05 | 1 |
| 1/4 | 1 | zero | 0.00340017 | 7.43731e-06 | 0.00228104 | 5.25159e-05 | 1,2,3 |
| 1/4 | 1 | random | 0.00328236 | 7.1238e-06 | 0.00228104 | 5.25159e-05 | 1,2 |
| 2/4 | 1 | zero | 0.00347293 | 9.05036e-06 | 0.00293013 | 3.43975e-05 | 1,2 |
| 2/4 | 1 | random | 0.00331491 | 6.45034e-06 | 0.00293013 | 3.43975e-05 | 1,2 |
| 4/4 | 1 | zero | 0.00332809 | 5.2863e-06 | 0.00279519 | 4.73331e-05 | 1,2 |
| 4/4 | 1 | random | 0.00324142 | 8.13999e-06 | 0.00279519 | 4.73331e-05 | 1,2,3 |
| 4/8 | 1 | zero | 0.00352812 | 7.04569e-06 | 0.0032451 | 4.7029e-05 | 1,2,3,4 |
| 4/8 | 1 | random | 0.00336715 | 7.04569e-06 | 0.0032451 | 4.7029e-05 | 1,2,3,4 |
| 32/32 | 1 | zero | 0.00364974 | 9.24559e-06 | 0.00344247 | 5.37457e-05 | 1,2,3,4 |
| 32/32 | 1 | random | 0.00344124 | 9.24594e-06 | 0.00344247 | 5.37457e-05 | 1,2,3,4 |
| 1/1 | 2 | zero | 0.00334979 | 8.08493e-06 | 0.00226412 | 3.3766e-05 | none |
| 1/1 | 2 | random | 0.0033409 | 8.14791e-06 | 0.00226412 | 3.3766e-05 | none |
| 1/2 | 2 | zero | 0.0033339 | 6.72962e-06 | 0.0029396 | 4.02336e-05 | none |
| 1/2 | 2 | random | 0.00332897 | 6.73323e-06 | 0.0029396 | 4.02336e-05 | none |
| 1/4 | 2 | zero | 0.00346096 | 8.37061e-06 | 0.00309948 | 6.2664e-05 | none |
| 1/4 | 2 | random | 0.00346096 | 8.33068e-06 | 0.00309948 | 6.2664e-05 | none |
| 2/4 | 2 | zero | 0.00351571 | 6.36949e-06 | 0.00279709 | 4.12771e-05 | none |
| 2/4 | 2 | random | 0.00351571 | 6.36949e-06 | 0.00279709 | 4.12771e-05 | none |
| 4/4 | 2 | zero | 0.00354327 | 6.75232e-06 | 0.00392571 | 4.48134e-05 | none |
| 4/4 | 2 | random | 0.00354327 | 6.75232e-06 | 0.00392571 | 4.48134e-05 | none |
| 4/8 | 2 | zero | 0.00358531 | 7.77189e-06 | 0.00332633 | 5.35306e-05 | none |
| 4/8 | 2 | random | 0.00358531 | 7.77189e-06 | 0.00332633 | 5.35306e-05 | none |
| 32/32 | 2 | zero | 0.00355188 | 8.12602e-06 | 0.00391629 | 8.91741e-05 | none |
| 32/32 | 2 | random | 0.00348796 | 8.26293e-06 | 0.00391629 | 8.91741e-05 | none |
| 1/1 | 3 | zero | 0.00327654 | 4.63554e-06 | 0.00283338 | 3.3319e-05 | none |
| 1/1 | 3 | random | 0.00327654 | 4.63554e-06 | 0.00283338 | 3.3319e-05 | none |
| 1/2 | 3 | zero | 0.00331168 | 6.00435e-06 | 0.00282613 | 6.27814e-05 | 1 |
| 1/2 | 3 | random | 0.0032067 | 6.00435e-06 | 0.00282613 | 6.27814e-05 | 1 |
| 1/4 | 3 | zero | 0.00341433 | 7.86223e-06 | 0.00331067 | 5.14723e-05 | 1,2,3 |
| 1/4 | 3 | random | 0.00341433 | 7.86223e-06 | 0.00331067 | 5.14723e-05 | 1,2,3 |
| 2/4 | 3 | zero | 0.00363655 | 6.7259e-06 | 0.00322291 | 9.68091e-05 | 1,2,3 |
| 2/4 | 3 | random | 0.00363655 | 6.7259e-06 | 0.00322291 | 9.68091e-05 | 1,2,3 |
| 4/4 | 3 | zero | 0.00347048 | 7.32666e-06 | 0.00307723 | 4.81834e-05 | 1,2,3 |
| 4/4 | 3 | random | 0.00347048 | 7.32666e-06 | 0.00307723 | 4.81834e-05 | 1,2,3 |
| 4/8 | 3 | zero | 0.00369628 | 7.3585e-06 | 0.00329182 | 5.69876e-05 | 1,2,3,4 |
| 4/8 | 3 | random | 0.00369628 | 7.3585e-06 | 0.00329182 | 5.69876e-05 | 1,2,3,4 |
| 32/32 | 3 | zero | 0.00370377 | 7.78108e-06 | 0.00342714 | 6.47362e-05 | 1,2,3,4 |
| 32/32 | 3 | random | 0.00370377 | 7.78108e-06 | 0.00342714 | 6.47362e-05 | 1,2,3,4 |
| 1/1 | 4 | zero | 0.00342616 | 4.96185e-06 | 0.00280667 | 4.39202e-05 | none |
| 1/1 | 4 | random | 0.0033367 | 6.12531e-06 | 0.00280667 | 4.39202e-05 | none |
| 1/2 | 4 | zero | 0.00344409 | 7.19121e-06 | 0.00303602 | 4.30071e-05 | none |
| 1/2 | 4 | random | 0.00344409 | 7.19121e-06 | 0.00303602 | 4.30071e-05 | none |
| 1/4 | 4 | zero | 0.00360353 | 7.15384e-06 | 0.00302091 | 5.45499e-05 | none |
| 1/4 | 4 | random | 0.00341994 | 7.15384e-06 | 0.00302091 | 5.45499e-05 | none |
| 2/4 | 4 | zero | 0.00363341 | 8.9875e-06 | 0.00323307 | 4.95203e-05 | none |
| 2/4 | 4 | random | 0.00363341 | 8.9875e-06 | 0.00323307 | 4.95203e-05 | none |
| 4/4 | 4 | zero | 0.00390429 | 8.5436e-06 | 0.00355425 | 6.17653e-05 | none |
| 4/4 | 4 | random | 0.00390429 | 8.5436e-06 | 0.00355425 | 6.17653e-05 | none |
| 4/8 | 4 | zero | 0.00357382 | 6.53404e-06 | 0.00379548 | 5.70978e-05 | none |
| 4/8 | 4 | random | 0.00357382 | 7.44744e-06 | 0.00379548 | 5.70978e-05 | none |
| 32/32 | 4 | zero | 0.00379813 | 9.79134e-06 | 0.00333826 | 8.86358e-05 | none |
| 32/32 | 4 | random | 0.00379813 | 9.79134e-06 | 0.00333826 | 8.86358e-05 | none |
| 1/1 | 5 | zero | 0.00341869 | 5.82077e-06 | 0.00357269 | 4.88721e-05 | none |
| 1/1 | 5 | random | 0.00341869 | 5.82077e-06 | 0.00357269 | 4.88721e-05 | none |
| 1/2 | 5 | zero | 0.00350589 | 9.08691e-06 | 0.00270456 | 4.49158e-05 | 1 |
| 1/2 | 5 | random | 0.00350589 | 9.08913e-06 | 0.00270456 | 4.49158e-05 | 1 |
| 1/4 | 5 | zero | 0.00383671 | 7.90681e-06 | 0.00299379 | 5.35473e-05 | 1,2,3 |
| 1/4 | 5 | random | 0.00383671 | 7.90681e-06 | 0.00299379 | 5.35473e-05 | 1,2,3 |
| 2/4 | 5 | zero | 0.00375868 | 6.4979e-06 | 0.00329335 | 5.65564e-05 | 1,2,3 |
| 2/4 | 5 | random | 0.00375868 | 6.4979e-06 | 0.00329335 | 5.65564e-05 | 1,2,3 |
| 4/4 | 5 | zero | 0.00380957 | 6.66338e-06 | 0.00426851 | 7.34571e-05 | 1,2,3 |
| 4/4 | 5 | random | 0.00380957 | 6.66338e-06 | 0.00426851 | 7.34571e-05 | 1,2,3 |
| 4/8 | 5 | zero | 0.00359188 | 9.6655e-06 | 0.00345449 | 6.0562e-05 | 1,2,3,4 |
| 4/8 | 5 | random | 0.00359188 | 9.6655e-06 | 0.00345449 | 6.0562e-05 | 1,2,3,4 |
| 32/32 | 5 | zero | 0.0038198 | 9.8719e-06 | 0.00339478 | 0.000100447 | 1,2,3,4 |
| 32/32 | 5 | random | 0.0038198 | 9.8719e-06 | 0.00339478 | 0.000100447 | 1,2,3,4 |
| 1/1 | 6 | zero | 0.00336604 | 5.86605e-06 | 0.00326417 | 5.40158e-05 | none |
| 1/1 | 6 | random | 0.00336604 | 5.86594e-06 | 0.00326417 | 5.40158e-05 | none |
| 1/2 | 6 | zero | 0.00380039 | 7.6904e-06 | 0.00342011 | 5.47254e-05 | none |
| 1/2 | 6 | random | 0.00380039 | 7.6904e-06 | 0.00342011 | 5.47254e-05 | none |
| 1/4 | 6 | zero | 0.00375921 | 8.70263e-06 | 0.00317027 | 4.19198e-05 | none |
| 1/4 | 6 | random | 0.00375921 | 8.70263e-06 | 0.00317027 | 4.19198e-05 | none |
| 2/4 | 6 | zero | 0.00352529 | 7.79459e-06 | 0.00334593 | 6.42389e-05 | none |
| 2/4 | 6 | random | 0.00352529 | 7.79459e-06 | 0.00334593 | 6.42389e-05 | none |
| 4/4 | 6 | zero | 0.00340616 | 6.93236e-06 | 0.00326252 | 5.24651e-05 | none |
| 4/4 | 6 | random | 0.00340616 | 6.93236e-06 | 0.00326252 | 5.24651e-05 | none |
| 4/8 | 6 | zero | 0.00362597 | 8.9783e-06 | 0.00323372 | 6.11814e-05 | none |
| 4/8 | 6 | random | 0.00362597 | 8.9783e-06 | 0.00323372 | 6.11814e-05 | none |
| 32/32 | 6 | zero | 0.00381424 | 1.23798e-05 | 0.00396918 | 7.64001e-05 | none |
| 32/32 | 6 | random | 0.00381424 | 1.23798e-05 | 0.00396918 | 7.64001e-05 | none |
