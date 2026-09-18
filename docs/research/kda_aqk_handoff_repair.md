# KDA Aqk handoff repair: A5K-01

The derived stable unit repairs the odd-chunk, repeated-head Aqk handoff defect.
All 336 native grid cases pass both CPU FP32 oracles. Repaired stage tensors and
every chunk state are byte-identical across block_dim 1, 2, 3 and 4; all even-C
cases are byte-identical to the original baseline. This is a correctness repair;
the timing experiment does not establish a consistent speed gain over baseline.

This report covers [A5K-01](https://github.com/ddddwee1/ascend_fla_dev/issues/76),
the narrow D-PM-21 exception derived from
[A2-04 / PR #60](https://github.com/ddddwee1/ascend_fla_dev/pull/60).
A5K-01 qualified the standalone unit without changing public dispatch.
The integration candidate proposed in [request #79](https://github.com/ddddwee1/ascend_fla_dev/issues/79)
selects that same repaired source for public stable forward and cached-forward.
The original upstream path rejects odd C with B*HV > block_dim; the old
claim that all C>=2 are safe was disproved by the C=3 silicon result below.
The stable first four stages plus original recurrent remain an explicitly
pinned negative control. Public integration validation is pending; the
A5K-01 numbers below qualify the earlier standalone composition only.
A5-01 through A5-06 remain gated on W-A3.

The initial integration draft passed 337 host tests (5 skipped), including 109 scoped
gate/dispatch checks, and 8 canonical CPU reference cases. Source emission passes
all five forward kernels, the original recurrent control and eight of nine
backward dependencies. The unchanged upstream `inverse_mm_kernel` is refused:
its cube side needs 34 local mutex IDs while the accepted limit is 32. Because
the real cached-forward API precompiles backward vendors before its first launch,
this is a cached-path blocker at the accepted pins. Neither precompilation nor
synchronization checks have been bypassed. Public native grid and timing remain
unqualified; the measurements below still describe A5K-01 only.

D-PM-24 approves a narrow local `inverse_mm` derivative and stable selector
update. The derivative changes only `l0c_dvh` and `l0c_dvbeta` from double buffers
to single-slot L0C tensors, with corresponding direct views. Emitted local mutex
counts become cube=32 and vector=15. M writes and every FIX read retain mode-zero
ownership; the final FIX release must retire before the next iteration's M write.
The three-slot cross-side buffers, event credits, two-work lookahead, drain and
all arithmetic remain unchanged. Upstream source and the 32-ID limit stay intact.
This is an emission result, with native leaf/backward and synchronization regression
still pending. `verify_kda_aqk_public.py inverse` first tests the full B1/T4096/HV32
leaf against a CPU FP32 reference, then single-item, repeated-slot and uneven-work
cases; public cache/grid verification remains a separate required check.
The existing backward inventory test caught the required contract amendment:
the derived `inverse_mm` must be listed as owned rather than shared. PM confirmed
that metadata write set, and the inventory now matches the selector. The gate is
retained unchanged. The full host suite now passes 338 tests with 5 skips, including
the 20 focused source-selection, safety and compiler-budget regressions. The
complete selected forward/backward chain also vendor-compiles
at block_dim=4; builds for the remaining block counts are in progress.

## Source and hardware scope

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

## Repair and lifecycle argument

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

## Native correctness

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

## Synchronization regression and host checks

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

## Same-device performance

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

## Reproduction and retained evidence

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

## Grid projection

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
