# BF-06: native BF16 KDA decode

## Precision contract fixed before kernel implementation

The public task covers direct token-major BF16 q/k/v and BF16 output with
FP32 gates, beta and recurrent state, plus an unchanged-result FP32 route.
The original FP32 unit remains read-only. Both dtype routes must perform
scaling, conversion and layout addressing inside the new custom kernel.

The CPU reference identities are FLA v0.5.2
`9c8e42e762fce087c27b673af4922795d9edb85e`, KDA naive SHA256
`60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016`,
and the independent `ascend_fla.reference.kda.kda_recurrent_ref`.
Both receive FP32 tensors containing the actual BF16-rounded q/k/v values.
Initial state, gates and beta are FP32 throughout each reference.

The per-case output rounding floor is
`F = ||BF16_RNE(o_ref) - o_ref||₂ / ||o_ref||₂`, computed separately for
each oracle. The fixed output budget is relative L2 `<= min(1e-2, 3*F)`;
the FP32 final-state budget is `<= 1e-5`. A zero reference requires an
exact zero residual. These limits will not be relaxed after implementation.
The ASSIGN reminder's looser state wording does not replace the task's
explicit `1e-5` FP32 state limit.

The pre-implementation study contains 82 cases: T={1,2,16}, GVA={1,2,4,8},
both initial-state options at B1/H1 and B2/H2; every intermediate T=3..15;
H32 and HV32 workloads; zero/tiny/deep/underflowing negative gates and
strong-then-weak patterns; nondefault/zero/negative scales; beta endpoints;
and all-zero inputs. Inputs and both goldens are generated at run time.

| CPU experiment | Passed | Worst output relative L2 | Worst state relative L2 |
| --- | ---: | ---: | ---: |
| BF16 GM values, FP32 scale and ordered FP32 recurrence, BF16 output | 82/82 | 0.001884841312 | 1.026706921e-7 |
| Round recurrent state to BF16 after each token (negative control) | 1/82 | 0.004637969152 | 0.006644053720 |
| Also round q*scale in BF16, as the old BF16 wrapper did | 82/82 | 0.002654213112 | 1.026706921e-7 |

The two independent CPU references agree for this study. Nonzero output F
ranges from 0.001616102595 to 0.001884841312. The selected design widens BF16
q/k/v in the kernel, scales q in FP32, keeps the two ordered K loops and the
entire state in FP32, then directly stores BF16 output with ties-to-even.
This deliberately removes the old BF16 q*scale rounding; the FP32 route's
arithmetic order and scaling precision must remain unchanged.

Reproduce with the accepted CPU environment, this repository on PYTHONPATH,
and the exact read-only FLA source supplied explicitly:

```bash
python kernels/projects/a5/kda_fused_recurrent_bf16/research.py \
  --fla-naive /path/to/pinned/fla/ops/kda/naive.py \
  --output tmp/bf06-cpu-study.json
```

The original complete result is in `evidence/cpu-precision-initial.json`.
This is CPU calibration, not source-emission, vendor-compilation, simulator
or NPU acceptance evidence. Subsequent hardware results are recorded below.

## Owner scope ruling

PM's issue #105 comment 5746501164 resolves the conflicting raw-flags clause:
default flags=False must pass both dtype host audits; the existing A2-44
raw-flags preprocessing stays unchanged as an explicitly recorded exception
pending BF-07, with a separate audit category and no conformance claim.
The default public dtype combinations are BF16 q/k/v with FP32 g/beta/state,
or all FP32. FP16/FP64 and unsupported mixed combinations are rejected with
the failing constraint. Existing FP32 block dimensions {1,2,4,8,16,28} remain;
1/2/4 need the complete grid and 8/16/28 need complete workloads plus boundary
cases, with old/new FP32 byte comparisons at all six dimensions.

Every native run must record SoC, CANN version and OPP package directories
in its original receipt. Public evidence is text; the previous task's binary
archive exception does not apply here.

## Implementation and static checks

The new BF16 and FP32 entries directly address token-major GM. Each vector
participant owns a whole FP32 state for each assigned head, with no cross-core
communication. The BF16 entry uses 123392 bytes of UB; the FP32 entry uses
107008 bytes. Both keep the original increasing-K accumulation order.
An absent initial state is initialized in the kernel. BF16 widening, q scaling,
GVA head selection and output conversion all execute inside the custom kernel.

Initial source inspection caught an implicit DMA inference problem: slicing
`q[b, 0:T, h, 0:128]` emitted zero row gap despite the intervening head axis.
The selected library rejects symbolic-shaped `.view()` composition. The final
source therefore uses the documented explicit padded-DMA APIs with T bursts,
128 payload elements, GM gaps `(H-1)*128` or `(HV-1)*128` elements, and zero UB
gaps. Generated pointers and byte-scaled gaps were inspected for both entries.
The scalar beta rows have 32-byte UB pitch and consume only the initialized
first scalar. Neither library nor generated code was modified.

Both entries pass IR lowering, event-balance checks and CCE source emission.
The default public wrapper passes a CPU proxy audit containing only two empty
allocations, with actual returned buffers, unchanged input pointers and dtype
checks. Together with the existing decode and raw-input regressions, 99 host
tests pass. The subsequent full host suite passed 768 tests with 5 skips.
The standalone unit passed all 83 CPU reference cases without importing the
DSL, custom kernels or simulator. These are static and host checks; vendor builds and NPU acceptance
remain separate requirements.

## Prefill/decode comparison fixed before its implementation

PM issue #105 comment 5746672559 defines three separate checks. Each decode
call starts both CPU FP32 oracles from the actual state it received, retaining
the original `min(0.01,3F)` output and `1e-5` state budgets. Every 16-token
segment must produce identical output and state when split into sixteen
single-token calls.

For prefill64/128 followed by decode64, run the existing stable long forward
on the same inputs in the same hardware run. Against each full-sequence oracle,
the complete chain's output and final state must meet `min(0.01,3*E_oneshot)`,
where each output's `E_oneshot` is that long forward's measured relative L2.
Prefix and suffix outputs are reported separately and obey the same output
limit. Prefix state is also checked against its own CPU prefix reference with
the same state limit. Chain-vs-oneshot errors are reported without a separate
threshold. The old chunk contract's 0.05 limit is not used for this acceptance.
These rules are fixed before executing the prefill measurements.

## Completed decode grid; integration and performance pending

The selected kernel source has SHA256
`c4e341a99c005259e0c9bfb59541ba192ac177d4da648862c1a68f7c8b3cea14`.
Native execution used Ascend950PR_9589, CANN compiler/OPP 9.2.0 build
`20260805_101249091`, Python 3.12.13, Torch 2.12.0+cpu with torch_npu 2.12.0,
and Ascriptor 0.1.0 at library `90cfcdc` / kernels `b3b3f9c`.
The Torch version suffix describes the base package; execution used actual NPU
tensors through torch_npu. CPU FP32 goldens were generated on the same host.

Every independent block-dimension process compiled and registered 17 vendors,
including all nine backward entries, before its first custom execution. The
first full workload was B2/T16/H4/HV32 with nonzero state. Its BF16 output error
was 0.001656522640 and state error 1.525832876e-7. Compilation of backward
dependencies does not constitute backward execution.

| Decode block dimension | Numerical cases | Additional checks | Result |
| --- | ---: | ---: | --- |
| 1, 2, 4 (each) | 136 | 57 | Passed |
| 8, 16, 28 (each) | 2 complete BF16/FP32 workloads | 57 | Passed |

The complete grid includes all 82 BF16 calibration shapes plus the initial
full workload, FP32 grid/real shapes, intermediate token lengths, grouped
heads, optional state, gate extremes and scale/beta endpoints. Additional
checks cover eight original unrounded FP32 shapes, optional returned state,
1/2/4/8-token segmentation, metadata rejection and all eight raw-flag combinations
for both dtypes. Raw preprocessing remains the separately audited exception.

The 414 numerical executions and 342 additional checks passed. Independent
replay checks the full compile manifests, both-oracle budgets, each head,
input hashes, NaN coverage and default host operations. All shared numerical
and 40 additional hashed cases have byte-identical outputs across block
dimensions. Worst BF16 output relative L2 is 0.001884841312; worst FP32 output
is 2.011265135e-7; worst global state is 1.711005737e-7 and per-head state
1.797956881e-7. Default-path audits contain only `aten.empty.memory_format`
and `aten.view.default`.

The grid used verifier commit `492127d` and the original read-only FP32 kernel
with its old input preparation. Subsequent prefill and performance modes use
the exact original public-wrapper snapshot `0f517ee` as well, including its
metadata, allocation, scaling and layout costs. Those modes remain unexecuted
at this checkpoint and must pass before BF-06 is complete.

After the first complete native workload, bounded BF16 sim/pipesim probes
passed at T1/H1/G1 without state and T2/H1/G4 with state; the latter retains
token row strides and repeated-head buffer reuse. The FP32 companion passed
T2/H1/G4 without state. All use bd1. Pipe-model cycles were 5658, 37248 and
24052 respectively, with empty event-balance/hazard lists and no deadlock.
These diagnostics do not qualify the whole native grid or measure device time.

The final host suite passed 768 tests with 5 skips. Public text receipts,
complete native logs and an independent replay are under the unit's
`evidence/`. The 23 warnings per native build are retained: 20 existing
forward-dependency ND-to-NZ transfer performance diagnostics and three
backward-dependency UB bank-stride performance diagnostics. Their owner is
`ascriptor/ir/lint.py`; they do not describe a new decode-kernel correctness
or synchronization failure. Neither dependencies nor warning controls changed.
