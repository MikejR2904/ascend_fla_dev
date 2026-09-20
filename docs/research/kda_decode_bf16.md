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
or NPU acceptance evidence. All BF-06 hardware stages are currently untested.

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
tests pass. These are static and host checks; vendor builds and NPU acceptance
remain separate requirements.
