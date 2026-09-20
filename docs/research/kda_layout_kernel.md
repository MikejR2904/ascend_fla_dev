# KDA layout and dtype conversion on A5

FMT-02 moves the existing public KDA layout/cast boundaries into the owned
`kernels/projects/a5/kda_layout` unit. The existing stable forward and backward
kernels, cache ABI, raw-input flags and forward/backward gate limits remain
unchanged. Implementation baseline: `f04047737ae91de8014a74c0e44fb31f7b3fba4a`.

## Comparison fixed before hardware execution

The layout/cast boundary must match the original public wrappers **bit for
bit**, including across block sizes 1, 2, 3 and 4. End-to-end comparisons include
public outputs, final state, nine saved checkpoints and six gradients. D-PM-44
qualifies the inherited scan h0 endpoint only on a device whose predecessor
public path repeats identically at least 12 times; every captured BF16 dh0
sample still requires exact three-way widening on every tested device. No
numerical tolerance substitutes for the conversion boundary requirement.
Existing CPU FP32 numerical budgets remain unchanged: forward relative L2 0.05;
backward default 0.05, `dk` 0.15, `dg` 0.25, with their existing elementwise
checks. The independent and pinned FLA references run on CPU FP32. Performance
uses actual NPU execution and the predecessor public path as its baseline.

The forward gate stays 155; the training gate stays 105. Out-of-domain historic
leaf cases remain out of the public domain and must exercise rejection rather
than bypassing either guard. CPU layout fallback is removed: `layout_device`
accepts the deprecated `auto`/`npu` aliases for the same custom kernel route;
`cpu` raises with the D-PM-37 constraint.

## Data movement and rounding

Four movement entries cover BF16/BF16, FP32/FP32, BF16/FP32 and FP32/BF16. Two
additional entries fill missing initial state and final-state gradient with zero.
A vector participant exclusively owns each contiguous output tile, containing
at most 4096 elements and always a multiple of 64. Source
coordinates come from explicit element strides, including noncontiguous and
broadcast upstream gradients; destination blocks are contiguous and disjoint.
The host supplies metadata, allocates outputs and launches. Same-dtype movement
preserves bits. BF16 widening is exact; FP32 narrowing uses ties-to-even.

The input is a metadata-only view of its backing storage, rooted at its existing
storage offset. Five-loop NDDMA gathers a tile into aligned UB. The host chooses only shape
metadata: each tile divides the logical shape, and an outer dimension grows only
when every inner dimension is complete. Thus every output interval is contiguous
and disjoint. BF16 unpack and FP32 narrowing still process 64 values per vector
iteration with the original even-lane packing. BF16 and FP32 buffers own 8192
and 16384 bytes respectively; casts use both (24 KiB total), single slot each.
`auto_sync` orders MTE2, vector conversion, MTE3 and buffer reuse. Zero fill
initializes one 4096-element UB buffer and stores a bounded final tile. Native
4160-element tests surround outputs with canaries and exercise the 64-element
zero tail. No overlap or double buffering is claimed. There are no cube allocations or inter-core synchronization edges.
Each transformation has one launch. Measured additional launches and cost are
reported below. After the initial device-cost measurement, PM requested the
within-scope batching optimization described here; there is no speed threshold.

## Registered arithmetic exceptions (D-PM-42)

These existing operations are separate from the layout migration and are not
claimed compliant:

- `chunk.py::_scan_states` as a whole: FP32 state/cache recomputation, including
  its casts, matrix products and stacking. Removing it requires the later kernel
  batch to emit `h` and `v_new` directly.
- The upstream branch's `log2(eg)`.
- `chunk_bwd.py`'s `dw = -d_vh`: there is no accompanying layout/cast at this
  boundary to fuse without introducing a separate arithmetic launch.
- Existing read-only domain/gate validation, separately classified as validation
  under the PM's provisional ruling, still pending user confirmation.
- Raw flags retain their existing preprocessing and are separately assigned to
  BF-07. FMT-02 introduces no new exception.

The stable gate's existing FP32 multiplication by the same rounded `1/ln(2)`
constant is fused into its layout/narrowing launch, as D-PM-42 permits. All
required native cases matched the exact predecessor checkpoint bytes.
There is no FMA or change to the materialization/rounding sequence.

Eight caches remain token-major because the actual backward GM contracts consume
that ABI; `h` remains `[B,C,HV,128,128]`. Existing backward entries already produce
the public gradient layouts, so no extra gradient transpose is introduced.

## Qualification and source identity

Completed on A5, CANN **9.1.0-beta.1**, compiler **20260509_173000235**,
OPP directories `ascend910_93`, `ascend910b`, **`ascend950`**; Python 3.12.14,
Torch 2.12.0+cu130, torch_npu 2.12.0. The selected read-only library is
`90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5`; kernels are
`b3b3f9c16df7c4626ed3c081032a1be5a753d0b1`. These results do not qualify the
previous CANN 9.2.0 environment or A2/A3. Private configuration supplies all
machine paths and device mapping. `evidence/native-manifest.json` associates
every original receipt/log with this environment, its original hash and its
sanitized hash. No numerical values or warnings were removed.

| Stage | Executed result |
|---|---|
| CPU reference | 27 contract cases passed |
| IR / CCE emission | All six new entries passed; emitted file hashes retained |
| Vendor compilation | All 22 dependencies at each bd=1,2,3,4, including all nine backward entries, before the process's first custom call |
| Native complete workload | dev-B: 116 plain/cached/backward cases per bd, 464 total |
| Native additional checks | Per bd: 31 layout leaves, five autograd gradient views, six gate checks, 24 stable contract checks, two decode domain checks |
| Cross-bd comparison | All 3358 input/output hashes per bd identical |
| Batched-tail checks | Six native canary cases per bd, 24 total; signed zero, rounding ties, subnormals, copy/cast and zero tail |
| Source sim / pipesim | 15 bounded diagnostics per stage, bd1, 128–1024 elements, only after the full native workload passed |
| CANN simulation | Not run |
| Host regression | Accepted device Python: 862 passed, no skips; local final suite 862 passed / five hardware skips; final focused suite 94 passed |

Decode retains its original supported dimensions: bd3's two dtype cases reject
before any launch; they are rejection checks, not native decode execution.
The CPU-only regression explicitly excluded hardware test files; it does not
substitute for native verification.

The first custom execution was the full Kimi B1/T4096/H=HV32 workload, including
plain forward, all nine saved caches and the complete nine-kernel backward.
Each native case compares 19 predecessor outputs, independent and pinned FLA
CPU FP32 references, each head/chunk output and state, all six gradient budgets,
unchanged inputs and actual outputs poisoned before their kernel writes. The
116 cases include the original five backward distributions, B2 with
C=1/2/3 and HV/H=1/2/4/8 for zero/nonzero state, Kimi T1024/T4096, exact gate
span 105 and the original 84 public grid cases (336 across bd). This preserves
even-C coverage as well as odd-C and repeated heads. At the full T4096 shape,
independent-reference relative L2 was 0.0032664 (o), 0.0025472 (state), 0.030350
(dq), 0.034513 (dk), 0.0035411 (dv), 0.0033178 (dbeta), 0.057061 (dg) and
0.0023428 (dh0). Existing 3F diagnostic exceedances remain in the receipts;
no acceptance budget changed.

All runtime audits had no unexpected operator. The registered exceptions remain
**not compliant**, with concrete source spans and operators:

| Exception / owner | Source | Observed operations |
|---|---|---|
| State/cache recomputation; later `fwd-caches-not-emitted` kernel batch | `chunk.py:630–636`, whole `_scan_states` | `_to_copy`, `matmul`, `sub`, `add`, `mul`, `stack`, plus metadata views |
| Existing dw sign; D-PM-42 exception | `chunk_bwd.py:286` | `aten.neg.default` |
| Upstream gate; D-PM-42 exception | `chunk.py:712` | `aten.log2.default` (upstream branch; not used by stable grid) |
| Read-only gate/domain validation; provisional D-PM-42 classification | `chunk.py:90–140`, `392–420`; runtime gate audit at 400–401 | `cumsum`, `amin`, `amax`, `sub`, `max`, scalar extraction |

The raw-input flags keep their existing BF-07 scope and are not claimed compliant
here. No stable forward/backward kernel, upstream file, resource cap, gate or
gradient budget was modified. Compiler output retains existing `[perf]` ND2NZ
burst and even-stride UB-bank warnings in unchanged kernels; these are performance
warnings, not a repaired issue or evidence of the scan failure's mechanism.
Torch NPU also reports required 32-byte allocation padding; allocation and
output canary checks passed, and the warning is retained in the full logs.

## Inherited scan limitation and D-PM-44

The read-only scan's BF16 dh0 output remains nondeterministic on **dev-A**.
The new qualification repeated the unchanged predecessor public path 12 times:
three h0 hashes, with frequencies 9/2/1; three runs exceeded the unchanged 0.05
CPU FP32 budget (maximum 0.358643). The candidate's 12 repetitions also had
three hashes (10/1/1). Both paths' other seven public outputs/gradients were
identical across all 24 runs. Under D-PM-44 this h0 endpoint is recorded as the
inherited baseline defect, neither a candidate pass nor a candidate failure.
Complete h0 endpoint qualification is limited to **dev-B**: its 12 old and
12 candidate public runs had one identical h0 hash, all other seven endpoints
were identical, and the h0 relative L2 was 0.00233687 against the unchanged
0.05 budget. Its 12 direct scan repeats also had one dh0 hash.

The public seed manifest identifies all eight public inputs, the generator and
all eight scan inputs. The replay rebuilds them in the recorded environment and
verifies every byte hash before launching. Its direct scan loop runs 12 times
without any new layout operation inside that loop. dev-A again had seven dh0
hashes, while its dAqk, dh and dv were identical. This observation does not locate
the defect mechanism or establish a repair.

All 36 historical BF16 dh0 samples, including every unstable observation, are
kept in private acquisition archives and identified publicly by hashes and
statistics. The new widening kernel is checked against old Torch NPU and CPU
FP32 conversion on those same tensors on both devices (72 comparisons), plus
all 12 newly captured direct-scan samples on each device (24 comparisons).
Inputs remain unchanged. No sample is dropped because its h0 endpoint is unstable.
The original candidate's raw numerical receipts, failures and performance are
retained under `evidence/history/v1/`; those do not qualify this batched source.

The unresolved gap is `kda-bwd-scan-dh0-nondeterministic`. Repair requires a later
approved kernel batch. FMT-02 does not change scan or investigate it further.
PM manages confirmation of D-PM-44 before merge; this report does not claim that
the baseline defect is fixed.

## Synchronized performance

The first table is the retained original 64-element candidate, **not the current
batched implementation**. Current measurements and event attribution follow it.

Same dev-B, bd4, Kimi B1/H=HV32/K=V128, BF16 data and existing FP32 gate/state;
one warmup then three **old / candidate / old** synchronized rounds. Values below
are the median of three candidate calls and six bracketing old calls, in ms.
Input preparation and compilation are excluded. Backward consumes existing
caches; it is not the latency of a complete training step.

| T | Public call | Old ms | Candidate ms | Candidate / old | Extra layout launches |
|---|---|---:|---:|---:|---:|
| 1024 | Plain forward | 1.563872 | 48.547256 | 31.043× | 6 |
| 1024 | Cached forward | 2.410407 | 122.146424 | 50.675× | 14 |
| 1024 | Backward, existing caches | 2.480316 | 2.578439 | 1.040× | 1 |
| 4096 | Plain forward | 5.694340 | 202.023412 | 35.478× | 6 |
| 4096 | Cached forward | 9.277456 | 477.483918 | 51.467× | 14 |
| 4096 | Backward, existing caches | 9.767145 | 10.371944 | 1.062× | 1 |

That initial implementation had a substantial forward performance regression. Its native layout kernels
move 64 elements per vector work item; per-layout serialized synchronized costs
are retained separately in `performance.json`. They are diagnostic timings and
must not be summed as an alternative end-to-end measurement. This task establishes
kernel-side layout/dtype behavior, not performance improvement.

A separate actual **Torch NPU FP32 recurrent** baseline also ran three synchronized
sandwich rounds. T1024 median before/candidate/after was
68.835940 / 47.845303 / 66.145383 ms; T4096 was
491.925117 / 202.166867 / 486.737733 ms. This is a differently implemented,
FP32 reference baseline; CPU FP32 remains the correctness golden. Neither this
comparison nor old A5 task results replace the predecessor-path regression above.

## Batched candidate performance and device attribution

Same protocol and device, new source identity; the values below are fresh measurements.

| T | Public call | Old ms | Batched ms | Batched / old | Extra launches |
|---|---|---:|---:|---:|---:|
| 1024 | plain_forward | 1.580521 | 2.639631 | 1.670× | 6 |
| 1024 | cached_forward | 2.391278 | 5.391900 | 2.255× | 14 |
| 1024 | backward_existing_caches | 2.476485 | 2.510197 | 1.014× | 1 |
| 4096 | plain_forward | 5.534615 | 10.259895 | 1.854× | 6 |
| 4096 | cached_forward | 9.218042 | 21.352520 | 2.316× | 14 |
| 4096 | backward_existing_caches | 9.760661 | 9.794137 | 1.003× | 1 |

The batched kernel substantially reduces the original candidate’s cost, while
plain and cached forward remain 1.67–2.32× slower than the predecessor public
path. Backward here consumes existing caches and remains close to the old path.
The task has no speed threshold; these regressions are explicitly retained.

Actual Torch NPU FP32 recurrent baseline, median before / candidate / after (ms):

- T1024: 67.225624 / 2.687383 / 67.090231.
- T4096: 511.457727 / 10.494397 / 512.437496.

The clean attribution sandwich adds no internal synchronization or event
instrumentation. A separate invocation records events on the same stream and
synchronizes once at the end. Host dispatch excludes event recording and device
completion. A third diagnostic counts hot-call source/hash/compile/registration
and Path read/traversal operations; all counts are zero over three calls.

| Version | T | Call | Clean candidate median ms | Layout device-event sum ms | Layout host-dispatch sum ms |
|---|---|---|---:|---:|---:|
| original-64 | 1024 | plain_forward | 47.015558 | 45.692225 | 0.473852 |
| original-64 | 1024 | cached_forward | 119.964190 | 117.819347 | 0.906107 |
| original-64 | 1024 | backward_existing_caches | 2.584898 | 0.135061 | 0.033520 |
| original-64 | 4096 | plain_forward | 202.400237 | 197.116210 | 0.208553 |
| original-64 | 4096 | cached_forward | 478.271309 | 469.605826 | 3.039210 |
| original-64 | 4096 | backward_existing_caches | 10.393996 | 0.637150 | 0.030836 |
| batched-4096 | 1024 | plain_forward | 2.617498 | 1.222200 | 0.310615 |
| batched-4096 | 1024 | cached_forward | 5.387604 | 3.293199 | 0.630465 |
| batched-4096 | 1024 | backward_existing_caches | 2.518489 | 0.039734 | 0.031097 |
| batched-4096 | 4096 | plain_forward | 10.237362 | 4.997096 | 0.163817 |
| batched-4096 | 4096 | cached_forward | 21.365180 | 12.984622 | 1.327821 |
| batched-4096 | 4096 | backward_existing_caches | 9.842349 | 0.042189 | 0.033680 |

Every layout launch is listed in
[`per-launch-costs.md`](../../kernels/projects/a5/kda_layout/evidence/per-launch-costs.md).
The initial cost was predominantly device work, not cache lookup or repeated
registration. At 16M elements, batching reduces 262144 work items to 4096
and, with eight vector participants, 32768 items per participant to 512.
Traffic, arithmetic, rounding and participants are unchanged; each DMA/store
moves more values. The maximum live UB footprint is 24 KiB. No throughput
or overlap claim is inferred from model cycles or summed diagnostic timings.

## Evidence and replay

All paths below are relative to `kernels/projects/a5/kda_layout/`:

- `evidence/native/`: 1131 current numerical receipts and unfiltered sanitized
  logs, compile hashes and environment records. `native-manifest.json` records
  every source/sanitized hash and runtime environment.
- `evidence/history/v1/`: original candidate sources, all earlier numerical
  observations, initial dependency/harness and dev-A failures, performance
  records, and their original/sanitized hashes. No historical pass qualifies
  the new batched kernel.
- `evidence/repro-bundle/seed-manifest.json`: seed, generator identity and
  fixed public/scan input hashes, plus every historical BF16 sample hash and
  statistics. `raw-tensor-manifest.json` identifies private acquisition files
  and their logical tensors. No tensor-value blob or pickle input is committed.
- `evidence/model/`, `evidence/host/`, `evidence/emission/`: bounded model results,
  host/reference logs and emitted-artifact hashes, kept distinct from hardware.
- `baseline/`: byte-preserved predecessor wrappers and their source receipt.

After selecting the recorded Python/library/kernel identities, holding both
shared device locks and configuring ignored output paths, run each bd in a
separate process. Every verifier process precompiles all dependencies:

```bash
python verify_native.py --block-dim 4 --mode suite --output "$FMT02_OUTPUT/suite-bd4"
# Repeat in independent processes for bd 1, 2 and 3.
python aggregate.py "$FMT02_OUTPUT"
python verify_native.py --block-dim 4 --mode perf --output "$FMT02_OUTPUT/perf-bd4"
python replay_scan_bundle.py --bundle evidence/repro-bundle/seed-manifest.json \
  --device-label dev-A --output "$FMT02_OUTPUT/replay-dev-A"
# Repeat on dev-B after a fresh health/occupancy/lock check.
python verify_evidence.py
```

`replay_scan_bundle.py` rebuilds inputs from the seed and refuses any byte-hash
mismatch before scan execution. Its direct scan loop has no layout call;
conversion checks run afterwards. `profile_dispatch.py` is byte-identical to
the actual attribution acquisition helper: `FMT02_NATIVE_ROOT` selects an
ignored task root containing `repo/` and output directories. Its clean sandwich
has only public-call boundary synchronization; event tracing and hot-path IO
counters run separately. Per-launch event times are diagnostic intervals,
not an alternative end-to-end latency measurement.

The older `diagnose_public_repeat.py` and `diagnose_scan_repeat.py` retain the
original capturing method. `verify_evidence.py` uses only the standard library
to recompute comparisons, unchanged budgets, cross-bd and source identities,
repeat distributions, every conversion check and performance medians from text
receipts. Historical qualification is explicitly separate. Machine details
remain in ignored external configuration.
