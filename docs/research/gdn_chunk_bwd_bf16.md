# GDN grouped BF16 backward

Status: both complete native grids pass, 276/276 records with exact cross-bd
bytes. Same-card timing and the isolated runtime-warning control are pending.

Task BF-02, issue #101; assigned session gdn-series-20260919T115559Z-29cb31d7.
Base e10e477b8b6fd6234a63f0aa9a7692c766699d63; original FP32 unit remains read-only.

## Public ABI and scope

Token-major contiguous q/k/v/do have one matching dtype, FP32 or BF16.
q/k are [B,T,H,128], v/do are [B,T,HV,128], HV is a positive multiple of H.
B/H/HV positive; T is a multiple of 64 in [64,4096]; A5 CCE, block_dim 1 or 2.
g/beta are FP32 [B,T,HV], dht is optional FP32 [B,HV,128,128].
At least one of do/dht must be present. Missing cotangents contribute zero.
Return dq/dk [B,T,H,128], dv [B,T,HV,128] in input dtype; dg/dbeta FP32.
Zero initial state only, no dh0, normalization, variable length, CP, head-first
or transposed state; scale=128**-0.5. Finite inputs, g<=0, beta in [0,1].
CPU diagnostic launchers retain read-only value checks under the PM common rule.
NPU production inputs retain caller preconditions and metadata checks.

## Arithmetic and decomposition

D=exp(g)*Sprev; r=v-k^T D; S=D+k*(beta*r)^T; o=(scale*q)^T S.
The reverse pass differentiates <do,o>+<dht,Sfinal>. q/k gradients sum all
consecutive value-head contributions in FP32 before their final BF16 store.

Three BF16 launches: boundary checkpoints, replay plus reverse adjoint,
ordered group reduction. One vector worker owns a complete (B,HV) recurrence;
group reduction owns each (B,T,H) row. No atomics or state inverses.
BF16 GM input rows widen to FP32 registers; primal/adjoint math, checkpoints,
64-token replay tape, dS and per-value-head dq/dk stay FP32. dv is narrowed
only on final publication, preserving its FP32 register for later adjoint math.
dq/dk narrow only after every group contribution. Final rounding is RNE.
Missing cotangents use full-shaped unread dummy buffers and presence flags;
the kernel clears its corresponding UB state/row. Host only allocates/launches.
The existing FP32 unit and its ordered arithmetic remain unchanged.

Checkpoint/tape publication uses the original explicit MTE3-to-MTE2 event;
Pipe.ALL retires each chunk before tape reuse. UB autosync protects each DMA/VF
single slot. No Cube or mixed-pipeline overlap is introduced. Lowered UB allocations
are checkpoint 66112, reverse 133504, group reduction 2560 bytes. All three
entries pass full lowering and event-balance checks and emit CCE source.

## Frozen numerical budget

The formal ASSIGN5746479185 and PM confirmation5746857760 freeze
**min(1e-2,3F_g)** against
each oracle, for every returned BF16-path gradient including FP32 dg/dbeta.
F_g is relative L2 between that FP32 oracle gradient and its BF16 roundtrip.
Each oracle supplies its own per-case floor. Zero F requires exact equality;
a zero reference has relative error zero iff the actual tensor is zero,
otherwise infinity. No epsilon relaxes a zero budget. max_abs is also reported.
FP32 public outputs and FP32 internal stages require relative L2 <=1e-4;
old/new FP32 public outputs additionally require byte identity.

A is literal pinned FLA naive autograd at e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2,
SHA256 d1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb.
B is an independent analytical adjoint. Fresh FP64 gradchecks for A precision
lift and B analytical backward passed ratios 1/2 in the recorded calibration.
Only q/k/v/do are BF16-rounded before both FP32 oracles; g/beta/dht stay FP32.

The pre-implementation reference investigation has 138 records: 23 cases x
3 cotangent modes x 2 input-storage choices. Source hashes were reverified
unchanged against the assigned base before adopting this calibration. It is
budget calibration only, not device acceptance. Native jobs generate fresh
inputs and independent references; no recorded tensors serve as fixtures.

| Gradient | Nonzero BF16-input F range (A/B) |
| --- | --- |
| dq | 0.00161668963756 .. 0.00168679954811 |
| dk | 0.00140734904703 .. 0.00183565901777 |
| dv | 0.00148141637081 .. 0.00177865508659 |
| dg | 0.000912684260435 .. 0.00293019382798 |
| dbeta | 0.000778698227521 .. 0.00323516643593 |

Raw values: evidence/calibration-pre-kernel.json. A/B max relative L2 is
5.599113205882739e-7; all six nonzero-output negative controls rejected.
The additional zero calibration rejected 36 actual nonzero perturbations
across ten cases, including noncontiguous oracle gradients.

## Validation matrix

Full B1/T4096/H=HV8 first, then ratios1/2/4/8 x C1/2/3/64, B2, all cotangent
modes, gate/beta/zero-input boundaries; independent leaves and composition,
NaN poison, unchanged input bytes, omitted-cotangent dummy poison, cross-bd
bytes, old/new FP32 bytes and actual public host-op audit for both dtypes.
Same-card complete old FP32 public backward / new BF16 / old FP32 sandwich,
T1024/4096, three rounds, 10 warmups and 50 synchronized samples per segment.
No speed threshold; no CUDA/Triton, weights or model-validation claim.

## Implementation validation

The implementation preserves partial dq/dk in FP32 through the final group sum.
Full lowering caught an initial authoring mismatch (BF16 temporary UB to FP32
partial-gradient GM); the two temporary buffers were corrected before any
hardware execution. The original failed source and lowering log are retained
in ignored task scratch. A host regression now lowers all three typed stages.

The focused public ABI, mixed-dtype rejection, independent-reference,
zero-floor, negative-control and lowering suite passes 88 tests. Native
same-card timing and isolated startup-warning qualification remain pending; no model or CUDA/Triton result
is used as device evidence.

CPU-isolated repository suite: 763 passed, 9 skipped, 5 existing importorskip
deprecation warnings (pytest8.3.2). Hardware tests run separately under the
device lock; these skips are not device acceptance. The initial unisolated
host invocation was interrupted after unrelated KDA device tests failed, and
is retained in ignored scratch; no test, requirement or tolerance was relaxed.

## First full native workload

B1/T4096/H=HV8/K=V128, block_dim2, both cotangents. Actual BF16 and FP32
public outputs pass fresh pinned CPU autograd A and independent analytical B,
NaN-poisoned composition and independently supplied stage inputs. Inputs are
unchanged and original/new FP32 public outputs are byte-identical. The public
host audit records exactly10 `aten.empty.memory_format` calls for each dtype.

BF16 dq/dk/dv relative L2 is approximately0.00163–0.00168, below each frozen
budget; FP32 public A/B max relative L2 is2.4538954e-7. This first case has
HV/H=1 and both cotangents; it does not establish grouped or omitted-cotangent
coverage by itself. The broader grid is a separate required run.

The actual environment is Ascend950PR_9589 V100, CANN9.2.0 with the ascend950
operator package, Python3.12.13, Torch2.12.0+cpu, torch_npu2.12.0, and the pinned
Ascriptor source. Shared-lock occupancy evidence contains202samples,
182positively observing both installed-driver context registries and matching
the exact compute child. No foreign context was observed; before/after were
empty and the device remained healthy. Machine identities are kept privately.

See the unit's `evidence/full-v4-bd2*` files for raw numbers/logs, source hashes,
environment and occupancy; `canonical-reference-summary.json` for138/138CPU
reference records; and the six bd2 vendor build logs/artifact manifest plus
`build-warning-resolution.md` for actual compilation and warning attribution.
The three new entries emitted and compiled without vector-loop-cond warnings.

## Complete native grids

All 69 cases pass for both actual BF16 and FP32 public paths at each of
block_dim1/2: 276 records. Coverage includes ratios 1/2/4/8, chunk counts
1/2/3/64, B2, three cotangent modes, and gate/beta/zero boundaries. Every row
includes fresh A/B comparisons, poisoned composition and independent leaves,
input byte hashes and the actual public host-op trace.

All 138 cross-bd case/dtype pairs match exactly for ten stage arrays (1,380
array pairs), including all five public returned gradients. The 138 FP32
rows also match the unchanged original public implementation byte for byte
(690 returned-array pairs). There are 392 exact zero-budget comparisons;
no tolerance was changed. The public-output counts overlap the stage counts.

Worst BF16 public errors across both grids are below. Each column is an
independent maximum; raw-record locations and budget fractions are in
`evidence/public-gradient-metrics.json`. Every individual comparison satisfies
its own frozen oracle-specific budget, including FP32 dg/dbeta.

| Gradient | A relative L2 | B relative L2 | A max absolute | B max absolute |
| --- | ---: | ---: | ---: | ---: |
| dq | 0.0016866822 | 0.0016866822 | 3.0506402e-05 | 3.0506402e-05 |
| dk | 0.0018356689 | 0.0018356689 | 0.0009700954 | 0.0009700954 |
| dv | 0.0017786248 | 0.0017786248 | 0.00048822165 | 0.00048822165 |
| dg | 8.2883566e-07 | 8.1444112e-07 | 4.1723251e-07 | 4.1723251e-07 |
| dbeta | 9.5043963e-07 | 9.5043963e-07 | 1.1920929e-07 | 1.1920929e-07 |

FP32 public A/B worst relative L2 is 1.0802102e-6, below 1e-4. BF16 host
traces contain only `aten.empty.memory_format`; FP32 additionally uses the
unchanged `aten.zeros.default` / `aten.zeros_like.default` allocations for
absent cotangents. Each dtype has 46 records for each cotangent mode.

The shared lock covers each whole run and its context drain. bd1 has 752
occupancy samples (739 positively matching the exact compute child in both
driver registries); bd2 has 615 (602 positive). Neither has a foreign context;
before/after are empty and post-run health passes. See `evidence/grid-v1-bd1*`,
`evidence/grid-v1-bd2*` and `evidence/native-grid-summary.json`.

## Environment qualification boundary

The primary toolkit's compiler/OPP timestamp is `20260805_101249091`. Fresh
read-only hashes of all 205 tracked library package files/resources match
clean pin90cfcdc, including CCE headers and build templates. Actual toolkit
header sites/hashes and component versions are recorded in
`evidence/library-toolkit-identity.json`.

A supplementary A5 environment with a later CANN9.2.0 build fails before
custom-kernel execution: its changed header guard collides with the pinned
library's `g_coreType` fallback, including for the unchanged FP32 checkpoint.
No pin, generated bundle or vendor source was patched. This secondary vendor
build remains unqualified; complete failure logs and the exact located cause
are in `evidence/secondary-build-failure.md` and accompanying JSON/logs.
