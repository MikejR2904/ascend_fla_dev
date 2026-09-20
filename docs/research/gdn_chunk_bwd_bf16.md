# GDN grouped BF16 backward

Status: contract frozen before kernel implementation; native acceptance pending.

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
single slot. No Cube or mixed-pipeline overlap is introduced. Raw UB estimates
are checkpoint 66112, reverse 133504, group reduction 2560 bytes, pending IR.

## Frozen numerical budget

The formal ASSIGN5746479185 states 1e-2/3F; the task file still states
2e-2/4F. This implementation adopts the stricter **min(1e-2,3F_g)** against
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

## Required validation (pending)

Full B1/T4096/H=HV8 first, then ratios1/2/4/8 x C1/2/3/64, B2, all cotangent
modes, gate/beta/zero-input boundaries; independent leaves and composition,
NaN poison, unchanged input bytes, omitted-cotangent dummy poison, cross-bd
bytes, old/new FP32 bytes and actual public host-op audit for both dtypes.
Same-card complete old FP32 public backward / new BF16 / old FP32 sandwich,
T1024/4096, three rounds, 10 warmups and 50 synchronized samples per segment.
No speed threshold; no CUDA/Triton, weights or model-validation claim.
