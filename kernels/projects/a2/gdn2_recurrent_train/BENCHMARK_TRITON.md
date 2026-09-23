# GDN-2 a2 kernel vs fla Triton — benchmark comparison (Ascend 910B3)

Same operation (GDN-2 recurrence, `update = w·v − erase`, per-K-channel gate), same inputs, on the
same NPU. fla's Triton kernels run on the Ascend NPU via the integrated triton backend (no GPU).
Correctness is cross-checked against `fla.ops.gdn2.naive_recurrent_gdn2`; the a2 kernel and fla's
naive are the same op to **8.1e-5** rel-L2 (fp32 recurrence; fla's own chunk/recur match naive to
~2e-7). Conventions aligned: q/k are l2-normalized externally, `scale = 1/√128`; the a2 kernel takes
`q·scale` (it does no preprocessing), fla takes `q,k` with `scale=` (fla scales q internally).

## Wall time (µs/call, host `torch.npu.synchronize` loop, forward)

| shape (B,T,H) | a2 kernel | fla_recur (Triton) | fla_chunk (Triton) |
|---|---|---|---|
| (1, 64, 16)  | 508.5 | 341.9 | 4179.9 |
| (1, 128, 16) | 512.1 | 343.5 | 4298.3 |
| (2, 64, 16)  | 522.8 | 350.2 | 4227.2 |

## Device time (µs/call, msprof Task Duration, forward, 1×64×16)

| kernel | device µs/call | note |
|---|---|---|
| a2 `gdn2_fwd_states` | 179.1 | block_dim=40, VEC-bound aiv_vec_ratio 0.879 (see PERF_ANALYSIS.md) |
| fla_recur (Triton `fused_recurrent_gdn2`) | 163.7 | 220 ops / 100 calls |
| fla_chunk (Triton `chunk_gdn2`) | ≫ | chunk path far slower on this recurrence shape |

## Reading

- The a2 kernel **beats fla's chunk Triton ~8×** on this recurrent GDN-2 shape.
- Against fla's **fused-recurrent** Triton it is **competitive but behind**: ~9% slower on device
  (179 vs 164 µs), ~1.5× slower at wall time (the wall gap is the a2's aclnn host-dispatch floor,
  ~330µs, vs fla's ~178µs).
- Target to beat the benchmark: a2 device time **< 164µs** and VEC utilization **> 90%** (from 0.879),
  plus cutting the aclnn dispatch floor (NPU-Graph capture) for the wall-time win.

_Measured on Ascend 910B3, CANN 9.2.0-beta.1, torch_npu 2.10, ascriptor library 90cfcdc / kernels
b3b3f9c; fla pinned commit e52dbc0 (0.6.0) via the integrated triton-ascend backend._
