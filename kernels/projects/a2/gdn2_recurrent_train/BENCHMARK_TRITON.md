# GDN-2 a2 kernel vs fla Triton — benchmark comparison (Ascend 910B3)

Same operation (GDN-2 recurrence, `update = w·v − erase`, per-K-channel gate), same inputs, on the
same NPU. fla's Triton kernels run on the Ascend NPU via the integrated triton backend (no GPU).
Correctness is cross-checked against `fla.ops.gdn2.naive_recurrent_gdn2`; the a2 kernel and fla's
naive are the same op to **8.1e-5** rel-L2 (fp32 recurrence; fla's own chunk/recur match naive to
~2e-7). Conventions aligned: q/k are l2-normalized externally, `scale = 1/√128`; the a2 kernel takes
`q·scale` (it does no preprocessing), fla takes `q,k` with `scale=` (fla scales q internally).

**Fairness note.** The a2 kernel always does the q/k l2-normalization inside the recurrence, so the
comparison must turn fla's in-kernel l2norm **on** (`use_qk_l2norm_in_kernel=True`) with **raw** q/k
fed to both — otherwise the a2 kernel pays for a normalization that fla skips. Both sides run **eager**
(no NPU-Graph on either), so wall and device numbers are like-for-like.

## Device time (µs/call, msprof Task Duration, forward, 1×64×16, both l2-normalizing)

| kernel | device µs/call | note |
|---|---|---|
| **a2 `gdn2_recurrent` (fwd)** | **197.5** | 110 ops/100 calls; **also checkpoints all T states for the backward** (extra work fla's inference kernel skips); VEC-bound aiv_vec_ratio 0.879 |
| fla_recur (Triton `fused_recurrent_gdn2`) | 206.6 | 220 ops/100 calls; l2norm is a separate pass (+43µs over the no-norm 163.7) |
| fla_chunk (Triton `chunk_gdn2`) | ≫ | chunk path far slower on this recurrence shape |

## Wall time (µs/call, host `torch.npu.synchronize` loop, forward, both l2-normalizing)

| shape (B,T,H) | a2 kernel | fla_recur (Triton) | fla_chunk (Triton) |
|---|---|---|---|
| (1, 64, 16)  | 432.1 | 352.1 | 4760.7 |
| (1, 128, 16) | 451.5 | 374.7 | 5169.0 |
| (2, 64, 16)  | 442.6 | 374.3 | 5093.4 |

Correctness (rel-L2 vs `naive_recurrent_gdn2`): a2 **5.3e-6**, fla_recur 2.3e-7, fla_chunk 4.1e-7.

## NPU-Graph (dispatch removed, fair: both graph-captured), wall µs/call, 1×64×16

| kernel | eager wall | NPU-Graph wall | correctness vs eager |
|---|---|---|---|
| **a2 `gdn2_recurrent`** | 414 | **200.1** | **relL2 0.00e+00 (bit-identical)** |
| fla_recur (Triton) | 349 | 202.0 | relL2 0.00e+00 |

NPU-Graph capture is **bit-for-bit identical** (no arithmetic change) — it only removes the per-call
launch overhead. It collapses both kernels to near their device time, so the eager-mode wall gap
(the a2's heavier aclnn dispatch vs Triton's launch) disappears: graph-vs-graph the a2 kernel is
200.1 vs 202.0 µs — a hair ahead, consistent with its device-time win.

## Reading

- **On device time (the dispatch-agnostic, fair kernel comparison), the a2 kernel WINS: 197.5 vs
  206.6 µs (~5% faster)** — while doing *more* work (per-step state checkpointing for the backward).
  fla's fused-recurrent Triton pays ~43µs for its separate l2norm pass; the a2 kernel fuses it into
  the recurrence.
- The a2 kernel **beats fla's chunk Triton ~8-11×** on this recurrent shape.
- fla wins on **wall** time (352 vs 432µs) purely on the a2's aclnn host-dispatch floor (~230µs),
  not compute — and both run eager, so this is the dispatch mechanism, not the kernel.
- Remaining stretch goal: push VEC utilization from 0.879 to **> 90%** (needs op-reduction beyond the
  frozen-arithmetic delta-checkpoint), and shrink the aclnn dispatch floor for the wall-time win.

_Measured on Ascend 910B3, CANN 9.2.0-beta.1, torch_npu 2.10, ascriptor library 90cfcdc / kernels
b3b3f9c; fla pinned commit e52dbc0 (0.6.0) via the integrated triton-ascend backend._
