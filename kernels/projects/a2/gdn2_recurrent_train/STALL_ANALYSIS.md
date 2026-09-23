# GDN-2 a2 forward: rigorous analysis of the per-step stall

Measured on Ascend 910B3 (a2 / c220), CANN 9.2.0-beta.1, block_dim=40, pure-vector kernel
(`gdn2_fwd_states_kernel`). Device time via msprof `Task Duration` and NPU-Graph replay (which
equals device time, no host-dispatch noise). Shape B=1, H=16 (B·H=16) unless noted.

## 1. The decomposition (single op, msprof)

```
Task Duration : 187.9 µs        (block_dim=40, aicore/cube = 0 → pure vector)
AIV active    :  79.9 µs (42%)  ← vec 64.9 + scalar 8.9 + mte2 9.3 + mte3 16.3 (these OVERLAP)
AIV idle/stall: 108.0 µs (58%)  ← the vector core executes NOTHING here; Task Wait Time = 0
```

The vector core is **idle 53–58% of the time**. That idle is the subject of this note.

## 2. It is a fixed per-step cost (T-sweep, NPU-Graph device time)

| T | device µs | µs/step |
|---|---|---|
| 8 | 35.0 | — |
| 16 | 56.2 | 2.65 |
| 32 | 99.4 | 2.70 |
| 64 | 184.5 | 2.66 |
| 128 | 355.6 | 2.67 |
| 256 | 706.5 | 2.74 |

Perfectly linear: **device ≈ 13.6 µs (fixed) + 2.67 µs · T**. So the stall is a constant paid once per
recurrence step — consistent with a per-step dependency chain, not a one-off setup cost.

## 3. What the stall is NOT (each ruled out by measurement)

- **NOT input DMA (MTE2).** aiv_mte2 = 5.68 µs = 2.9% of device, already overlapped inside `aiv_time`
  by `auto_sync` (the six independent loads issue in parallel behind the q/k-normalize compute).
- **NOT the state checkpoint (MTE3).** Control experiment: `gdn2_fused_recurrent` is the identical
  recurrence with **no per-step [128,128] state write**. It runs at **2.657 µs/step vs 2.67** — the
  checkpoint costs ~0. `auto_sync` fully hides that MTE3 behind compute.
- **NOT cube.** `aicore/cube = 0` (pure vector kernel).

## 4. What it IS — serial op latency on a single vector pipe

The lowered kernel body contains, per step:

| category | count | notes |
|---|---|---|
| vector/compute ops | 78 | the frozen arithmetic (l2-norm rsqrt-Newton, two 7-add reduction trees, rowscale/outer) |
| sync ops | 47 | 17 `sync.set` + 17 `sync.wait` + 12 `sync.event` — cross-pipe handoffs |
| DMA ops | 14 | 6 loads + 3 stores + copies |

≈139 real ops/step, ~19 ns each on the core's **single** vector execution unit. So:

```
~66% of the stall = serial latency of the 78 vector ops (one vector pipe, mostly data-dependent)
~34% of the stall = cross-pipe synchronization (the 47 set/wait/event ops)
```

The 47 syncs are overwhelmingly **correctness-mandatory**: each of the 6 loads needs
`set(load-done)→wait(before the vector op reads it)`; each of the 3 stores needs
`wait(producer)→set(WAR before overwrite)`; and `state[t]→state[t+1]` needs its own fence.

## 5. Instruction-level parallelism does NOT help (three experiments)

The two heaviest sub-chains — q-normalize and k-normalize (~15 tiny ops each) — are mutually
independent. Three correctness-preserving attempts to exploit that, all **bit-for-bit identical** to
the original (dq 9.79e-07, dk 1.01e-05, … unchanged), all **zero speedup**:

| variant | correctness | µs/step |
|---|---|---|
| original | ✓ | 2.670 |
| private normalize temporaries (break the shared-scratch WAR) | ✓ bit-identical | 2.68 |
| explicit op-by-op interleave of the q/k normalize chains | ✓ bit-identical | 2.67 |

Conclusion: independence cannot be turned into parallelism here — there is **one** vector execution
unit per core, and the scheduler/hardware serialize the ops regardless of source-level ILP. The 40
cores parallelize across (b,h) items, not within one item's op stream.

## 6. Why the remaining levers are blocked (all tested on a duplicated kernel)

- **Fewer ops via wider vector instructions** → **hardware-blocked.** Collapsing a `NGROUP=2`
  two-op (2×64-lane) helper into one 128-lane op is rejected by the compiler:
  `error[E0301]: count_per_rep ... valid range is [0, 64], got 128`. The a2 vector unit is
  hard-limited to **64 lanes per repeat**, so 128-wide tiles *must* be two ops; and the two
  column-chunks are not a single strided extension of the row-repeats, so they can't fold into one
  `repeat` either. The two-op pattern is mandatory.
- **Fewer ops via cheaper arithmetic** → the remaining ops (rsqrt-Newton, the reduction trees) *are*
  the frozen bit-exact-validated math. Ruled out.
- **Manual sync** → `auto_sync`'s 17 `set` / 17 `wait` pairs already match the *necessary* cross-pipe
  handoffs almost exactly (6 loads + 3 stores + the `state[t]→state[t+1]` fence ≈ 17). It is
  **precise, not conservative**, so there is essentially no removable sync; a whole-kernel manual
  `setflag`/`waitflag` rewrite would gain ~0 while risking correctness.
- **Coalesce loads** → the 6 inputs are 6 separate GM tensors; they cannot merge into one DMA.
- **Interleave two (b,h) recurrences per core** to fill the pipe → **UB-limited**: two states +
  two reduction scratches = 256 KB > 192 KB UB. Does not fit.

## 7. Context

- fla's Triton `fused_recurrent_gdn2` has the **identical** sequential-recurrence stall and is
  **slower on device** (206.6 vs 197.5 µs, fair — both doing the l2-norm). So this kernel is already
  at/past the reference for this exact bottleneck.
- **NPU-Graph** capture removes the aclnn host-dispatch floor **bit-for-bit identically** (relL2
  0.00e+00), taking wall from 414 → 200 µs. See `BENCHMARK_TRITON.md`.

## 8. Verdict

The ~53% per-step stall is at the **correctness-preserving floor** for a per-channel-gated sequential
delta recurrence on a single-vector-pipe core: ~2/3 irreducible serial latency of the frozen-arithmetic
op stream, ~1/3 required cross-pipe sync. Going lower requires changing the arithmetic, hand-removing
required syncs, or a fundamentally different (chunk-parallel) algorithm — the last of which is ~8×
slower on this shape (see `BENCHMARK_TRITON.md`). The design is frozen here; further manual-sync and
op-reduction experiments are carried out on duplicated files without touching this validated kernel.
