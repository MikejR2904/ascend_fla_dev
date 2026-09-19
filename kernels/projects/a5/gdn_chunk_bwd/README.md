# Grouped GDN chunk backward (GDA-03)

Standalone A5 CCE backward of raw, unnormalized GDN. The unit recomputes its
own chunk checkpoints and returns five FP32 mathematical gradients. The public
`ascend_fla.ops.gdn_chunk_bwd.chunk_gdn_bwd` wrapper additionally accepts matching
BF16 q/k/v/do and returns dq/dk/dv in their input storage dtype. g/beta/dht and
dg/dbeta remain FP32. Existing forward and autograd dispatch are unchanged.

The native grid passed 276 case/dtype records across block_dim 1/2, and 2070
returned-array pairs were byte-identical. Required same-device timing also passed,
with 1800 raw samples and an explicitly identified saved-checkpoint cost baseline. See the task's gate-range
document and evidence directory for exact source identity and measured scope.

Three ordered launches generate chunk-start states, replay and reverse each
64-token chunk, then sum grouped q/k contributions deterministically. No inverse
decay, cumulative gate, normalized key, cube rounding, or atomic sum is used.
Each recurrence has one vector owner; the GM replay tape is explicitly published
before consumption and drained before reuse. Internal read-only inputs include
zero-filled absent cotangents; output storage is freshly allocated.

From an accepted ascriptor environment, the project is independently runnable:

```bash
python run.py reference --case grid_r2_c2_both_bd1 --output <ignored-output>
python run.py check --case grid_r2_c2_both_bd1 --launcher board --board <configured-name> --output <ignored-output>
```

The board alias and machine configuration are external ignored configuration.
The public in-process `benchmark.py` is a separate acceptance path through the
repository runtime bridge. It checks actual returned arrays against independent
stage references plus the exact pinned naive FP32 autograd, poisons fresh stage
outputs, checks input immutability, and records per-gradient numbers and byte
hashes. Native builds for different block_dim values require separate processes
and build directories. All stages must compile before first CANN resolution.

The complete canonical case set contains three upstream-gradient combinations,
ratios 1/2/4/8, chunks 1/2/3 and 4096 tokens, multi-batch, beta/gate endpoints, zero
keys, and underflow/spike cases. Full native B1/T4096/H=HV8 is run first. Sim and
pipesim are only later bounded diagnostics, with their scope recorded separately.

For the test-only A oracle, point `FLA_GDN_NAIVE` at the exact pinned source.
It is hash checked, imported only by validation, never used by device execution.
`python -m ref.calibrate --output <ignored-report>` runs A/B reference calibration,
FP64 finite differences and negative controls. Literal A forces FP32; only the
clearly labeled FP64 qualification copy receives two in-memory dtype changes.
