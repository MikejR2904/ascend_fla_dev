# Native BF16 grouped GDN backward

Three self-contained A5 CCE launches compute boundary checkpoints, replay and
reverse gradients, then group sums. q/k/v/do and dq/dk/dv use BF16 storage;
gates, state, dht, partial gradients and arithmetic use FP32. Missing cotangents
are zeroed in the kernel. No host conversion or grouped operand copy is used.

The public API is `ascend_fla.ops.gdn_chunk_bwd.chunk_gdn_bwd`; its matching
FP32 path still uses the unchanged original unit. See `contract.json` and
`docs/research/gdn_chunk_bwd_bf16.md` for the frozen domain and per-gradient
`min(1e-2,3F_g)` budget against both pinned FLA autograd A and independent B.

With the pinned accepted environment and explicit `FLA_GDN_NAIVE` selected:

```sh
python run.py reference --case all --output /your/ignored/output
python benchmark.py --block-dim 2 --baseline-root /your/original/repo --output /your/ignored/full.json
python benchmark.py --block-dim 2 --case all --oracle-workers 4 --baseline-root /your/original/repo --output /your/ignored/grid.json
```

Hardware invocations require the assigned device, shared lock and isolated
output/cache environment. The default benchmark runs the full workload first;
its grid checks poisoned leaves/composition, actual public A/B and host audit,
input immutability, absent-cotangent dummy poison, original/new FP32 bytes and
cross-block hashes. `measure.py` measures three complete original-FP32 / new
BF16 / original-FP32 sandwiches on one card, including backward checkpoints.

Status: 88 focused host tests pass; all three entries lower with balanced
events and emit CCE. The complete bd2 grid passes 138/138 native records against A/B and
independent leaves/composition. The bd1 grid and timing are pending. Calibration
JSON files are pre-implementation CPU evidence, not device results.

CPU-isolated repository suite: 763 passed, 9 skipped, 5 existing importorskip
deprecation warnings (pytest8.3.2). Hardware tests run separately under the
device lock; these skips are not device acceptance. The initial unisolated
host invocation was interrupted after unrelated KDA device tests failed, and
is retained in ignored scratch; no test, requirement or tolerance was relaxed.

First native evidence: `evidence/full-v4-bd2.json`, its unfiltered redacted log,
source manifest, receipt and occupancy record. Canonical CPU reference passed
all138cases; six bd2 vendor build logs and artifact hashes are retained.
`summarize_native.py` rechecks both final grids and every cross-block hash;
`summarize_timing.py` recomputes all retained same-card sample medians.
