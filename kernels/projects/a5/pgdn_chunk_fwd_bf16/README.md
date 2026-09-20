# Native BF16 PGDN forward

Self-contained six-stage A5 CCE unit: BF16 q/k/v enter device kernels directly;
normalization, ATK and all intermediate tensors remain FP32. The output is BF16;
main state `[B,HV,128,128]` and ATK state `[B,H,128]` remain FP32.

The domain matches `pgdn_chunk_fwd`: contiguous token-major inputs, K=V=128,
positive B/H, HV a positive multiple of H, T a positive multiple of 64 through
4096, block_dim 1 or 2. Main/ATK gates and beta are FP32. Initial states are zero,
normalization is enabled, and scale/x/eps/center retain the existing constants.
Read correction uses normalized `k_read`; state writes use `k_write`. No backward
or decode support is included.

Run the independent CPU reference from this directory:

```bash
python run.py reference --device a5 --backend cce --case all --output <ignored-output-dir>
```

For literal A/B calibration, set `FLA_PGDN_NAIVE` through external configuration to
the pinned FLA file and run:

```bash
python -m ref.calibrate --output <ignored-calibration-json>
```

`contract.json` declares 66 BF16 cases (33 shapes/seeds × two launch configurations).
`unit.py` checks actual returned composition and independent leaf results. It has
no production dependency on the original unit. CPU references alone do not
qualify the device implementation; support entries record executed stages only.

The native `benchmark.py` validates both dtype paths, public A/B outputs,
independent leaves, composed intermediates, NaN-prefilled outputs, unchanged
inputs, original FP32 byte equality and host dispatch. All kernels are compiled
before first CANN operator resolution; run each block_dim in a separate process.
Machine selection, locks and outputs belong to external ignored configuration.

The implementation preserves the FP32 Vector arithmetic; it makes no Cube
throughput claim. Precision budgets were committed before kernel code:
`o` and main state each use `min(1e-2,3F)` against each oracle; ATK uses `1e-4`.
Zero F requires exact equality. See
[the precision and validation record](../../../../docs/research/pgdn_chunk_fwd_bf16.md).
