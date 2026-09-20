# BF-06 evidence

This directory contains text evidence, not compiled vendors or machine
configuration. The recorded acceptance includes the standalone decode grid, prefill integration
at all six block dimensions and synchronized performance at bd4.

- `cpu-precision-initial.json`: 82-case CPU calibration fixed before kernel
  implementation, including the BF16-state negative control.
- `initial-full/`: first full BF16/FP32 native workload at bd4, before the
  bounded model diagnostics. Its initial environment lookup failed; the raw
  failed line and subsequent correct compiler/OPP version output are both kept.
- `native-grid/receipts/`: complete numerical, boundary, compilation, source
  identity and device receipts at all six block dimensions. Case-level metrics
  are retained inside the original summaries; duplicate individual case files
  are not needed for replay.
- `native-grid/logs/`: complete execution/build output, exit codes and raw
  CANN/OPP identification. Only private absolute path roots are replaced.
- `native-grid/replay.json`: independently recomputed budgets, required
  backward entries, audits and byte equality between block dimensions.
- `native-acceptance/`: 126 prefill/decode chains across six block dimensions,
  original-wrapper FP32 comparisons, synchronized timings and their independent
  replay. The timing receipts retain every sample and each slowdown.
- `models/`: bounded source-based functional and pipe-model diagnostics,
  separate from vendor compilation and native execution.
- `emitted/`: source-emission manifests and hashes; generated build source
  and binaries remain outside the committed tree.
- `warning-review.json`: classification of every retained build warning.
- `publication.json`: raw and published text hashes, with redaction flags
  and preserved line counts. Numerical JSON values and warnings are unchanged.
- `fp32-entry-before.py`: exact original public wrapper from commit `0f517ee`,
  used by the later prefill and performance modes. This baseline intentionally
  preserves the old host conversion costs; it is not a production path.

From the repository root, independently replay the complete published grid:

```bash
python kernels/projects/a5/kda_fused_recurrent_bf16/aggregate.py \
  --root kernels/projects/a5/kda_fused_recurrent_bf16/evidence/native-grid/receipts \
  --output tmp/bf06-replayed.json
python kernels/projects/a5/kda_fused_recurrent_bf16/replay_acceptance.py \
  --root kernels/projects/a5/kda_fused_recurrent_bf16/evidence/native-acceptance/receipts \
  --output tmp/bf06-acceptance-replayed.json
```

The native grid ran verifier commit `492127d`; the subsequent prefill and timing
stage ran verifier commit `caf75e7`. Kernel and public-wrapper hashes
are recorded in each run's environment JSON. Later verifier improvements do
not retroactively qualify unexecuted checks. Original FP32 kernel sources and
the selected library/kernels checkouts remain unchanged.
