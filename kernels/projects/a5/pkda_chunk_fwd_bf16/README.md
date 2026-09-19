# Native BF16 PKDA forward

Public entry: `ascend_fla.ops.pkda_chunk_fwd.chunk_precond_kda`. BF16 q/k/v enter custom kernels directly; `o` is written BF16. Gate tensors, main/ATK states and accumulations are FP32. Raw q/k are consumed unchanged. The accepted numeric domain is in `contract.json` and the [precision report](../../../../docs/research/pkda_chunk_fwd_bf16.md).

The six-launch BF16 graph initializes/validates independent optional states, prepares ATK/Kahan workspaces and guards inputs, forms scores, solves WY, scans states, then computes output with BF16 cube operands. The FP32 public path retains its original five kernels and read-only numeric validation. There is no backward/autograd support.

## Reproduce

Use the pinned Ascriptor library and SHA-checked FLA naive selected by `FLA_PKDA_NAIVE`. Select an accepted Python environment, and keep all outputs/cache/tmp paths in an ignored task directory. Machine configuration is external through `ASCRIPTOR_MACHINE_SPECS` and `ASCRIPTOR_BOARDS`.

```sh
# CPU-only precision study; not hardware qualification.
python ref/precision_study.py --output "$RESULTS/precision.json"
python run.py reference --output "$RESULTS/reference"

# Run under the external device lock after fresh health/occupancy checks.
# Each invocation must be a separate process. All eleven BF16+FP32 vendors
# compile before the first custom launch, including the full-workload run.
python verify_native.py --block-dim 1 --mode suite --output "$RESULTS/suite-bd1"
python verify_native.py --block-dim 2 --mode suite --output "$RESULTS/suite-bd2"
python verify_native.py --block-dim 3 --mode suite --output "$RESULTS/suite-bd3"
python verify_native.py --block-dim 4 --mode suite --output "$RESULTS/suite-bd4"
python compare_runs.py "$RESULTS/suite-bd1" "$RESULTS/suite-bd2" \
  "$RESULTS/suite-bd3" "$RESULTS/suite-bd4" --output "$RESULTS/aggregate.json"
python verify_native.py --block-dim 4 --mode perf --output "$RESULTS/perf-bd4"

# Bounded diagnostics only after original native workloads.
python diagnose_output.py --output "$RESULTS/output-diagnostic"
python run.py check --case t1_h1 --launcher sim --output "$RESULTS/unit-sim"
```

Public applications can precompile with `prepare(dtype=torch.bfloat16, block_dim=...)`; call `prepare(dtype=torch.float32, ...)` too before using both dtypes in one process. Never change block_dim within a process or register another vendor after the first custom call.

The verifier generates both inputs and independent CPU FP32 goldens on the testing machine. It checks fixed per-case <=1e-2 AND <=3F budgets for output/main state and <=1e-4 for ATK, poisons allocations, hashes all inputs and19 stage/status buffers, audits public host operators, tests independent state continuation and numeric rejection, and compares FP32 bytes to the recorded original wrapper. Read-only FP32 validation, constant allocation and tiny control-code readbacks are explicitly identified audit categories under PM clarification5743858594; computational host transforms are prohibited.

Performance uses three synchronized rounds of original FP32/BF16/original FP32 at T1024/T4096, plus three separate pinned Torch NPU naive/BF16/Torch NPU naive rounds. The FP32 comparison warms each dtype three times before the rounds; Torch NPU has one validation/warmup call. Measured repeats are five per FP32-comparison segment and two/three/two per Torch-comparison round. CPU references are excluded from timing. No speed threshold is imposed; raw samples, including slower observations, are retained.

`evidence/native-results.tar.gz` contains complete sanitized JSON receipts and unfiltered test/performance logs with only private machine prefixes redacted. Extract it to an ignored directory and rerun `compare_runs.py` on its `receipts/suite-bd*` directories. `evidence/native-summary.json` records recomputed budgets and cross-bd byte comparisons. Qualification is limited to the exact environment and executed cases recorded in `validation.json`; SSH board/aclnn launchers, other SoCs and backward are not qualified.

The restored archive is independently hash-checked and re-aggregated; `evidence/restoration.json` records source identities, isolated source tests, standalone reference/model replay and final native closeout. Full-chain pipe simulation remains untested; the output component diagnostic has a narrower scope.
