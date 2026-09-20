# Native token-major KDA decode

`kda_decode_bf16_kernel` reads BF16 q/k/v and writes BF16 output. Gates, beta,
initial/final state and the recurrence are FP32. `kda_decode_fp32_kernel` is
the companion all-FP32 entry. Both implement T=1..16, K=V=128, grouped value
heads and optional initial state without host scaling, casts or layout copies.
The original `kda_fused_recurrent` unit is read-only.

One vector participant owns each assigned head's complete state. Explicit GM
DMA gaps address token-major rows across the physical head dimension; beta
uses a padded 32-byte UB row. There is no cross-core communication. BF16
widening and final ties-to-even conversion happen inside the kernel.

The public API is `ascend_fla.ops.kda.fused_recurrent_kda`. A process combining
prefill and decode must call `ascend_fla.ops.kda.prepare(decode=True)` before
its first custom kernel. This registers both new dtype vendors and, by default,
all five forward and nine backward dependencies. The accepted decode block
dimensions are 1/2/4/8/16/28; each runs in a separate process.

Default flags=False has only output allocation and launch on the host. The
existing raw-input flags still use the unchanged A2-44 preprocessing helper;
this is the explicitly recorded legacy exception pending BF-07. It is audited
separately and is not claimed to comply with the default-path restriction.

## Reproduction

Use the selected Ascriptor sources, an accepted Python environment and private
machine configuration. Set `BF06_FLA_NAIVE` to the pinned FLA v0.5.2 KDA
`naive.py` file (SHA256 is checked by `research.py`). Put this repository and
the selected Ascriptor library on `PYTHONPATH`.

```bash
python kernels/projects/a5/kda_fused_recurrent_bf16/run.py reference \
  --output tmp/bf06/reference
```

Full hardware acceptance uses `verify_native.py`. The external launcher must
check health/occupancy, hold the machine's shared device locks and isolate its
outputs before setting `BF06_EXTERNAL_DEVICE_LOCK=1`. It must select a device
mapping and the private CANN/cache/log paths. No machine values belong here.

```bash
python kernels/projects/a5/kda_fused_recurrent_bf16/verify_native.py \
  --mode suite --block-dim 4 --output tmp/bf06/native-bd4
```

The verifier compiles all 17 vendors, including the original decode baseline
and all nine backward entries, before the first custom execution. Its first
case is B2/T16/H4/HV32, followed by both-dtype checks and the requested mode.
Run the complete suite independently at bd=1/2/4; use `--mode boundaries`
at bd=8/16/28 and `--mode perf` for three synchronized baseline/candidate/baseline
rounds plus the actual Torch NPU reference baseline. No backward execution
claim follows merely from compiling the backward dependencies.

Run `--mode prefill` separately at each accepted block dimension. It checks
21 stable-prefill/decode chains against both CPU FP32 references, including
64/128-token prefixes, 64 decoded tokens, grouped heads and gate-span boundaries.
The three comparison layers and their fixed budgets are in the research report.
The original FP32 baseline is the exact public wrapper snapshot in
`evidence/fp32-entry-before.py`, using the unchanged original kernel.

Only after the full hardware workload, run bounded source-based diagnostics:

```bash
python kernels/projects/a5/kda_fused_recurrent_bf16/run.py check \
  --case grid_b1_h1_t1_g1_s0 --launcher sim --block-dim 1 \
  --timeout 100 --output tmp/bf06/sim-t1
python kernels/projects/a5/kda_fused_recurrent_bf16/run.py check \
  --case grid_b1_h1_t2_g4_s1 --launcher pipesim --block-dim 1 \
  --timeout 150 --output tmp/bf06/pipesim-t2-g4
```

The second probe retains token row strides and repeated-head UB reuse: four
heads on two vector participants. The first exercises kernel zero-state
initialization and the T=1 tail. Neither diagnostic replaces native acceptance.
The companion FP32 entry has a bounded zero-state T2/H1/HV4 probe:

```bash
python kernels/projects/a5/kda_fused_recurrent_bf16/diagnose_fp32.py \
  --launcher pipesim --output tmp/bf06/fp32-pipesim
```

Use `--launcher sim` for the functional model. Grid receipts can be independently
replayed with `aggregate.py --root <receipts-directory> --output <result.json>`;
the replay checks all nine backward compile entries, numerical budgets, audits
and cross-block-dimension output bytes. Prefill and performance are separate
acceptance stages.

## Precision and evidence

Both CPU FP32 references consume actual BF16-rounded input values. The fixed
BF16 output relative L2 budget is `min(0.01, 3F)` for each oracle; the FP32
state budget is `1e-5`. FP32 output also uses `1e-5`, and old/new FP32 results
must be bitwise identical. Cross-bd bytes, per-head errors, NaN coverage and
input immutability are recorded separately.

See `docs/research/kda_decode_bf16.md` for the study fixed before implementation,
`contract.json` for qualification by stage, and `evidence/` for public text
receipts. Unexecuted stages remain untested; CPU or build success is not NPU
acceptance. The recorded decode grids, prefill integration and performance measurements are complete; performance includes regressions and has no speed pass threshold.
