# PK-04 evidence index

These are observations from the repaired PKDA FP32 implementation on
SoC950PR_9589 V100, CANN9.2.0, OPP ascend950/ascend910b/ascend910_93,
Python3.12.13, Torch2.12.0+cpu, torch_npu2.12.0 and Ascriptor0.1.0 at
library90cfcdc720bbcd66e8bd4361c4dd4fbc1a2a57b5. Native execution uses the
inprocess bridge. No SSH board/aclnn launcher result is implied.

| Evidence | Content |
|---|---|
| `native-acceptance.json` | Independent collection check:112 cases,40 API checks, cross-bd input/stage hashes, maxima and environment |
| `native-bd{1,2,3,4}-summary.json` | Every case/seed/shape, CPU-input hashes,14 actual-stage metrics/hashes and both public-output CPU oracle comparisons |
| `native-bd{1,2,3,4}-compile.json` | All five vendor signatures and binaries' SHA256, before each process's first custom execution; cache hits included |
| `native-bd{1,2,3,4}-public-boundaries.json` | Independent optional states, default center, split carry and output_final_state=False |
| `native-bd{1,2,3,4}.log` | Complete numerical runner logs, with unchanged original line numbering |
| `portable-bd4-*`, `portable-bd4.log` | Independent28+10 rerun through delivered verifier, same stage hashes |
| `tails-bd4-*`, `tails-bd4.log` | Fresh fullT4096 case plus all64 possible dynamic VF tail counts; same comparisons and output poison |
| `vendor-diagnostics.json` | All20 build-log identities, generated VF hashes and compiler warnings with original line numbers |
| `perf-t{1024,4096}.json`, `performance.log` | All synchronized samples from three baseline/candidate/baseline rounds; baseline is Torch NPU eager FLA naive |
| `closeout-bd4-*`, `closeout-bd4.log` | Final delivered verifier bytes, fresh fullT4096 actual run, unchanged14stage hashes |
| `original-failure-preservation.json` | Preserved original tensor sample SHA256, actual input/stage hashes and recomputed original failure |
| `original-native-failure.json` | Original prepare's in-domain gate failure, preserved as a failure |
| `diagnose-{old,repaired}-{sim,pipesim}.json` | First incorrect prefix boundary and repaired bounded diagnostic; models are not native receipts |
| `canonical-repaired-*.json`, `repaired-*.json` | Fresh16case CPU reference and bounded sim/pipesim checks on repaired source |
| `host-tests.log`, `host-pkda-tests.log`, `board-cpu-pkda-tests.log` |521passed/5skipped fullhost and51PKDA tests on each CPU environment |
| `host-pk02.json` | Historical old-source PK-02 host qualification, retained unchanged; not evidence for repaired-source execution |

Raw log anchors on the environment above:

- `native-bd4.log:2,4,6,8,10`: all five vendors ready.
- `native-bd4.log:17`: fullB1/T4096/H8 actual public o/S/A naive relative L2
  `4.4586199458489094e-7 /4.016814744062547e-7 /7.202348048215453e-8`.
- `native-bd4.log:192`: repaired adversarial seed71 o/S/A relative L2
  `6.19120824226093e-6 /6.617198916201339e-6 /5.6550887037066276e-8`.
- `native-bd4.log:207–208`:10 API boundaries and28 full cases complete.
  The other three bd logs have the same case and completion line locations.
- `portable-bd4.log:207–208`: independent delivered-verifier repeat.
- `performance.log:18–23`: all six timing rounds; complete raw samples are in JSON.

Each native process holds an externally managed shared device lock after fresh
health/occupancy checks. Its successful closeout reports healthy status. Private
machine paths, coordinates and foreign-process attribution stay in ignored
configuration/logs. They are intentionally absent here. Raw numerical values,
case identities and output hashes are retained. Canonical-model import locations
are replaced with relative selected-source identities only.

The vendor's loop-condition compatibility warning is **retained**, not disabled.
The precision report records the bounded-source argument and exhaustive native
count1..64 audit; no qualification is extended to other compiler/header variants.
An alternate CANN9.2.0 installation's duplicate-g_coreType failure is retained in
private raw build logs and described in the public precision report.

`../validation.json` records delivered runtime/verifier/test hashes. The original
native controller snapshot had base075c95a plus the prepare repair; later base
updates and qualification metadata do not alter the executed kernel/API sources.
The portable rerun and tail audit exercise the delivered harness; a final full-workload closeout records its exact delivered bytes after an EOF whitespace cleanup.
The main112/40 counts exclude repetitions and the additional65-case tail audit.

For recovery, keep these text receipts with the source files at the hashes in
`../validation.json`, install the explicitly pinned dependencies, and rerun the
commands in the precision report under fresh external device checks. The ignored
sanitized archive's restoration receipt is `recovery.json`; it records extracted
source hash verification and actual reference/regression re-execution. Private
machine configuration and compiled binaries are not part of that source archive.
