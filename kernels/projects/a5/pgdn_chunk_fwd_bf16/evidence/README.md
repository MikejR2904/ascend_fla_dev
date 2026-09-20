# BF-03 evidence index

Start with [native-summary.json](native-summary.json): 132 unique native records,
66 original-FP32 byte comparisons and 1,122 cross-block-dimension stage-array hash
pairs. All 16 zero-budget observations pass. Every case retains its own A/B
rounding floor and budget; a maximum across cases is not used as a shared budget.

The first native job ran the full B1/T4096/H=HV8 workload. Then each block_dim ran
all 33 shapes/seeds in a separate process, with both BF16 and FP32 inputs. Each
record checks 17 composed arrays, independent leaves, public A/B outputs, input
bytes, NaN-prefilled outputs and actual production/validation dispatch events.
The public output hashes overlap the 17 stage arrays and are not counted twice.

| Evidence | Scope |
| --- | --- |
| `calibration-pre-kernel.*`, `calibration-summary.json` | First commit: 66 CPU A/B calibration records and frozen budgets, before kernel implementation |
| `canonical-reference*.log` | Initial metadata rejection retained; corrected runner reference stage passes all 66 cases |
| `host-*.log`, `host-summary.json`, `base-negative-control.log` | Full host suite, focused tests and original-entry negative control |
| `source-emission.json`, `storage-balance.json`, `arithmetic-unchanged.json` | Actual lowering, allocations/event balance and unchanged FP32 arithmetic functions; no simulator claim |
| `full-bd2-v1.*` | First full native workload, both dtype paths |
| `grid-bd1-v1.*`, `grid-bd2-v1.*` | Complete actual native grids, per-case values and full unfiltered console logs |
| `*-receipt.json`, `*-source-manifest.json` | Actual environment/version files and checked source hashes for each job |
| `*-occupancy.json` | Shared-lock sampling from two driver context registries, exact owned child and locked drain |
| `builds/build-artifact-manifest.json` | 24 real builds; 2,056 file hashes including generated sources, device objects and host libraries; no binaries committed |
| `builds/build-review.json`, `builds/*.log` | All build/harness log lines retained; each warning classified against its actual owner source |
| `input-quantization.*` | CPU near-zero input quantization; native cross-dtype differences are separate fields in grid records |
| `timing-summary.json`, `sandwich-bd2-v1.*` | Three same-card FP32/BF16/FP32 rounds per T1024/4096; 900 measured samples, correctness checked before/after |
| `runtime/runtime-review.json`, `runtime/*.log`, `startup-control-v1.*` | All 1,310 runtime log lines; two warning classes reproduced in a fresh torch-only control, exact returned values checked |
| `native-post-run-sources.json` | Final source recheck; kernel/public sources match every run, with the documented first-full benchmark metadata difference |
| `manifest.json` | Complete evidence-file byte counts and SHA256 hashes |

The measured complete-call medians (FP32/BF16) are 143.957737/144.2576425 ms
for T1024 and 566.7901335/567.0028915 ms for T4096, B1/H=HV8/K=V128/bd2.
They include allocations, existing validation and all six launches, with device
synchronization before/after every call. No throughput improvement is claimed.

Native qualification is for the recorded Ascend950PR_9589V100 environment with
CANN 9.2.0 compiler/opp timestamp `20260805_101249091`. CPU reference, emission,
vendor compilation and native execution are separate stages. Simulator,
pipe-simulator, CUDA/Triton and model-weight validations were not run.

Machine identities and paths are redacted without removing log lines or replacing
numerical measurement fields. Original unredacted evidence remains in ignored
task configuration. Every committed blob is below 5 MB.
