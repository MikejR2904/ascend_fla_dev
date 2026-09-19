# Native PGDN evidence

Physical-device in-process CCE on Ascend950PR_9589 V100; CANN9.2.0;
OPP `opp/built-in/op_impl/ai_core/tbe/kernel/ascend950`. Docker Python3.12.13,
Torch2.12.0+cpu, TorchNPU2.12.0, Ascriptor0.1.0. See
[validation.json](../../validation.json) for selected source pins and fresh Docker
compiler artifact hashes. Machine bindings and unredacted runtime logs stay private.

The result JSON and runner logs here are exact copies of actual outputs, without
reformatting or removed lines. Their hashes and line counts are in validation.json.
All cases generate inputs and independent references at run time.

| Run order | Workload / result | Original runner log |
| --- | --- | --- |
| 1 | [Full B1/T4096/H=HV8, bd2](first-full-bd2-result.json) | [lines1–7](first-full-bd2-run.log), oracle metrics at7 |
| 2 | [Same full case, bd1](first-full-bd1-result.json) | [lines1–7](first-full-bd1-run.log), oracle metrics at7 |
| 3 | [Remaining29 cases, bd1](grid-bd1-result.json) | [lines1–203](grid-bd1-run.log), each seventh line has oracle metrics |
| 4 | [Remaining29 cases, bd2](grid-bd2-result.json) | [lines1–203](grid-bd2-run.log), each seventh line has oracle metrics |
| 5 | [Four timing workloads, bd2](measure-bd2-result.json) | [lines1–36](measure-bd2-run.log), three rounds/three phases per workload |

[Grid summary](native-grid-summary.json):60 cases,1020 composition-stage arrays,
1020 independent-leaf arrays,120 public FP32/BF16 calls, all passed.
[Cross-block comparisons](byte-comparison-full.json):30 input sets, all17 stage
arrays and all3 public outputs in both dtypes bitwise equal. The four correctness
reports retain the earlier **unused** measure.py hash. All executed computational
sources match the delivered bytes; the timing report binds the final measure.py.

## Three-round same-card timing

B1/H=HV8/K=V128, block_dim2. Each phase:10 warmups,50 measured synchronized
wall times, including dispatch. Baseline uses the five PGDN-owned chunk kernels
with equal normalized read/write keys and no ATK. Normalization, allocation and
output casting on NPU are timed. Candidate is the complete public PGDN call with
ATK, validation predicates and both final states. This baseline is not the GDA-02
implementation, nor a semantically equivalent PGDN algorithm. Pre/post pinned
GDN and PGDN A/B checks passed; no input mutation. No speed threshold.

| T | dtype | round | GDN before (ms) | PGDN (ms) | GDN after (ms) | PGDN/mean GDN |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | float32 | 1 | 547.686 | 566.339 | 547.668 | 1.034075 |
| 4096 | float32 | 2 | 547.669 | 566.421 | 547.669 | 1.034239 |
| 4096 | float32 | 3 | 547.664 | 566.280 | 547.662 | 1.033995 |
| 4096 | bfloat16 | 1 | 547.864 | 566.907 | 547.836 | 1.034784 |
| 4096 | bfloat16 | 2 | 547.836 | 566.630 | 547.833 | 1.034309 |
| 4096 | bfloat16 | 3 | 547.829 | 566.604 | 547.865 | 1.034238 |
| 1024 | float32 | 1 | 137.843 | 143.647 | 137.819 | 1.042196 |
| 1024 | float32 | 2 | 137.855 | 143.646 | 137.821 | 1.042133 |
| 1024 | float32 | 3 | 137.821 | 143.651 | 137.870 | 1.042113 |
| 1024 | bfloat16 | 1 | 137.816 | 143.999 | 137.858 | 1.044706 |
| 1024 | bfloat16 | 2 | 137.833 | 144.070 | 137.775 | 1.045468 |
| 1024 | bfloat16 | 3 | 137.770 | 143.936 | 137.774 | 1.044736 |

All1800 raw samples, phase timestamps and before/after numerical checks are in
[the timing report](measure-bd2-result.json). No block_dim1 timing was run.

## Runtime warnings and acceptance limits

All21 complete runtime logs,1214 lines, were reviewed. ERROR/FATAL/CRITICAL:0.
Ten WARNING entries are five occurrences each of the following initialization
messages. They were retained without suppression. Sanitized excerpts retain the
original first-process line offsets; machine identifiers are removed.

```text
host log L76: [WARNING] plugin_version_manager.cpp:38 GetPluginUpdateStrategy: halGetDeviceInfo(INFO_TYPE_SWPLUGIN_UPGRADE_POLICY) failed, retCode[3], fallback to PLUGIN_NOT_FORCE_UPDATE
host log L77: [INFO] plugin_version_manager.cpp:76 CompareHostDeviceCompatPluginVersion: device plugin pkg:cann-hcomm-compat.tar.gz version unavailable, fallback to checkcode compare
host log L78: [INFO] package_loader.cpp:487 LoadSinglePackageToDevice: skip load compat plugin package:cann-hcomm-compat.tar.gz by version/strategy check
device log L29: [INFO] aicpusd_threads_process.cpp:62 LoadKernelSo: Start to preload aicpu so.
device log L30: [WARNING] ae_kernel_lib_fwk.cc:298 LoadTfSo: load tensorflow so failed, ret is[11002]
device log L31: [INFO] aicpusd_threads_process.cpp:66 LoadKernelSo: End to preload aicpu so.
device log L37: [INFO] main.cpp:363 main: Aicpu schedule attach and init successfully.
```

The installed driver maps3 to DRV_ERROR_INVALID_VALUE: plugin policy discovery
uses its explicit fallback. No support claim for that feature. The second message
is from TensorFlow framework preloading; the installed SDK lacks the loader source,
so its exact missing-library cause remains unresolved and TensorFlow availability
is not claimed. The task dataflow uses custom CCE plus actual validated Torch NPU
operators, without a TensorFlow graph/backend. There are no downstream execution
errors in the retained logs. These runs certify only the tested PGDN/GDN-equation
calls; they do not certify general framework or driver compatibility.

Fresh final health:Healthy. Shared lock released; no task compute process remains.
Standalone board/aclnn launchers, CUDA/Triton and checkpoint execution are untested.
