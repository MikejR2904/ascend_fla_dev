# Secondary environment: vendor build blocked

This is a failed supplementary qualification, not device acceptance. The same
source snapshot and pinned library were tested on another A5 environment.
The full workload was selected, but no custom kernel reached execution.

The first attempt failed while compiling the unchanged original FP32
checkpoint: the CANN TBE Python package could not import `decorator`.
The installed TBE metadata declares `decorator` and `scipy`; both missing
dependencies were installed only in the task's private virtual environment.
A fresh TBE import then passed. Its scratch directory was explicitly bound
inside the permitted workspace. The shared Python and CANN trees were unchanged.

The next attempt failed earlier than BF16 execution, again in the unchanged
FP32 checkpoint. At library pin 90cfcdc, generated `tensorutils_cce.h`
tests `IMPL_UTILS_SYS_MACROS_H` before providing its fallback `g_coreType`
definition. This CANN 9.2.0 installation instead uses
`IMPL_UTILS_SYS_MACROS_IMPL_H` in `sys_macros_impl.h:15` and defines
`g_coreType` at lines 72/74. Both definitions become visible, producing
the actual compiler error `redefinition of 'g_coreType'`.

The CANN compiler package reports timestamp `20260909_000323409`. Its actual
header SHA256 is
`8ab88268cc820b7580da708b783ca35e12ba55a2f1baac2d343e3248fc365982`.
A read-only toolkit inventory found one installed toolkit; the other names
resolve to that same tree. No alternate toolkit was selected, library pin
changed, generated bundle patched, or vendor header modified. This mismatch
belongs to the pinned library/vendor-header interface. It remains unresolved
for this secondary environment and was reported to PM.

The failed job released its shared lock; driver context registries were empty
afterwards and health remained good. All required numerical and timing
acceptance continues on the primary environment, whose actual vendor builds
and full workloads pass. That success does not establish compatibility with
this secondary toolkit. The secondary Torch distribution includes CUDA in
its version label; no CUDA or Triton workload was run.

`secondary-build-failure.json` retains environment versions, exact header
sites and hashes. The four accompanying failure logs retain every line after
redacting machine information; unredacted logs and occupancy are private.
