# GDN-2 chunk forward (experimental)

A five-launch FP32 CCE implementation: normalization/cumulative gates, causal
scores, exact triangular solve, cross-chunk state scan, and output composition.
This is a vector baseline; Cube acceleration remains open. Native correctness
is qualified for the cases and block counts recorded in `contract.json`.

The unit is portable: copy this directory alone, install the accepted Ascriptor
and CPU Torch dependencies, then run `run.py reference` or `run.py check`.
`contract.json` is the supported-domain and error-budget specification.

```sh
python run.py reference --output tmp/reference
python run.py check --launcher aclnn --case t4096_h16_prefill --output tmp/board-full
python run.py check --launcher pipesim --case t65_h1_diagnostic_tail --block-dim 1 --output tmp/diagnostic
```

`aclnn` needs CANN and an available NPU, but Torch stays on CPU. Use the full
workload on the assigned board first, then reduced diagnostics for a located
problem. Each launch returns to the CPU harness; host timing includes transfers
and runtime overhead. It is not in-process NPU model performance.

The public API is the explicit sibling module `ascend_fla.ops.gdn2_chunk_fwd`.
CPU tensors need `chunk_gdn2(..., launcher="aclnn")`; NPU tensors use
`launcher="inprocess"`. Both execute the same CCE launch graph. The CPU
mathematical composition is `ref/chunk.py`; it is never a fallback for CCE.
This task does not modify the existing layer/model selection or operator package.

No training caches or backward path are provided. Runtime workspaces are fresh
FP32 allocations and cannot alias input state. Public execution retires each
workspace after its final consumer. The unit checker retains intermediate
outputs for comparison. Further fusion and Cube optimization require measured
hardware evidence, not a simulator latency claim.

The scan/output preweights are computed per FP32 channel row in private UB.
This preserves the sum order while removing repeated broadcast exponentials.
WY solves adjacent rows together to reuse previous U/W row loads, retaining
each row's FP32 product/subtraction order and the native loop workaround.
`benchmark.py` compares against GD2-01 or the qualified preweight source with
CPU goldens and a same-process NPU sandwich. Commands, source digests and measurements
are in `docs/research/gdn2_chunk_fwd_gate_range.md` at the repository root.
