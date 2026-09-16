# `a5.gdn2_fused_recurrent`

Repository-owned short-sequence GDN-2 forward kernel. It is authored with
`ascriptor.a5`, emitted and built with the CCE backend, then called through the
repository's in-process aclnn bridge.

The layer supplies activated channel-wise `g`, erase `b`, and write `w`. The
kernel performs FP32 q/k L2 normalization, keeps one complete `[128,128]` state
per head in UB, advances `1..16` tokens serially, and writes FP32 final state.
Inputs and output use token-major `[B,T,H,128]`; state uses `[B,H,128,128]`.

Declared first scope: `B=1`, `H in {1,16}`, `1<=T<=16`, `block_dim=8`, A5 CCE.
Each of the 16 vector participants in the real-model launch owns one head, so
there is no cross-core state reduction or synchronization.

The generated HostSpec names the erase tensor `erase_gate`: CANN lowercases the
shape symbol `B` to scalar `b`, so a tensor also named `b` would produce a C++
declaration collision. The public Python API remains `b`.

Run the independent reference and model checks from this directory with the
accepted Ascriptor library on `PYTHONPATH`:

```bash
python run.py reference --case all --device a5 --backend cce --block-dim 8
python run.py check --case single_zero_state --launcher sim --device a5 --backend cce --block-dim 8
python run.py check --case single_zero_state --launcher pipesim --device a5 --backend cce --block-dim 8
```

Actual CCE vendor build and NPU execution are performed by
`ascend_fla.runtime.compile.compile_kernel(..., backend="cce")`; model integration
does not use the simulator or a Torch fallback.

## Validation snapshot

On 2026-09-14, `ascriptor check` reported zero errors and zero warnings. The
independent reference passed all five cases, functional simulation passed the
three scoped cases, and pipesim passed its two scoped cases without hazards.
The canonical unit runner then executed all five cases through the CCE/aclnn
launcher on A5. Output relative-L2 was at most `4.169e-6` and final-state
relative-L2 was at most `9.108e-8`, both below the `1e-4` contract budget.

The public BF16 wrapper also passed the real prompt shape, and a four-token call
matched four chained one-token calls exactly. For B1/T1/H16, the complete public
call measured `88.165 us` versus `148.400 us` for the torch_npu composition
(warmup 10, 100 synchronized iterations). Whole-model measurements belong in
the repository support matrix rather than this unit contract.
