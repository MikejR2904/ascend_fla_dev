# `a5.gdn2_fused_decode`

Model-specific BF16 decode unit for the released GDN-2 1.3B shape.  Unlike the
standalone FP32 `gdn2_fused_recurrent` ABI, this unit deliberately owns the full
per-layer decode boundary after projection: raw gate activation, q/k norm, FP32
state recurrence, output RMSNorm+swish, and the final BF16 store.

The narrow contract is intentional: B=1, T=1, H=16, K=V=128,
`allow_neg_eigval=false`, block_dim=8.  Unsupported model configurations stay on
the separately declared CCE recurrent path; there is no Torch fallback.

Run from this directory with the Ascriptor 0.1.0 environment:

```bash
python run.py reference --case all --device a5 --backend cce --block-dim 8
python run.py check --case all --launcher sim --device a5 --backend cce --block-dim 8
python run.py check --case all --launcher pipesim --device a5 --backend cce --block-dim 8
python run.py emit --case real_decode --device a5 --backend cce --block-dim 8
```

Correctness stages are green for the declared simulator domain; the real decode
case also passed CCE/aclnn.  The model reached the 74-Cast floor and a stable
1.410x--1.650x paired speedup.  Two statically named 64-row state slots now expose
the legal MTE2/Vector and Vector/MTE3 overlap without changing the ABI or VF math:
two 54-sample silicon profiles measured a combined 4.247 us mean versus a matched
4.733 us whole-state control.  The four AIV pipe ratios sum to 111.55%, confirming
overlap.  A single pipe still cannot approach 80% inside this boundary because the
all-row erase reduction must complete before delta and the second pass.  Exact
evidence and the still-untested board cases live in `contract.json`.
