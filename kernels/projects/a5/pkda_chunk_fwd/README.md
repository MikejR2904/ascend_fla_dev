# a5.pkda_chunk_fwd

FP32 PKDA forward in five Ascriptor launches: fused ATK preparation, asymmetric
causal scores, triangular solve, state scan and output. The public entry is
`ascend_fla.ops.pkda_chunk_fwd.chunk_precond_kda`. It follows pinned FLA naive
semantics and consumes q/k without hidden normalization. Both main and ATK
states are independent, optional inputs and fresh outputs.

The declared kernel domain is B1/2,T1..4096,H1..32,K=V128, contiguous FP32,
a5/cce, block_dim1..4. See `contract.json` for all numeric gates and executed
support scope. BF16, backward and unimplemented FLA options fail explicitly.
Host references cover the 340M training geometry H8/K=V128; K120 is unsupported.
Native board/aclnn and performance remain untested. No checkpoint or Triton
execution is claimed.

Run independently with the accepted Ascriptor library on PYTHONPATH:

```bash
python run.py reference --output <ignored-output>/reference
python run.py check --launcher sim --case t1_h1 --output <ignored-output>/sim
python run.py check --launcher pipesim --case t2_h1 --output <ignored-output>/pipesim
# Set FLA_PKDA_NAIVE to the pinned FLA naive.py for both commands below.
python ref/survey.py --output <ignored-output>/survey.json
python verify.py --launcher pipesim --case t65_h1 --case b2_t2_h3 --block-dim 1 --output <ignored-output>/bd1
```

`run.py` uses the unchanged canonical runner, whose digest is recorded in
`runner-source.json`. Models check all14 stage outputs against independent
CPU math; `verify.py` additionally checks actual returned outputs against
pinned FLA naive. Each block_dim is run in a separate process. The comparison
budget for all three outputs remains relative L2<=1e-4.

See [the ABI and precision report](../../../../docs/research/pkda_chunk_fwd_gate_range.md)
for formula provenance, buffer ownership, gate/normalization limits and
reproduction. `validation.json` records only the evidence actually obtained.
