# a5.pkda_chunk_fwd

FP32 PKDA forward in five Ascriptor launches: fused ATK preparation, asymmetric
causal scores, triangular solve, state scan and output. The public entry is
`ascend_fla.ops.pkda_chunk_fwd.chunk_precond_kda`. It follows pinned FLA naive
semantics and consumes q/k without hidden normalization. Both main and ATK
states are independent, optional inputs and fresh outputs.

The declared kernel domain is B1/2,T1..4096,H1..32,K=V128, contiguous FP32,
a5/cce, block_dim1..4. See `contract.json` for all numeric gates and executed
support scope. BF16, backward and unimplemented FLA options fail explicitly.
References and native tests cover the 340M training geometry H8/K=V128; K120 is unsupported.
Native inprocess execution is qualified on950PR_9589 V100 with the selected
CANN9.2.0 toolchain:112 full-chain cases and40 API checks acrossbd1..4.
All14 stage hashes agree across bds. Same-card Torch NPU eager timing is
recorded in the report. SSH board/aclnn launchers, checkpoint and Triton
execution remain untested.

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

For actual NPU acceptance, first verify the selected environment and acquire
the shared device lock externally. Use one block_dim per process:

```bash
python verify_native.py --block-dim 4 --mode suite --output <ignored-output>/native-bd4
python verify_native.py --block-dim 4 --mode perf --output <ignored-output>/perf-bd4
```

The verifier precompiles all five vendors, then starts with B1/T4096/H8.
Repeat the suite independently atbd1/2/3 and compare all stage hashes.
See [evidence](evidence/README.md) for receipts, the original gate-prefix
failure, the FP32 compensated-sum repair and the exact qualification scope.
