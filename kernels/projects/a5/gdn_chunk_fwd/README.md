# Scalar-gated GDN chunk forward

This standalone A5 CCE unit implements non-GQA Gated DeltaNet, with scalar
beta/log-decay, zero initial state and no q/k normalization. The fixed chunk
length is 64 and K=V=128. Inputs to the kernel unit are FP32, including exact
widenings of BF16 values. The public `ascend_fla.ops.gdn_chunk_fwd.chunk_gdn`
entry accepts BF16 or FP32 q/k/v and returns matching output dtype plus
optional FP32 state. It rejects unsupported layouts, GQA, tails, scale and
initial state explicitly. There is no reference fallback or backward.

The five-stage graph uses launch-ordered GM edges. CPU references use a
unit-lower triangular solve independently of the device's WY substitution.
FLA's recurrent CPU oracle is an additional comparison, loaded only by tests
and the native benchmark through an explicit local file argument.

```sh
python run.py reference
python run.py check --launcher aclnn --case b1_t4096_h16_g1 --block-dim 2
python run.py check --launcher board --board a5 --case b1_t4096_h16_g1 --block-dim 2
python benchmark.py --fla-naive <local-naive.py> --block-dim 2 --output <receipt.json>
python benchmark.py --fla-naive <local-naive.py> --case b1_t1024_h16_g0.03 --profile --output <profile.json>
```

Compile every stage before CANN first resolves an operator. Each block_dim
or source candidate runs in its own process and cache. `benchmark.py`
compares all checkpoints, both CPU references and the public BF16/FP32
entry before recording synchronized host-inclusive NPU timings. The full
shape runs precede any reduced simulator diagnostic.

Hardware scope and numerical/performance receipts are recorded in the
contract and `validation.json`. The sibling GDN-2 unit supplied generic WY/scan/output mechanics
(adaptation provenance in `stages.py`); it is not a runtime dependency or
source of GDN qualification. The upstream a5.gdn_fwd unit is only an ABI
comparison source. See the assigned research document for equations,
range analysis and the distinct state/layout/scale/rounding choices.

To reproduce paired performance on one reserved device, keep separate
immutable baseline/candidate checkouts and run:

```sh
python sandwich.py --baseline <baseline-repo> --candidate <candidate-repo> --fla-naive <local-naive.py> --output <sandwich.json>
```

The baseline revision and exact stage hashes are in `validation.json`.
The caller supplies the device environment and exclusive lock. No machine
configuration, saved golden tensors or network dependency is shipped.
