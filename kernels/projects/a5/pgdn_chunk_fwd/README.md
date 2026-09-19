# PGDN forward + ATK, A5 CCE

`ascend_fla.ops.pgdn_chunk_fwd.chunk_pgdn(q,k,v,g_atk,g,beta_atk,beta,...)`
implements grouped, inference-only PGDN with zero initial main and ATK states.
K=V=128, T is a positive multiple of64 up to4096, HV is a positive multiple of H,
and block_dim is1 or2. Both optional final states are independently shaped:
`S:[B,HV,128,128]`, `A:[B,H,128]`.

FP32 normalization follows pinned FLA **naive**: `x/max(norm(x),1e-12)`.
It intentionally differs from the FLA Triton chunk path's near-zero formula and
BF16 materialization. No Triton/CUDA/checkpoint validation is claimed.
See [the frozen ABI and range review](../../../../docs/research/pgdn_chunk_fwd_gate_range.md)
for defaults, rejected options, key roles, decomposition and evidence scope.
The correction reads normalized k_read; state writes use k_write=k_read*M.

Use the accepted environment and pinned source identities from the workspace
contract. Place all outputs under ignored task scratch. Standalone examples:

```sh
python run.py reference --output "$TASK_TMP/reference"
python run.py check --launcher pipesim --sim-processes fork \
  --case grid_r1_c1_bd1 --timeout 300 --output "$TASK_TMP/pipesim"
python verify.py --launcher sim --case norm_tiny_bd1 \
  --fla-naive "$FLA_PGDN_NAIVE" --output "$TASK_TMP/dual-reference.json"
```

`run.py` is the canonical portable helper and verifies both independent leaves
and actual composition. `verify.py` adds A/B metrics and output-storage quality
for the actual returned model outputs. Both generate inputs/references at run time.
`benchmark.py` verifies actual native CCE and optionally measures synchronized
latency; it requires current device health/occupancy checks and an externally
held shared lock. Keep each block dimension in a fresh process/build environment.
A full chosen hardware workload must precede reduced diagnostic simulations when
hardware is available. A model check is never hardware acceptance.

`contract.json` contains60 cases; large cases are reference/native workloads.
Name a bounded case for sim/pipesim rather than running that entire grid in a model.
All machine bindings belong to ignored external configuration.

[validation.json](validation.json) binds the submitted sources to 60 reference
cases, seven bounded model cases, dual-reference metrics and both six-kernel
CANN builds. Native execution and performance remain unverified.
