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
Repeat `--case` to select several cases for that block dimension; the default
still runs its full grid. `--torch-oracle` checks the pinned recurrence on the
selected NPU; `--profile-torch-oracle` also times it. Native FP32 matmul/conv HF32
settings are disabled and recorded. Match inputs and device for timing comparisons.
After native correctness acceptance, `measure.py` measures three synchronized
GDN/PGDN/GDN rounds at T=4096 and T=1024, with FP32 and BF16 public inputs.
The GDN cost baseline executes the five PGDN-owned chunk kernels with equal
normalized read/write keys and no ATK; normalization, zero-state allocation,
dispatch and output casting on NPU are timed. It is a GDN-equation cost baseline,
not GDA-02 implementation performance or an equivalent PGDN algorithm.
GDN is checked against its pinned naive semantics; PGDN against both pinned
recurrence and independent block solve, before and after timing.
All six operators are prepared before execution. The completed measurement uses
block_dim2; correctness covers block_dim1/2 in separate processes/builds.
A full chosen hardware workload must precede reduced diagnostic simulations when
hardware is available. A model check is never hardware acceptance.

`contract.json` contains60 cases; large cases are reference/native workloads.
Name a bounded case for sim/pipesim rather than running that entire grid in a model.
All machine bindings belong to ignored external configuration.

[validation.json](validation.json) binds the submitted computational sources to
60 reference cases, seven bounded model cases, dual-reference metrics, both
six-kernel CANN builds and **60 passed native cases / 120 public dtype calls**.
All 17 actual stage arrays and all three public outputs in both dtypes match
bitwise across block_dim1/2 for every one of the 30 input sets.
Three same-card GDN/PGDN/GDN rounds passed at both lengths and dtypes, retaining
all 1800 measured samples plus pre/post numerical checks. PGDN medians are
143.65–144.07 ms at T1024 and 566.28–566.91 ms at T4096 (block_dim2).

See [native evidence](evidence/native/README.md) for exact shape, environment,
per-round medians, raw report/log locations and the two disclosed initialization
warnings. No TensorFlow backend support is inferred from these runs.
Standalone board/aclnn launchers, CUDA/Triton and weights remain untested.
Container CPU validation also passes all 104 PGDN tests.
