# a5.kda_fwd_stable

KDA 定长前向，**门控算术改成对深衰减数值稳定的形式**。派生自 ascriptor 的 `a5.kda_fwd`
（只读引用，见仓库 `AGENTS.md` §3：需要改 kernel 时在本仓 `kernels/` 下建自己的单元，
不改 ascriptor 仓）。

## 为什么存在

`a5.kda_fwd` 的 gate kernel 只写出 `eg = exp(g_cumsum)`，于是下游必须取比值：

| kernel | 用法 | 深衰减时 |
|---|---|---|
| scores | `k / eg` | `eg → 0` 后是 `k/0 = inf`，再乘 `qg = 0` → **NaN** |
| wy | `eg_last / eg` | 两者同时为 0 → **`0/0` = NaN** |

`eg` 在 `-ln(FLT_MIN_NORMAL) ≈ 87.3` 处下溢（fp32 非正规数被硬件 flush 到 0）。
**信息在 `exp()` 落盘那一刻就没了，下游无论怎么写都救不回来。**

而按 fla 自己的 KDA 初始化（`A_log = log(U(1,16))`、`dt ∈ [0.001, 0.1]`），chunk 内门控
跨度约 **94** —— 越过这条线。也就是说上游 kernel 配 fla 的默认初始化**直接不可用**。
见 `docs/matrix/gaps.json` 的 `gate-range-beyond-declared`。

## 改了什么

三个 kernel，矩阵乘结构、流水、掩码一概不变：

- **`kernels/gate.py`** — 额外写出 log 空间的 `g_cumsum`。cumsum 本来就在寄存器里，
  只多一次 store。顺带解决 `a5.kda_bwd` 九个检查点之一（`fwd-caches-not-emitted`）。
- **`kernels/intra.py`** — 收 `g_cumsum`，按**逐通道中点** `m[d] = gc[L-1,d]/2` 构造
  `exp(gc-m)` 与 `exp(m-gc)`，把 `k / eg` 换成乘法。乘积 `exp(gc_i - gc_j)` 与原版同义
  （m 抵消），但两个因子的指数都被压到 `±span/2`，上限因此翻倍。
- **`kernels/wy.py`** — 收 `g_cumsum`，`kg` 由 `eg_last/eg` 改为 `exp(gc_last - gc)`，
  先减后指数，恒 ≤1。`qg`/`kexp` 仍是绝对量，下溢到 0 即正确。

`inverse` reuses the upstream source. The standalone unit now derives `recurrent`
locally to repair the independent Aqk handoff defect described below.

**这不是"完全不分解"。** 用 matmul 在通道维求和就必须分解；能做到的最好是**对称地**分解。
要彻底去掉上限得把 64×64 的 tile 再按行列分块（每对子块用各自的 m），那是更大的改动，
等实测需要再做。

## 实测（Ascend950PR / CANN 9.2.0，2026-09-11）

`o` 对逐 token 递推 oracle 的相对 L2：

| chunk 内门控跨度 | upstream | stable |
|---|---|---|
| 1.11 | 2.956e-03 | 2.956e-03 |
| 11.14 | 2.939e-03 | 2.939e-03 |
| 44.56 | 2.975e-03 | 2.975e-03 |
| 66.84 | 2.925e-03 | 2.925e-03 |
| 89.12 | **NaN** | 2.924e-03 |
| 111.40 | **NaN** | 3.191e-03 |
| 133.69 | **NaN** | 3.053e-03 |
| 155.97 | **NaN** | 2.850e-03 |

重叠域内**四位有效数字相同** —— 数学没变。之后上游失效，本单元精度不退化。

## 怎么跑

The public runtime API below still selects the original recurrent kernel and
retains its existing guards. A5K-01 does not change that dispatch:

```python
from ascend_fla.ops.kda import chunk_kda_fwd
o, state = chunk_kda_fwd(q, k, v, g, beta, initial_state=h0,
                         output_final_state=True, impl="stable")   # impl 默认就是 stable
```

验证见 `tests/test_kda_layer_npu.py` 与 `benchmarks/` 下的扫描脚本。

## A5K-01: continuous Aqk handoff

The standalone `unit.py` and native `verify_repair.py` use
`kda_sub45_aqk_repaired_kernel`. Its producer and consumer select the L1 slot with
`((pair_idx - pair_begin) * C + c_idx) % 2`. The two event credits now match the
two rotating slots across odd-C head boundaries. Arithmetic, casts, other buffer
indices and event operations are unchanged. WY explicitly declares `depth=2`,
preserving its former default under the accepted library API.

See [the repair report](../../../../docs/research/kda_aqk_handoff_repair.md) for
the source pins, exact measured scope and performance results. Native correctness
uses independent Torch CPU FP32 recurrence and pinned FLA CPU naive. The existing
`rtol=atol=0.02`, relative-L2 `<=0.05` budget is unchanged. Native profiling uses
Torch NPU tensors, synchronized samples and three baseline/candidate/baseline
rounds. The native timing excludes compilation and public API validation.

The canonical runner is exported from the accepted kernels checkout; references
do not import the DSL. Set `ASCRIPTOR_WORKSPACE` and `PYTHONPATH` to the accepted
workspace, and `FLA_KDA_NAIVE` to the pinned FLA `fla/ops/kda/naive.py` for dual
oracle verification. Run from the repository root:

```sh
unit=kernels/projects/a5/kda_fwd_stable
python "$unit/run.py" reference --output tmp/A5K-01/reference
python "$unit/run.py" check --launcher sim --case aqk_c1_gqa --block-dim 1 --output tmp/A5K-01/sim
python "$unit/run.py" check --launcher pipesim --case aqk_c1_gqa --block-dim 1 --output tmp/A5K-01/pipesim
python "$unit/repair_diagnostics.py" --output tmp/A5K-01/diagnostics.json
```

On the assigned, health-checked and locked A5 device, use the task's external
environment configuration for CANN, device visibility and the task-private cache:

```sh
python "$unit/verify_repair.py" grid --case kimi_t4096 --block-dim 4 --output tmp/A5K-01/full-bd4
python "$unit/verify_repair.py" grid --block-dim 1 --output tmp/A5K-01/grid-bd1
# Repeat the grid in separate processes for block_dim 2, 3 and 4.
python "$unit/compare_repair.py" --root tmp/A5K-01 --output tmp/A5K-01/cross-bd.json
python "$unit/verify_repair.py" profile --block-dim 4 --warmup 10 --repeat 50 --output tmp/A5K-01/profile
```

`repair_diagnostics.py` preserves the original and qg-only negative controls;
only reduced shapes run in the pipe model. Complete workloads run on hardware
first. The public dispatch and the A5 wave gates require their own integration
decision; this unit's evidence does not change them.
