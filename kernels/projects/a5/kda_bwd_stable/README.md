# a5.kda_bwd_stable

KDA 定长反向，**把成对衰减的分解锚点从 `g_last` 换成 `g_last/2`**。派生自 ascriptor 的
`a5.kda_bwd`（只读引用，见仓库 `AGENTS.md` §3）。与 `a5.kda_fwd_stable` 成对使用。

## 为什么存在

前向改好之后反向还是不能用。`a5.kda_bwd` 的 `finalize_pre` / `finalize_post` 把 chunk 内
成对衰减 `exp(g_i − g_j)` 分解成两个因子，锚点取 chunk 末行 `g_last`：

```
rscale = exp((g − g_last)·ln2)     g − g_last ≥ 0   → 最大 exp(+span)
cscale = exp((g_last − g)·ln2)     ≤ 0              → 最小 exp(−span)
```

`span` 是 chunk 内门控跨度。**`rscale` 是真上溢**（不是前向那种下溢）：fp32 与 bf16 的
上溢线都在 `ln(MAX) ≈ 88.72`，而 `q_scaled` / `k_scaled` 是 **bf16** GM 输出
（`finalize_pre.py` 的签名），所以 span 超过 88.72 时它们直接是 `inf`；配对的 `kg` 同时
下溢到 0，下游四个矩阵乘里 `inf × 0 = NaN`。

本机复现（纯算术，不需要 NPU）：

| span | `rscale` 最大 | `cscale` 最小 | `q_scaled`(bf16) | `kg`(bf16) |
|---|---|---|---|---|
| 1.92 | 6.82e+00 | 1.47e-01 | ok | ok |
| 66.84 | 1.07e+29 | 9.37e-30 | ok | ok |
| 88.72 | 3.39e+38 | 2.95e-39 | ok | ok |
| **94.00** | **inf** | 1.50e-41 | **inf** | **0** |
| 155.97 | **inf** | 0 | **inf** | **0** |

94 正是 fla 默认初始化的 KDA 层给出的跨度。也就是说：**前向修好后，反向仍然在默认初始化
下不可用。** 见 `docs/matrix/gaps.json` 的 `bwd-gate-range-overflow`。

## 改了什么

两个 kernel，各 2 行：`gmid = g_last × 0.5`，两个指数都以它为锚。

```diff
-        tmp <<= g - glast
+        tmp <<= g - gmid
-        tmp <<= glast - g
+        tmp <<= gmid - g
```

**为什么锚点可以任选**：`rscale_i · cscale_j = exp(g_i − g_j)` 与锚点无关，而下游
`finalize_pair` 的四个矩阵乘**全部成对**：

| finalize_pair 的 matmul | 带 rscale 的一侧 | 带 cscale 的一侧 | finalize_post 里补的因子 |
|---|---|---|---|
| `Mqk @ kg.T` → `qk_left` | — | `kg = k·cscale` | `rscale * qkl` |
| `Mqk.T @ q_scaled.T` → `qk_right` | `q_scaled = q·rscale` | — | `cscale * qkr` |
| `Mbase @ kg.T` → `s_base` | — | `kg` | `rscale * sbase` |
| `Mbeta.T @ k_scaled.T` → `t_beta` | `k_scaled = k·rscale` | — | `cscale * tbeta` |

`finalize_post` 里 `rscale` / `cscale` 一共只出现这四处（逐行核对过），每一处都与配对量
相乘，所以常数逐项抵消 —— **恒等变形，数学没变**。两个文件必须用同一个锚点，因为
`finalize_pre` 产出的三个张量要和 `finalize_post` 补的因子配对。

改完两个因子各压到 `±span/2`，上限正好翻倍到 `2 × 88.72 ≈ 177`。

## 其余七个 kernel 为什么不用改

它们里面的指数都 ≤1，下溢到 0 正是「完全衰减」的正确结果，不存在 `0/0` 或 `0×inf`：

| kernel | 指数 | 范围 | 下溢后 |
|---|---|---|---|
| `inverse_epilogue` | `exp(g·ln2)` | ≤1 | `dq`/`kexp` → 0，即该行完全衰减，正确 |
| `inverse_epilogue` | `exp((g_last−g)·ln2)` | ≤1 | 同上。注意它**本来就是先减后指数** |
| `scan_fused` | `exp(g_last·ln2)` | ≤1 | 整 chunk 完全衰减，正确 |

`k_exp = k·exp(g)` 往下只进 `inverse_dainv` 的 `dw @ kexp.T`，对侧 `dw` 不带 `exp(−g)`，
所以没有补偿项可丢。

## 静态代价

`ascriptor check` 的 IR op 数（**不是周期数**，只用来界定改动规模）：

| kernel | 上游 | stable | 差 |
|---|---|---|---|
| `finalize_pre` | 247 | 251 | +4 |
| `finalize_post` | 368 | 372 | +4 |

多出来的是 `gmid <<= glast * MID`（2 个 RegList lane），且它在 64 行循环**外**，
每 chunk 只算一次。

## 怎么跑

```python
from ascend_fla.ops.kda import chunk_kda
o, state = chunk_kda(q, k, v, g, beta, initial_state=h0,
                     output_final_state=True, impl="stable")   # 默认就是 stable
```

`impl` 同时选前向与反向两套 —— **不要混用**。混用会让门控检查的上限对不上实际会失效的
那一侧（前向 stable + 反向 upstream 时，跨度 94 能过检查却在反向吐 NaN）。

## 实测（Ascend950PR / CANN 9.1.0，经本仓 runtime 桥，2026-09-11）

### 有限性：六个梯度里非有限元素的个数

| 跨度 | 1.12 | 74.83 | 89.35 | 125.09 | 149.66 | 159.71 | 169.76 | 174.23 |
|---|---|---|---|---|---|---|---|---|
| 上游 | 全有限 | 全有限 | **六项全坏** | 全坏 | 全坏 | — | — | — |
| 本单元 | 全有限 | 全有限 | 全有限 | 全有限 | 全有限 | 全有限 | 全有限 | dq/dk/dbeta/dg 各 64 个坏 |

上游在 89.35 一步跨过去就是 `dh0` 16384/16384、`dk`·`dv`·`dg` 各 8192/8192 —— 与
`ln(MAX) = 88.72` 吻合。本单元到 **169.76** 仍全部有限，**174.23** 才开始坏，且只坏 64 个
元素（`gate_span` 报的是**逐通道最大值**，边缘上只有最深那个通道越界），与理论上限 177.4
吻合。`recommended_limit = 160` 落在实测通过的 159.71 与首次失败的 174.23 之间。

### 精度：上游 `a5.kda_bwd` 契约的五个 case

预算取契约的 `comparison`（默认 0.05 / `dk` 0.15 / `dg` 0.25），**五个全过**。括号里是上游
反向的同一指标：

| case | dq | dk | dv | dbeta | dg | dh0 |
|---|---|---|---|---|---|---|
| single_chunk | 1.282e-02 | 7.330e-02 | 3.089e-03 | 3.407e-03 | 1.094e-01 | 2.977e-03 |
| multi_chunk | 2.239e-02 *(2.238)* | 9.168e-02 | 3.504e-03 | 3.337e-03 | 1.649e-01 | 2.589e-03 |
| grouped_heads | 2.152e-02 *(2.149)* | 9.298e-02 | 3.307e-03 | 9.192e-04 | 1.441e-01 | 2.785e-03 |
| gentle_decay | 3.662e-03 *(3.674)* | 5.383e-03 *(5.396)* | 3.821e-03 | 3.928e-03 *(3.854)* | 6.211e-03 *(6.199)* | 5.163e-03 |
| grouped_idle_cores | 1.442e-02 *(1.441)* | 8.685e-02 *(8.687)* | 2.927e-03 | 6.476e-03 | 1.555e-01 | 2.542e-03 |

**三位有效数字全同，只第四位有差**（最大是 `gentle_decay` 的 `dbeta`，差 1.9%）。这正是
换锚点应有的效果：实数上恒等，但中间量落在 bf16 GM 上，舍入发生的位置变了。

## 验证状态

- ✅ `ascriptor check`：两个 kernel 各 0 error / 0 warning（本机跑，纯 Python）。
- ✅ 上溢机制：本机算术复现（见前面的表）。
- ✅ 抵消关系：逐行核对 `finalize_post` 里 rscale/cscale 的全部四处用法。
- ✅ 真机有限性 + 契约五个 case 的精度（见上）。
- ⏳ **默认初始化（跨度 ~94）下的整层反向未跑** —— 需要有 `ascend950` 算子包的机器
  （层里的投影/卷积/softplus 都是 torch_npu 算子，9.1.0 那台跑不了）。测试已写好：
  `tests/test_kda_layer_npu.py::test_default_init_backward_matches_cpu_reference`。
- ⏳ harness 集成同 `kda_fwd_stable`，见 `stable-unit-no-harness`。

## 层级验证（2026-09-11 补齐）

算子级证据齐了之后还差一项：**默认初始化下整层反向的精度**。它做不了的原因是机器 ——
KDA 层里的投影、causal conv、softplus、RMSNorm 全是 torch_npu 算子，缺 `ascend950`
内置算子包的机器上一个都跑不了（AGENTS.md §5）。换到有该算子包的 CANN 9.2.0 机器后补上。

门控跨度**确定性校准**到 94（平移 `A_log`，不靠默认初始化 —— 跨度是随机变量，见下），
B1/T128/H1/HV2/hidden256，对同一份权重的 CPU fp32 层（只把 KDA 算子换成逐 token 递推）
逐参数比对，预算 0.25：

| 量 | 相对 L2 |
|---|---|
| output | 4.694e-03 |
| dx | 9.024e-03 |
| `A_log` | 1.551e-01 |
| `dt_bias` | 6.542e-02 |
| `f_proj.0.weight` / `f_proj.1.weight` | 5.040e-02 / 5.153e-02 |
| 其余 14 项（q/k/v/o 投影、conv、norm） | ≤1.090e-02 |

**18 项全部在预算内。** `A_log` 与 `dt_bias` 比别的大一个量级是预期的 —— 它们的梯度都要
穿过 `exp` / `softplus`，深衰减下对 `g` 的扰动放大最厉害，与算子级曲线里 `dg` 的预算
（0.25，比 `dq` 的 0.05 宽五倍）是同一个原因。

同一次运行里的算子级宽域三档：跨度 46 → `dq` 2.889e-02、94 → 4.550e-02、104 → 4.782e-02。
`tests/test_kda_layer_npu.py` 8 项 + `tests/test_kda_bwd_deep_npu.py` 4 项 = **12 passed**。

### 跨度是随机变量，这改了闸的取法

我一开始把"默认初始化 ≈ 94"当常数用，结果整层反向那个测试的前置断言（跨度 >80）自己先挂了
—— 那个 seed 实测只有 55.2。跨度 ∝ `max_hv exp(A_log)`，而 fla 取 `A_log = log(U(1,16))`，
所以它是随机变量：同一个 seed 只要换一下 RNG 的消耗顺序就从 64.55 变 94.0，HV=8 实测 8 个
seed 落在 54.2~100.6，上界 `exp(A_log)≤16 × dt≤0.1 × 63 步 ≈ 100.8`。

两个后果，都已落地：

1. 测试用 `_calibrate_span` 平移 `A_log` **确定性地**标定到目标跨度，不指望默认初始化碰上。
2. **闸的下界由 100.8 定** —— `recommended_limit` 从 100 改成 **105**。低于 100.8 的闸会把
   默认初始化的层用我们自己的门控拒掉，那等于自断链路。上界仍由契约预算定（110 处 `dq`
   只剩 0.2% 余量，不取）。**闸是双边约束：上边界看精度实测，下边界看调用方真会送什么进来。**
