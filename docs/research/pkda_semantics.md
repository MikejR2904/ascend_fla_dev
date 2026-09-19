# PKDA / PGDN：权威来源、语义定义与量程判定

任务：PK-01（#59）。2026-09-17 更正—— PK-01.md 写的"fla 0.5.2 没有 pkda/pgdn，本仓无 oracle"
**在当时（fla 0.5.2）成立，现在不成立**。用户提供了一份新的本地 checkout
（`/Users/limjiunnbin/work/flash-linear-attention`，fla 0.6.0，HEAD `e52dbc0e`——
与 GD2-01 已经在用的 FLA pin **是同一个 commit**），里面有完整实现。

## 1. 权威来源（找到了，不是"没找到"）

- **论文**：*Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences*，
  [arXiv:2604.21100](https://arxiv.org/abs/2604.21100)，ICML 2026。
- **训练代码原仓**：<https://github.com/ntumm120/preconditioned-deltanet>（bundled fork of fla +
  用 [flame](https://github.com/fla-org/flame) 训练；340M/1B 配置与训练脚本都在）。
- **合入 fla 上游**：PR [fla-org/flash-linear-attention#950](https://github.com/fla-org/flash-linear-attention/pull/950)
  "[Model] Add Preconditioned Gated DeltaNet (PGDN) and KDA (PKDA)"，2026-06。
  后续有一个 bugfix：`c0068787` "[Fix] Prevent PGDN naive packed state leakage"。
  fla README 也记了这条（"2026-06 Add PGDN and PKDA... curvature-aware preconditioning via ATK"）。
  **78 个算子测试 + 7 个模型测试全过**（PR 描述里的测试计划），覆盖 dense/varlen/prefill、
  `safe_gate` 开关、transpose-state、`disable_recompute`、fp16/bf16/fp32、GVA。
- **代码位置**（都在 fla 里，不是训练仓专属）：
  - `fla/ops/atk/`：两个算子共用的 ATK 预条件 kernel（fwd/bwd）
  - `fla/ops/precond_gated_delta_rule/`：PGDN 的 chunk + fused_recurrent + naive
  - `fla/ops/precond_kda/`：PKDA 的 chunk（+ intra）+ fused_recurrent + naive
  - `fla/layers/precond_gated_deltanet.py`、`fla/layers/precond_kda.py`
  - `fla/models/precond_gated_deltanet/`、`fla/models/precond_kda/`（HF 风格 config，见 §5）

`docs/pm/PROTOCOL.md` §6 那条"Gemini 文档不是权威"依然成立——但这次不是引 Gemini 的公式，
是找到了 Gemini 文档描述的机制（TTR / ATK）对应的**真实、已发表、已合入上游 fla 的实现**。
Gemini 文档里"对角 Hessian 预条件 `p_t = λp_{t-1} + k_t⊙k_t`，`k̃_t = k_t/(p_t+ε)`"
是一个粗糙的转述，真实算法（见 §3）明显更谨慎——这条本身就是本文档要更正的一点。

## 2. 与 KDA 的关系：PKDA 是 KDA + ATK，不是新算子族

`chunk_precond_kda` 的公开签名（`fla/ops/precond_kda/chunk.py`）：

```
chunk_precond_kda(q, k, v, g, g_atk, beta_atk, beta, ...,
                   use_gate_in_kernel=False, safe_gate=False, lower_bound=None,
                   solve_tril_precision=None, disable_recompute=False, ...)
```

`q,k`: `[B,T,H,K]`；`v`: `[B,T,H,V]`；`g`（**channel-wise** decay）: `[B,T,H,K]`；
`beta`: `[B,T,H]`；`initial_state`: `[N,H,K,V]`。**这四项和本仓 `ascend_fla/ops/kda/chunk.py`
的 ABI 逐项对得上**——`use_gate_in_kernel`、`safe_gate`、`lower_bound` 三个参数名
直接照抄自 KDA 自己的门控实现（`fla.ops.kda.gate`），`solve_tril_precision`/
`disable_recompute` 也是 KDA chunk 实现里已有的开关。

**多出来的只有三项，都是 ATK 相关**：`g_atk` `[B,T,H]`（ATK 衰减，per-head 标量）、
`beta_atk` `[B,T,H]`（ATK 更新率）、`initial_A_state`/`final_A_state` `[N,H,K]`
（ATK 的对角状态，本仓已有的 `initial_state`/`final_state` 之外的第二个递推状态）。

**结论**：PKDA 是派生单元，不是新单元。可以直接复用本仓 `kernels/projects/a5/kda_fwd_stable`
与 `kda_bwd_stable` 的 chunk 分解（causal score / WY 三角解 / 状态扫描 / 输出组合），
只需要新增一步 ATK 预条件（见 §3），把它插进"算 `k_precond` 代替原始 `k` 参与三角解"
这一个环节。GD2-01 那套"5 launch"的分解方式可以直接借鉴。

## 3. PGDN 呢：是 GDN + ATK，但 GDN 本身在本仓还没打通

`chunk_precond_gated_delta_rule` 的签名：`g_atk,beta_atk: [B,T,H]`（同 PKDA），
但 `g,beta: [B,T,HV]`——**标量门控**（不是 GDN-2 那种 channel-wise），`v: [B,T,HV,V]`
支持 GVA（`HV>H`）。这正是本仓 `AGENTS.md` §2 记的 **`gdn-no-gqa`**
（"GDN 的正式 ABI 只有 `B,H,C`，没有独立 value-head 维度"）卡住的地方——PGDN 的 ABI
需要的正是 GDN 缺的那个 GQA/GVA 分组维度。

**结论**：PGDN 排在 GDN 自己的 ABI 缺口之后，不是"先做哪个都行"的并列关系——
在本仓能表达 GQA 分组之前，PGDN 无法落地成本仓能编译执行的单元。这条排 PK-03，标 gated。

## 4. ATK 预条件的真实算法（推翻 Gemini 文档那条"除法会下溢"的担心）

真实递推（`fla/ops/atk/chunk_atk_fwd.py` 的模块 docstring，与 `naive.py` 逐行核对一致）：

```
A_t = exp(g_atk_t) · A_{t-1} + beta_atk_t · k_t²        # 对角状态，[B,H,K]，逐 key-channel
ell = log(A_t + eps)
r   = ell - log_atk_scale                                # 偏离一个可学习/固定的中心，默认 -0.2
s   = r / (1 + |r|)                                       # "symmetric fast squash"，恒落在 (-1, 1)
M   = exp(-log(x) · s)                                    # 恒落在 (1/x, x)，默认 x=1.5 → (0.667, 1.5)
k_precond_t = k_t · M
```

**这不是 Gemini 文档转述的"`k̃_t = k_t/(p_t+ε)`"那种直接除法。** 真实实现把 `A_t` 先取对数、
再经过一个**恒有界**的 squash 函数，最后指数化成一个乘子 `M`。`s ∈ (-1,1)` 是 `r/(1+|r|)`
这个函数形式本身保证的硬数学界，不依赖 `A_t` 的大小——所以**不存在"`A_t` 太大或太小导致
`M` 溢出/下溢"这回事**，PK-01.md 当初担心的"没有 oracle 就不知道除法会不会崩"这条担心，
在看到真实算法后不成立：这个设计从一开始就是为了避免这个问题。

**仍然要测的两处**（不是"要不要溢出"，是"精度够不够"）：

1. `log(A_t + eps)`：`A_t` 是 `exp(g_atk)` 衰减 + `beta_atk·k²` 累加的正定量，其增长界与
   KDA/GDN 自己的状态衰减同构（`g_atk ≤ 0`），预期行为已有先例，但**要用本仓真实会遇到的
   `k` 量级（L2 norm 后的 `k`，见 `use_qk_l2norm_in_kernel`）重新测**，不能假设 fla 的
   GPU 测试参数直接适用。
2. `M` 恒有界不等于"乘出来的 `k_precond` 参与后续 KDA 三角解时精度不受影响"——
   `k_precond` 替代原始 `k` 进入 §2 说的 causal score / WY 三角解，那一段本仓已经踩过
   KDA 自己的门控跨度坑（`AGENTS.md` §6），**`k_precond` 改变了 `k` 的尺度，需要重新过一遍
   KDA 那张"每处 exp 判量程"的表，不能假设原来的判定还成立**。

`g`（KDA 的 channel-wise decay）本身的量程判定**不用重做**——PKDA 的 `g` 与本仓 KDA 的 `g`
同形状、同语义，本仓已有的门控跨度结论（前向 155、反向 105，`_calibrate_span` 那套方法论）
直接适用，除非 PK-02 测出 `k_precond` 改变了触发这条闸的条件。

## 5. 目标形状：有真实训练配置，没有已发布的预训练权重

搜过 HuggingFace（`precond`/`preconditioned` 关键词）与两个相关 GitHub 仓
（`ntumm120/preconditioned-deltanet` 训练代码仓、`MachineLearning-Nerd/icml26-preconditioned-deltanet`
独立复现审计仓），**没有找到已发布的 PGDN/PKDA 预训练 checkpoint**。训练仓提供的是从零训练的
配置与脚本（SlimPajama-627B，flame 训练框架），不是可下载的权重。

**这与 KDA/GDN-2 的情况不同**——那两个都有真实模型权重可做端到端 logits/cache 验证
（Kimi-Linear、gdn2-1.3B checkpoint）。PKDA/PGDN 目前只能做到：

- **强 oracle**：`fla.ops.precond_kda.naive`/`fla.ops.precond_gated_delta_rule.naive`
  （CPU fp32 递推）+ fla 自己的 Triton chunk 实现（GPU，78 个测试验证过）——**双重**参考，
  比多数已排期任务只有一个 naive.py 更扎实。
- **真实规模的形状，不是真实权重**：训练仓的配置表给了会真的被训练出来的形状
  （见下表），可以当"真实"形状用于性能与量程测试，但不能做端到端 logits 比对。

| Config | 规模 | hidden | heads | head_dim | layers |
|---|---|---|---|---|---|
| `precond_kda_340M` | 355M | 1024 | 8 | 128 | 24 |
| `precond_kda_1B` | 1B | 1792 | 14 | 120 | 24 |
| `precond_gated_deltanet_340M` | 340M | 1024 | 8 | 128 | 24 |
| `precond_gated_deltanet_1B` | 1B | 1792 | 14 | 128 | 24 |

来源：<https://github.com/ntumm120/preconditioned-deltanet>（README 的 Model Configs 表）。
`fla/models/precond_kda/configuration_precond_kda.py` 的默认值另有一套（`hidden_size=2048,
num_heads=16, head_dim=128, num_hidden_layers=24`），是 HF config 的占位默认值，不代表
论文实际训练过的规模——**排期用上表的训练配置，不用 HF config 默认值**。

## 6. 排期建议

- **PK-02（开放，可派）**：PKDA chunk 前向 —— 复用 `kda_fwd_stable`/`kda_bwd_stable` 的分解，
  新增 ATK 预条件步骤。验收对 fla naive **与** fla Triton chunk 双 oracle，`k_precond` 之后
  的门控跨度重新过一遍 §4 说的复核。规格见 `docs/pm/tasks/PK-02.md`。
- **PK-03（gated，等 GDN 的 GQA 缺口先解决）**：PGDN。规格写在 `docs/pm/tasks/PK-03.md`，
  但排在 GDN 自己的 ABI 之后——PGDN 的 ABI 需要 GDN 现在没有的 value-head 维度，
  在那之前无法落地。
- **两者共同的前提**：都没有已发布权重，端到端验证只能到"真实规模形状 + 双 oracle"，
  到不了"真实 logits/cache"那一级——这条要在两个任务的验收里显式声明，不假装有权重。

## 7. 已知陷阱（留给 PK-02/PK-03 的执行者）

- `A_state`（ATK 对角状态）是**第二个**递推状态，跟主状态 `S`/`final_state` 分开传递——
  写单元时两个状态的初始化、chunk 边界处的进位都要各自处理，别把两个状态的 shape 搞混
  （`A_state: [B,H,K]`，主状态 `[B,H,K,V]`，维度数不一样，混了会在形状检查这一步就报错，
  但如果凑巧广播成功就会是静默错误——参照 AGENTS.md §7，入口要显式检查两个状态的独立性）。
- `use_gate_in_kernel`/`safe_gate`/`lower_bound` 这三个参数是从 KDA 抄来的，语义要去读
  `ascend_fla/ops/kda/chunk.py` 里对应参数的现有实现，不要凭参数名猜测。
- 别把 PGDN 和 PKDA 的 `g`/`beta` 混着看——PGDN 是标量门控 `[B,T,HV]`，PKDA 是 channel-wise
  `[B,T,H,K]`，两者的门控跨度分析不能共用一份结论。
- **PGDN 与 PKDA 的 `naive.py` 对 q/k 的归一化不同**（2026-09-19，PM 对 pin 住的源码核实，PK-02 的 RISK
  `contradicts-handoff` 触发）：`precond_kda/naive.py` **不做**任何 q/k 归一化（只把 q 乘 scale）；
  `precond_gated_delta_rule/naive.py` **做** `F.normalize`（torch 默认 eps=1e-12）。两者的公开 chunk 路径都归一化
  （PKDA 在 `chunk.py:785` 以字面 True 强制，PGDN 的 `use_qk_l2norm_in_kernel` 默认 True），用 `l2norm_fwd`，
  即 `x/sqrt(Σx²+1e-6)` 并以输入 dtype 物化。所以 naive 与 chunk 的差别对 PKDA 是"是否归一化"，对 PGDN 是"eps 公式
  与物化 dtype"，别把一个的结论套给另一个。§4 里"`k` 经过 `use_qk_l2norm_in_kernel` 归一化之后的量级"指的是训练规模的
  工况，不代表 naive 会替调用方归一化。
