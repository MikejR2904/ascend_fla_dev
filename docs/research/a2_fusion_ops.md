# A2-40：整网融合算子的清单、ABI 与验收设计（主机侧）

> 任务：`docs/pm/tasks/A2-40.md`（issue #54）。只做调研和设计，**没有写或改任何 kernel 和模块代码**。
> 本文不含任何真机运行。**文中所有性能说法一律是"待测"**，不是结论（AGENTS.md §6 铁律一）。
> 引用的实测数字都注明 SoC 和形状。其中 A5 上的数字不能拿来当 A2 的预期（AGENTS.md §6「结论不跨 SoC 继承」）。

## 0. 证据来源

| 来源 | 修订 | 怎么读的 |
|---|---|---|
| fla | `v0.5.2`（`9c8e42e`） | `fla/layers/kda.py`、`fla/ops/kda/{chunk,fused_recurrent,gate}.py`、`fla/modules/{l2norm,fused_norm_gate}.py`、`fla/modules/conv/{short_conv,triton/*}.py`、`fla/ops/utils/softplus.py`、`fla/ops/common/gate.py` |
| ascriptor library | `90cfcdc`（`agent/compatibility.json` 的 pin） | `docs/migration/a2-a3-handoff.md`、`docs/cce-support.md`、`ascriptor/a2.pyi` |
| ascriptor kernels | `b3b3f9c`（同上 pin，按 SHA 浅取） | `algorithms/{matrix_normalization,gated_approximations,chunk_row_scan,convolution_layout_pipeline,a2_*}`、`projects/a5/kda_fwd` |
| 本仓 | `main@4c11e04` | `ascend_fla/{layers/kda.py,modules/*,ops/kda/*}`、`kernels/projects/a5/*`、`docs/matrix/*.json`、`docs/handoff.md` |

ascriptor 按 AGENTS.md §3 从 gitcode 取得，未做任何修改。

**性能数字的出处**。`decode-layer-overhead` 里的"整层 458 µs / KDA 算子 18% / 层其余 67%"
是 **A5（Ascend950PR，CANN 9.2.0）** 上测的，而且测试形状是 `hidden=2048 / H16 / HV32 / bd1`，
**不是** Kimi-Linear 的真实形状（`hidden=2304 / H=HV=32`，见 `models.json`）。下文引用这组数时，
只用来说明"为什么要做"，不用来预测 A2 上能快多少。

---

## 1. 总览

| 对象 | 现状 | 与 fla 有无静默差异 | 可复用单元 | 建议归属 | 工作量（agent 时） |
|---|---|---|---|---|---|
| `causal_conv1d` | torch：`F.conv1d` 后接 `F.silu`，cache 每步返回新张量 | 有两处（§2 的 D3、D5） | 没有能直接用的；swiglu 的 sigmoid 写法可借 | A2-41 | 前向 10–14，反向另加 8–12 |
| `fused_rms_norm_gated` | torch，分两趟，fp32 | 无（数值只差舍入顺序） | swiglu 的 sigmoid；`row_l2` 只能借规约骨架 | A2-41 | 前向 8–12，反向另加 8–10 |
| `qk_l2norm_gate` | 层里做，fp32，然后降 bf16 | **有**（D1、D2），另有 API 面的静默面（§2.2） | `row_l2` 借骨架；`kda_fwd_stable/gate.py` 接 inlet | 语义修正走新任务 A2-44；融合走 A2-42 | 见 §6 |
| `packed_projection` | 9 个 `nn.Linear` 分开算 | 无 | 不需要 kernel（用 torch_npu matmul） | A2-43 | 6–10（含 checkpoint 兼容） |

---

## 2. 与 fla 的语义差异

### 2.1 差异表

"静默"一列的含义：结果错了或偏了，但不报错、不出 NaN，形状也对。

| # | 对象 | fla v0.5.2 | 本仓当前 | 后果 | 静默 |
|---|---|---|---|---|---|
| D1 | l2norm 公式 | `x / sqrt(Σx² + 1e-6)`（`modules/l2norm.py:43,105`；`fused_recurrent.py:155`） | `F.normalize(x, eps=1e-6)` = `x / max(‖x‖, 1e-6)`（`layers/kda.py`） | ‖x‖ 大时两者相对差 ≈ `1e-6/(2‖x‖²)`，低于 bf16 分辨率。**‖x‖ 小时分叉**：‖x‖=1e-4 时 fla 输出范数 0.0995，本仓输出 1.0。算例：`1/sqrt(1e-8+1e-6)=995`，乘 1e-4 得 0.0995。全零行两边都给 0 | **是**（仅限近零行，比如 padding token 或刚初始化的卷积输出） |
| D2 | decode 路径 q/k 的精度 | `fused_recurrent_kda` 在 kernel **内部**用 fp32 归一化，归一化后的 q/k **不落 bf16** | 层里先 fp32 归一化，再 `.bfloat16()`，`fused_recurrent_kda` 进门时又升回 fp32 | 本仓 decode 比 fla 多一次 bf16 舍入。这是精度差，不是语义差 | 否（在预算内），但它决定 §5.3 为什么"不逐位" |
| D3 | `use_short_conv=False` | q/k/v 各过一次 `F.silu`（`layers/kda.py` 的 else 分支） | **不做 silu**，直接进 l2norm | 这个开关的整条语义都不同。Kimi-Linear 用短卷积所以碰不到，但层的构造参数接受它 | **是** |
| D4 | 短卷积的中间舍入 | fp32 累加，fp32 上做 silu，**舍入一次**到 x.dtype（`conv/triton/kernels.py:87-130`） | `F.conv1d` 先出 bf16，再对 bf16 做 `F.silu`：**舍入两次** | 精度差，在 bf16 预算内 | 否 |
| D5 | 卷积 cache 的写法 | 解码步**原地**更新 cache（`short_conv.py` 的 `step`："updates the cache in-place"） | 每步返回新张量，`cache["conv_state"]` 换成新对象 | 数值相同。但**图捕获要求地址固定**，这是 A2-43 要解决的两种写法之一（§7） | 否（是 API 面的差异） |
| D6 | chunk 路径的"in_kernel" | `use_qk_l2norm_in_kernel` / `use_beta_sigmoid_in_kernel` 在 chunk 路径上是 `ChunkKDAFunction` **里面的独立 launch**（`ops/kda/chunk.py:56-62`），只有 gate 真正融进了 `kda_gate_chunk_cumsum`。fused_recurrent 路径则三步全在 kernel 里 | 三步都在层里做 | 见 §2.2。**fla 的"in_kernel"主要是 API 语义，不等于单 kernel 融合**，别按字面理解去设计 | — |
| D7 | softplus | 阈值 20：`x > 20` 时直接返回 x（`ops/utils/softplus.py`） | `F.softplus`，默认 `threshold=20` | 一致 | 否 |
| D8 | 门控的对数底 | chunk 路径把 cumsum 乘以 `RCP_LN2` 进 log2 空间（`chunk_fwd.py:47-60`） | 自然底，stable gate kernel 直接写 `g_cumsum` | 内部表示不同，对外语义一致 | 否 |
| D9 | beta | `sigmoid(b)`，`allow_neg_eigval` 时为 `2·sigmoid(b)` | `sigmoid(b)`；`allow_neg_eigval=True` 报错 | 一致，缺的能力会显式报错 | 否 |
| D10 | o_norm | `x·rstd·w (+b)` 再 `·sigmoid(g)`，fp32 计算，输出 dtype 跟随 x（`fused_norm_gate.py:88-108,473`） | 同样的顺序，fp32 | 一致 | 否 |
| D11 | 门控 kernel 的入参命名 | fla 的 `g` 在 `use_gate_in_kernel=True` 时是**激活前**的原始值 | ascriptor 的 `kda_sub1_gate_kernel(g_raw, …)` 里，`g_raw` 是**激活后**的 per-token log 衰减 | **命名陷阱**。§4.3 的 ABI 把激活前的输入命名为 `f_raw`，避开这个撞名 | 设计阶段要注意 |

### 2.2 `qk-l2norm-not-in-kernel` 的静默面到底有多大

`gaps.json` 把这条定为"本仓唯一错了不报错的语义缺口"。逐个调用方式核对后，结论比这句话窄，也更具体：

1. **照抄 fla 的调用会报错，不会静默**。`chunk_kda` 和 `fused_recurrent_kda` 的签名里没有 `**kwargs`，
   传 `use_qk_l2norm_in_kernel=True` 会得到 `TypeError`。
2. **真正的静默面**是调用方为了消掉这个 `TypeError`，把不认识的 kwargs 删掉，然后传入原始的 q/k/g/beta。
   原始值在数值上完全合法，门控跨度的闸也未必拦得住，因为 `g_raw` 的尺度小时跨度也小。
3. **能用廉价的定义域检查把大部分静默面变成报错，不必等 kernel 融合**。KDA 语义下：
   `g = -exp(A)·softplus(·) ≤ 0` 必然成立；`beta = sigmoid(·) ∈ (0,1)`；归一化后的 `‖q‖, ‖k‖ ≤ 1`（容差按 bf16 取）。
   原始的 `g_raw`、`b_raw` 几乎必然有正值或越出 [0,1]，原始 q/k 的范数几乎必然 > 1。
   这是**启发式**检查（范数 < 1 的原始 q 抓不到），要如实这么写。它在 chunk 路径上只多几次规约；
   在 decode 路径上每次规约都是一次 launch，正是 §7 要压的开销，所以 decode 路径不装这个检查，交给 A2-42 的融合来解决。
4. 所以这条 P1 分成两件事：**语义修正**（加 fla 式的原始输入 API + 定义域检查，不动 kernel，建议新开 A2-44）
   和**性能融合**（A2-42，需要 kernel 批次审批）。前者不必等后者。

---

## 3. ascriptor 侧可复用单元（pin `b3b3f9c`）

**先说 A2 上最要紧的一条**。A2（dav-c220）的向量单元是 **tensor 级**指令（UB→UB，带 mask/repeat），
A5（c310）是**寄存器级**（`library/docs/cce-support.md:345`："c310's vector unit is register-level"）。
本仓所有 A5 单元（`kda_fused_recurrent`、`gdn2_*`、`kda_fwd_stable`）都用 `@vf` / `Reg` / `RegList` 写，
**不能按原样搬到 A2**。A2 的写法可以参照 `gated_approximations/swiglu_*`，它的 kernel 里写着
"This tensor-vector recipe supports A2/A3"。`a2.pyi` 也导出了 `Reg` 名字，
但 A2 后端能不能真的发射寄存器级代码，**本次没有核实**，A2-41 动手前要先查。

| 单元 | 路径 | 设备 / 状态 | 能用的部分 | 缺的部分 |
|---|---|---|---|---|
| `matrix_normalization.row_l2` | `algorithms/matrix_normalization` | 只有 A5（release_scope）；board passed | 向量侧的"行平方和 → 开方 → 除"骨架 | ① 它是**矩阵乘 + 行归一化**的融合（fp16 输入、fp32 输出），而我们的 l2norm 前面隔着卷积和 silu，**融不进投影 GEMM**。② **没有 eps**（README："The source has no epsilon"），fla 的公式要 `+1e-6`。③ 归一化粒度是整行、`N%256=0`，我们要的是每个头 128 列一段。④ 不支持 A2 |
| `gated_approximations/swiglu_{f32,bf16}` | `algorithms/gated_approximations` | **A2/A3**（tensor-vector 写法）；各 stage 都是 untested | sigmoid 的写法 `x/(1+exp(-βx))`，fp32 计算，最后 RNE 一次。对 x→-∞ 是安全的：`exp` 上溢到 inf，`x/inf=-0`，§8 有量程分析。**这是四个对象里唯一在 A2 上现成的向量 recipe** | 只做逐元素运算，没有规约；值域契约只到 `x∈[-1,1]`，要扩展并重测；A2 上本身没有任何实测 |
| `gated_approximations/gelu_*` | 同上 | A2/A3 | — | 四个对象都用不到 |
| `chunk_row_scan` | `algorithms/chunk_row_scan` | 只有 A5 | chunk 内 cumsum 的参考实现 | KDA 的 gate kernel 本来就做 cumsum，这里只用来对照 |
| `convolution_layout_pipeline/*` | `algorithms/convolution_layout_pipeline` | A5 native；A3 有历史 board | — | **用不上**：它们是 NCHW 的 2D cube 卷积（3×3、stride/dilation 2）。depthwise、W=4 的 1D 因果卷积没有通道规约，是纯向量运算 |
| `a2_adamw` / `a2_anchor_mask` | `algorithms/a2_*` | A2/A3 | 只能当 A2 tensor-vector 写法的样例（DBuff、owner 按 32B 对齐） | 语义不相关 |
| `projects/a5/kda_fwd` 的 gate kernel | `projects/a5/kda_fwd/kernels/gate.py` | 只有 A5 | `qk_l2norm_gate` 融合的落点：本仓 `kda_fwd_stable/kernels/gate.py` 的 `kda_sub1_gate_stable_kernel` 已经派生自它 | 没有 `A_log` / `dt_bias` 入口（D11），写法是 A5 的 |

**本仓的先例**（全部只有 A5，只作参考，不写；GDN-2 的 reserved 路径不碰）：

- `gdn2_short_conv_decode`：B=T=1、D=6144、W=4、bf16 的打包 q/k/v 卷积，加 silu 和 cache 更新，一次 launch。
  验收写法可以直接借：**`new_cache` 按逐位比较**，y 用 `atol=1/64, rtol=2e-3, rel_L2≤1e-3`。
- `gdn2_fused_decode`：**在 kernel 内完成门控激活和 q/k 归一化**，是 `qk_l2norm_gate` 融进 decode 的直接先例。
  两点值得照抄：① `decay_rate = -exp(A_log)` 在打包期预算好，kernel 里不再算 `exp(A_log)`（只适用于推理）；
  ② softplus 用无上溢的形式 `max(x,0) + log(1+exp(-|x|))`。
- `gdn2_norm2_w12_swiglu`：RMSNorm 和 GEMM 融在一次 launch 里。它的 v1 在 kernel 级快 1.0772×，
  但因为真实 cache 递推误差达 1.523e-2 被拒（A5，`modules-are-torch-not-kernels`）。
  **教训：融合验收必须包含多步递推的累积误差，单步对得上不够。**

---

## 4. ABI 草案

四个对象的共同约定：

- 公开布局一律 **token-major**（`[B, T, …]`），跟 KDA 算子的公开 ABI 一致，省掉一次转置。
- 输入要求连续，不连续就报错，不悄悄 `.contiguous()`（AGENTS.md §5：缺算子包的机器上做不到）。
- 输出预先填 NaN，要求每个元素都被覆写（沿用 ascriptor 的约定）。
- `block_dim` 的上限由契约声明，超出就报错。**A2 的物理核数和 `block_dim` 上限都未知**，
  要在 A2-10 上机时查实，再写进契约（AGENTS.md §2）。契约写好之前，入口先只接受 `block_dim=1`。
- 所有报错信息都要写清：哪条约束没满足、实际值是多少、该换什么路径。

### 4.1 `causal_conv1d`

```
causal_conv1d(x, weight, bias=None, cache=None, *, activation="silu",
              update_cache_inplace=True, block_dim=1) -> (y, cache)
```

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| `x` | `[B, T, D]` | bf16（判定模式下是 fp32，见 §5.1） | 允许 packed q/k/v，此时 `D = 3·key_dim`（Kimi 是 12288） |
| `weight` | `[D, W]` | 同 x | 从 `nn.Conv1d` 的 `[D,1,W]` 零拷贝 view 过来 |
| `bias` | `[D]` 或 None | 同 x | Kimi-Linear 的 `conv_bias=False` |
| `cache` | `[B, D, W]` | 同 x | 与 fla 同形状。给出时**原地更新**（D5，图捕获需要） |
| `y` | `[B, T, D]` | 同 x | fp32 累加，fp32 上做 silu，**舍入一次**（对齐 fla，消掉 D4） |

标量：`T`、`D`，W 固定为 4（两个目标模型都是 4），`activation ∈ {None, "silu"}`。

**分两期交付**：
① **decode 形态**：`T=1` 且必须给 cache。这是 `gdn2_short_conv_decode` 的推广版，把 D 从 6144 改成符号维，改写成 A2 的 tensor-vector 写法。
② **prefill 形态**：任意 T，cache 可有可无，T < W 时左侧补零。

反向（dx、dw、db）单独立项，训练才需要。

报错条件：

- `W != 4`
- `activation` 不在支持集合里
- `cache` 的形状不是 `[B,D,W]`，或 dtype 和 x 不同
- T>1 却走 decode 形态，或 decode 形态没给 cache
- 输入不连续
- `D` 不满足 32B 对齐的 owner 切分（bf16 下 `D % 16 == 0`；具体切分规则 A2 上要实测）

### 4.2 `fused_rms_norm_gated`

```
fused_rms_norm_gated(x, gate, weight, bias=None, *, eps=1e-5,
                     activation="sigmoid", block_dim=1) -> y
```

| 张量 | 形状 | dtype | 说明 |
|---|---|---|---|
| `x` | `[B, T, HV, 128]` | bf16 | KDA 算子的输出 `o` |
| `gate` | `[B, T, HV, 128]` | bf16 或 fp32 | `g_proj` 的输出。**不在 host 侧降精度**，层的注释里写了原因 |
| `weight` | `[128]` | fp32 | 打包期从 bf16 升一次（做法同 `gdn2_fused_decode`） |
| `y` | 同 x | 同 x | `x·rsqrt(mean(x²)+eps)·w (+b) · act(gate)`，fp32 计算，**舍入一次** |

标量 `eps` 按 aclnn 约定走 `double`（AGENTS.md §5）。`activation ∈ {"sigmoid", "swish"}`，Kimi 用 sigmoid。

报错条件：最后一维不是 128；`gate` 形状和 x 不同；`weight` 长度不对；activation 不支持。

建议 A2-41 顺手做 `o_proj` 前的 layout：让 y 直接以 `[B, T, HV·128]` 连续写出，省掉层里那次 `reshape` 和 `.to(o_proj.weight.dtype)`。

### 4.3 `qk_l2norm_gate`

分成**两个 ABI**，对应 §2.2 的两件事。

**(a) 公开 op 层（A2-44，不动 kernel）**：给 `chunk_kda` 和 `fused_recurrent_kda` 加上 fla 式的原始输入入口。

```
chunk_kda(q, k, v, g, beta, ..., A_log=None, dt_bias=None,
          use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
          use_beta_sigmoid_in_kernel=False, check_domain=True, ...)
```

- 参数名、默认值、`dt_bias` 的形状 `[HV·K]` 都和 fla v0.5.2 一致。区别是本仓**不静默忽略**：
  flag 为真但缺 `A_log` 就报错；`allow_neg_eigval`、`lower_bound`、`safe_gate` 仍然报错。
- 在 A2-42 落地之前，实现就是在 op 内部调用和层里现在一样的 fp32 计算，只是从层**挪进 op**。
  l2norm 改用 fla 的公式（消掉 D1）。
- `check_domain=True` 时，在 flag 为假的那几路上检查 `g ≤ 0`、`beta ∈ [0,1]`、`‖q‖,‖k‖ ≤ 1+tol`（只在 chunk 路径上做，理由见 §2.2 第 3 条）。
  报错信息要写明"看起来传的是激活前的原始值，请传 `use_*_in_kernel=True`"。
- `tol` 要按 bf16 舍入标定：先实测 bf16 归一化向量的最大范数，再往外留余量。不能拍脑袋定一个数。

**(b) kernel 层（A2-42，需要 kernel 批次审批）**：

| 路径 | 融合点 | 新增输入 | 新增输出 | 说明 |
|---|---|---|---|---|
| chunk 前向 | ① 新增一个**前置向量 kernel**：对 q/k 做 l2norm、`beta` 做 sigmoid，写出 bf16 的 q̂/k̂ 和 fp32 的 `rstd_q`/`rstd_k`（`[B,T,H]`）；② gate kernel（`kda_sub1_gate_stable_kernel`）加 `A_log[HV]`、`dt_bias[HV,K]` 两个 inlet，在 cumsum 之前做 `-exp(A)·softplus(f_raw + dt_bias)` | `f_raw [B,T,HV,K]`（**不叫 `g_raw`**，见 D11）、`b_raw [B,T,HV]`、`A_log`、`dt_bias` | `rstd_q`、`rstd_k`、`g`（激活后、cumsum 前的值，反向要用）、**每个 chunk 的门控跨度** `span [B,HV,C]` | q̂/k̂ 必须先写出来：scores/wy 是 cube kernel，要从 GM 读 bf16 的 q/k。这和 fla 在 chunk 路径上的做法一致（D6） |
| chunk 反向 | l2norm 反向 `dx = rstd·(dy − ŷ·(ŷ·dy))`；gate 反向 `df_raw = dg·(−exp(A))·sigmoid(f_raw+dt_bias)`、`dA_log = Σ dg·g`、`d dt_bias = Σ_{B,T} df_raw`；`db_raw = dβ·β(1−β)` | 前向存下的 `rstd_*`、`g`、`beta` | `dq_raw`、`dk_raw`、`df_raw`、`dA_log`、`ddt_bias`、`db_raw` | fla 在 `chunk.py:166-170` 和 `kda_gate_bwd_kernel` 里用的就是这些式子 |
| decode | 融进 `kda_fused_recurrent`：进门就在 UB 里把 q/k 归一化（**保持 fp32，不落 bf16**，消掉 D2）；门控激活和 sigmoid 也放进去 | 同上。推理可以改收 `decay_rate = -exp(A_log)`（仿 `gdn2_fused_decode`） | 无 | 同时去掉 host 侧 `fused_recurrent.py` 里的 `.to(fp32)`、`repeat_interleave`、`permute().contiguous()` 这几次 launch（`decode-call-overhead` ②） |

**门控跨度检查要搬家**。`_check_gate_range` 现在在 **launch 之前**用 host 侧的 `g` 算跨度。
门控融进 kernel 之后，host 不再持有 `g`，而为了检查在 host 上再算一遍 `g` 又会把融合省下的东西赔回去。
做法是：gate kernel 顺手写出 `span[B,HV,C]`（cumsum 本来就在寄存器里，只是多一次规约和一次小写回）；
host 在**把 o 交给调用方之前**读出 `max(span)`，超过 `MAX_GATE_SPAN[impl][path]` 就报错，把 o 丢掉。
这样仍然是"不满足就报错"（AGENTS.md §7），只是时点从 launch 前挪到了返回前。
**decode 路径不需要这个检查**（`kda_fused_recurrent/README.md`：recurrent 形式逐 token 只用 `exp(g_i)`，跨度没有上限）。

### 4.4 `packed_projection`

**不写 kernel。** 投影本身就是 GEMM，交给 torch_npu 的内置 matmul。理由：

1. 我们自己写 cube kernel 就会落在 A2 的 split-K FP32 cube 缺陷上（A2-01 / A2-11）。内置 matmul 走 vendor kernel，不受这个缺陷影响。
2. 打包之后是一次更大的 GEMM，数学上不变。

Kimi-Linear（hidden=2304、H=HV=32、K=V=128）一层有 **9 个** `nn.Linear`，不是 handoff 里说的 7 个：
f_proj 和 g_proj 各有两段。

| 组 | 成员 | 输入 | 打包后形状 | launch 数 |
|---|---|---|---|---|
| P1 | q, k, v, f₁, g₁, b | `x [B,T,2304]` | `W [12576, 2304]` = 3×4096 + 128 + 128 + 32 | 6 → 1 |
| P2 | f₂, g₂（g₂ 带 bias） | 分别是 f₁、g₁ 的输出（各 128 维） | 两次小 GEMM，或一次 `bmm`（2 个 batch） | 2 → 1 或 2 |
| — | o | 融合 norm 的输出 | 不变 | 1 |

合计从 9 次降到 3–4 次。

**权重不复制**。先分配一块连续的 packed buffer，再把每个 `nn.Linear.weight` 设成它的**行切片 view**
（行切片是连续的，`nn.Linear` 可以直接用）。state_dict 的名字不变，也不额外占显存。
这是对 GDN-2 先例的有意偏离：那条路径保留了两份权重，18 层多出 1.03 GB（`modules-are-torch-not-kernels`）。

报错条件：打包时各成员的 dtype 或 device 不一致；有人在打包之后替换了某个成员的 `weight` 对象，
这会让 view 关系断掉。检测方法是 forward 时比较 `data_ptr` 是否仍然落在 packed buffer 内，不一致就报错，
不静默退回分开算。

---

## 5. 验收判据

共同纪律（AGENTS.md §6）：

- 正确性**一律在 fp32 下判定**。每个单元都要有 fp32 I/O 的判定模式；bf16 模式只额外检查"舍入一次"。
- 报 `max_abs_diff`、相对 L2、bf16 ulp 距离，不报"OK"。
- 双 oracle：fla 的纯 torch 公式在 CPU 上用 fp64 或 fp32 计算，外加 torch_npu 的组合实现。两者之间的差异也要报出来。
- **形状在真实形状上测**：Kimi 是 `hidden=2304、H=HV=32、D=4096/12288`。
  对 `(T, B·HV 与 block_dim 的关系, D 的对齐余数)` 取笛卡尔积的边界点，不要每个参数各扫一遍。
- **`bd=1` 和 `bd=max` 的输出必须逐位相同**。这是不需要参考值的硬判据，能直接抓切分 bug。

### 5.1 `causal_conv1d`

| 检查 | oracle | 形状 | 阈值 |
|---|---|---|---|
| y，fp32 判定模式 | fp64 的 depthwise 因果卷积再做 silu | `B∈{1,2}`、`T∈{1,2,3,4,5,64,65}`（覆盖 T<W、T=W、T 跨 chunk），`D∈{4096,12288}`，有 cache 和无 cache | rel L2 ≤ 1e-6，max_abs ≤ 1e-5·max\|y\| |
| y，bf16 模式 | 同一个 fp64 结果做 RNE | 同上 | **ulp 距离 ≤ 1**，并报告距离为 1 的元素占比。这条能抓出 D4 那种"舍入两次" |
| 新 cache | 由 x 和旧 cache 的拼接截取 | 同上 | **逐位相等**（纯搬运，沿用 `gdn2_short_conv_decode` 的判据） |
| 原地语义 | — | decode 形态 | 调用前后 `cache.data_ptr()` 不变 |
| prefill→decode 串接 | 一次调 T 个 token，对比逐 token 调 T 次并传 cache | T=16 | y 与 cache **逐位相同** |

### 5.2 `fused_rms_norm_gated`

| 检查 | oracle | 形状 | 阈值 |
|---|---|---|---|
| y，fp32 判定模式 | fp64 实现 fla 的公式 | `B·T ∈ {1, 64, 4096}`，HV=32，gate 覆盖 `\|g\|` 到 100（验证量程，见 §8 的 R7） | rel L2 ≤ 1e-6 |
| y，bf16 模式 | fp64 结果做 RNE | 同上 | ulp 距离 ≤ 1 |
| 极端输入 | 同上 | x 全零的行；x 的元素大到 1e4 | 结果有限，且与 oracle 一致 |

### 5.3 `qk_l2norm_gate`：怎么证明"融进去"和"在层外做"一致

**证明分三层，每层回答一个问题。**

**第一层：交接张量逐位对照**（回答"融合 kernel 算的东西和层里算的是不是同一个东西"）。

先定义交接张量 `H = (q̂_bf16, k̂_bf16, g_fp32, β_fp32)`，也就是 chunk kernel 真正吃进去的东西。
A2-44 把层里的 l2norm 换成 fla 的公式之后，层外路径产出 `H_ref`；融合路径的前置 kernel 和 gate kernel 以调试输出的形式写出 `H_fused`。

- q̂、k̂：两边都是 fp32 计算后做 RNE，差别只来自平方和的规约顺序，所以**判据是 ulp 距离 ≤ 1**，同时报告不等元素的占比。
  注意是占比，不是"全部逐位相等"：规约顺序不同会在 RNE 平局点翻转最低位，这属于同义但不逐位相同（AGENTS.md §6）。
- g、β：fp32 的超越函数实现不同，**判据是 rel L2 ≤ 1e-6**，并单独报 `max_abs`。
- 测试行要包括 ‖x‖ ∈ {0, 1e-4, 1e-2, 1, 30} 的构造行。这组值专门盯 D1：**如果融合 kernel 错用 `max(‖x‖,eps)` 的写法，就会在 1e-4 那一行上露出来。**

**第二层：给定相同的 H，下游逐位相同**（回答"融合有没有顺带改掉下游"）。

把 `H_fused` 直接喂给未融合的 chunk kernel，对比融合路径的 o 和 final_state。**判据是逐位相同**：
下游的 kernel 和算式没变，只是 H 的来源变了。不相同就说明融合时误改了下游。

**第三层：端到端对 oracle**（回答"整体还在预算内"）。

- chunk 路径：o 和 final_state 对 fp32 递推 oracle，沿用现有契约预算（o 的 rel L2 约 3e-3 量级）；
  **另外要求融合路径与层外路径之间的 rel L2 ≤ 1e-3**，比各自对 oracle 的误差小一个量级以上。
- decode 路径：融合后 q̂/k̂ 不再落 bf16（D2），所以**和当前层外路径不会逐位相同，这是有意的**，而且更接近 fla。
  判据分两条：① 对"l2norm 在 fp32 下完成、不经过 bf16"的 fp32 递推 oracle，rel L2 ≤ 1e-6，与现有 `kda_fused_recurrent` 的 1e-7 级别同档；
  ② 与当前层外路径的差异如实报数，预计是 bf16 舍入量级（待测），**不设成逐位判据**。
- 多步递推（`gdn2_norm2_w12_swiglu` v1 的教训）：decode 连续 64 步，比较 final_state 的累积误差，
  判据与 prefill→decode 一致性测试相同。
- 反向：`dq_raw`、`dk_raw`、`df_raw`、`dA_log`、`ddt_bias`、`db_raw` 对 fla 纯 torch autograd（CPU fp32），
  预算沿用 `kda_bwd_stable` 契约（dq 0.05、dg 0.25）。另要求与当前层外 autograd 路径的 rel L2 ≤ 1e-3。
- 门控跨度检查：构造跨度 = 闸值 ± 1 的输入。闸值 +1 时必须报错，**而且调用方拿不到 o**；闸值 −1 时正常返回。

**A2-44 的定义域检查**也要逐条钉进测试：把原始 `g_raw`、`b_raw`、原始 q 分别单独传入，各自必须报错，
报错信息要点名是哪一路；传入正常激活后的值时不能误报。`tol` 的标定过程要写进测试注释。

### 5.4 `packed_projection`

| 检查 | 判据 |
|---|---|
| 打包前后层的输出 | fp32 下 rel L2 ≤ 1e-6，并如实报告是否逐位相同（GEMM 的切分可能变，所以不强求逐位） |
| checkpoint | 打包前 `state_dict()` → 打包 → `load_state_dict()` → 再导出，前后逐位相同；每个成员的 `weight.data_ptr()` 都落在 packed buffer 内 |
| 反向 | 各成员权重的梯度，与不打包时的 rel L2 ≤ 1e-6 |
| 显存 | 打包前后参数显存之差 = 0（没有复制） |

---

## 6. 工作量与拆分（可以直接落到看板）

单位是 agent 时，按"有 A2 机器、A2-10 已经给出核数和算子包覆盖"估算，**不含等机器的时间**。
A2 上 ascriptor 的 tensor-vector 写法还没有本仓自己的先例，所以第一个单元多算一倍学习成本。

| 任务 | 范围 | 依赖 | gate | 估计 |
|---|---|---|---|---|
| **A2-44（新）** | 纯主机侧 + CPU 测试：① `chunk_kda` / `fused_recurrent_kda` 加 fla 式 flags 和 `A_log`/`dt_bias`，三步从层挪进 op；② l2norm 改用 fla 公式（D1）；③ `use_short_conv=False` 补 silu（D3）；④ chunk 路径的定义域检查；⑤ 按 §5.3 第三层写 CPU 测试 | 无 | 无（`needs.npu=false`） | 10–16 |
| **A2-41** | `causal_conv1d`（先做 decode 形态，再做 prefill 形态）和 `fused_rms_norm_gated` 的 A2 自编译单元，只做前向；同时核实 A2 能否用 `@vf` | A2-40、A2-10 | `machines:a2` | 30–45（反向另立项，16–22） |
| **A2-42** | `qk_l2norm_gate` 融进 KDA：chunk 路径的前置 kernel、gate inlet 和 `span` 输出；反向链；decode 融进 `kda_fused_recurrent` | A2-40、A2-K1、**A2-44**（拿它的测试当第一层 oracle），以及 A2 上已有可用的 KDA kernel | `kernel-batch-approval` | 35–55 |
| **A2-43** | 层侧：① `packed_projection`（零拷贝 view）；② 固定 cache 地址（原地更新，或在图末尾拷回，§7）；③ 图捕获；④ 用 profiler 拆出整层 decode 的每项耗时，**先测再动** | A2-40、A2-15 | `machines:a2` | 20–30 |

**有一条要请用户或 PM 拍板，本文不自行决定**：A2-41 的四个对象都是纯向量运算，不经过 cube，
所以技术上不受 split-K FP32 cube 缺陷的影响。但 AGENTS.md §2 写的是"在 A2-11 之前，A2 上**任何**算子结论都不算数"。
要不要对纯向量单元放宽这一条，是规则层面的决定。默认按原规则办，不放宽。

---

## 7. `decode-layer-overhead` 的处置顺序

原来的顺序是：① 图捕获 → ② 合投影 → ③ 去掉 l2norm 的 dtype 往返。**建议保留 ①② 的先后，但改成有条件的，并把 ③ 并入 A2-42。**

**为什么图捕获排第一**：

- 层侧那 67% 在 A5 上判断为"launch 开销之和"。**这个判断本身也要在 A2 上重新 profile**，A2-43 的第④步就是做这个。
- 如果 67% 确实是 launch 开销，图捕获能一次藏掉**所有** launch，包括永远不会融合的那些，而且不改数值
  （GDN-2 在 A5 上连续 4 个 token 的 replay，logits 和两类 cache 都与 eager **逐位相同**，handoff §1.6）。
- **前提是地址固定**。本仓的卷积 cache 每步都换新张量，KDA 的 final_state 也是新分配的。两种解法：
  一是改成原地更新（D5，与 fla 一致）；二是像 GDN-2 那样，在图的末尾把新 cache 拷回固定的输入地址。
  前者省掉每步一次拷贝，后者不改模块语义。A2-43 从两者中选一个，并实测拷贝的代价。
- A5 上的先例是 torch_npu 2.10 的 `NPUGraph` 能捕获本仓的 ctypes→aclnn 自定义算子（handoff §1.6）。
  GDN-2 整网固定 prompt-cache 时快了 1.756×，真实生成时快了 1.41–1.65×。
  **这些是 A5、GDN-2 的数，在 A2 上能不能捕获、效果如何，全部待测。**

**为什么合投影排第二，而且是有条件的**：

- 图捕获成功的话，launch 开销已经被藏掉，合投影剩下的收益只在设备侧。T=1 时 GEMV 受带宽限制，
  打包前后读的权重字节数一样，**收益可能很小，待测**。
- 图捕获在 A2 上不可用的话，**合投影就升为第一**：它把 9 次 launch 降到 3–4 次，在 eager 模式下直接生效。
- 不管图捕获成不成，合投影对 **prefill 和训练**都有用，因为那两条路径用不了图捕获（形状动态、要建 autograd 图）。

**为什么 ③ 不单独立项**：decode 里 l2norm 的 dtype 往返，在 A2-42 把 l2norm 融进 `kda_fused_recurrent` 之后会连同 D2 一起消失。
单独去掉往返（比如让层直接交 fp32）虽然能省两次 cast，但它会改变 chunk 路径交给 cube kernel 的 dtype，得不偿失。
A2-44 保持现在的往返不动。

---

## 8. 量程表

判定规则：所有 `exp`、除法、开方、规约，逐处写出指数或分母的范围，再判"有限"和"准"。
"≤1，下溢到 0 就是正确结果"这类结论也写下来，免得下一个人重查（AGENTS.md §6）。

| # | 对象 | 算式 | 参数范围 | 失效方式 | 结论 / 要求 |
|---|---|---|---|---|---|
| R1 | conv | `Σ_{w<4} x·w (+b)`，fp32 累加 | bf16 乘积，4 项 | 只有 \|x·w\| > 3.4e38 才会上溢 | 实际不可达。**安全** |
| R2 | conv | `silu(y) = y / (1 + exp(−y))` | y 无界 | y < −88.7 时 `exp(−y)` 上溢为 inf，`y/inf = −0`，正是 silu 的极限值；y > 88.7 时 `exp(−y)` 下溢为 0，结果为 y，也正确 | **按这个写法安全**。如果写成 `exp(y)/(1+exp(y))`，y > 88.7 时得到 `inf/inf = NaN`，**禁止这种写法**。反向的 `σ(1−σ)` 同理，用 σ 来表达 |
| R3 | conv | cache 更新 | — | 纯搬运 | 必须逐位相等 |
| R4 | norm | `mean(x²)`，128 项 fp32 求和 | x 来自 KDA 算子的 bf16 输出 | \|x\| > 1.8e19 才上溢 | **安全** |
| R5 | norm | `rsqrt(ms + eps)`，eps=1e-5 | ms ≥ 0 | 分母 ≥ 1e-5，所以 rstd ≤ 316 | **安全**。**eps 不能被 kernel 省掉**：`row_l2` 就没有 eps（§3） |
| R6 | norm | `y·w (+b)` | 权重有限 | — | **安全** |
| R7 | norm | `sigmoid(gate) = 1/(1+exp(−gate))` | gate 来自 `g_proj`，带 bias，无界 | gate < −88.7 时 `exp` 上溢为 inf，`1/inf = 0`，正确 | 与 R2 相同。判定用例里 gate 要覆盖到 ±100 |
| R8 | l2norm | `Σ q²`，128 项 | q 来自卷积和 silu，bf16 | 实际不上溢；\|q\| < 1e-19 的元素平方后下溢为 0，这时 eps 主导，结果正确 | **安全** |
| R9 | l2norm | `1/sqrt(s + 1e-6)` | s ≥ 0 | 分母 ≥ 1e-3，所以 rstd ≤ 1000 | **安全**。必须用 fla 的这个公式（D1）。反向 `rstd·(dy − ŷ(ŷ·dy))` 的放大倍数也 ≤ 1000，有限 |
| R10 | gate | `softplus(u)`，u = f_raw + dt_bias | u 无界 | 朴素写法 `log(1+exp(u))` 在 u > 88.7 时 `exp` 上溢，得到 g = −inf，进 cumsum 后跨度变成 inf，下游出 NaN | **必须用阈值形式**（u > 20 时直接取 u，对齐 fla 和 torch）**或无上溢形式** `max(u,0) + log(1+exp(−|u|))`（同 `gdn2_fused_decode`）。u → −∞ 时 `exp(u)` 下溢为 0，softplus = 0，绝对误差不超过 e^u，**可以接受** |
| R11 | gate | `exp(A_log)` | A_log 是训练参数；初始化 `log(U(1,16))` ≤ 2.77 | A_log > 88.7 才上溢 | 初始化时安全，训练中无硬上界。**不单独装闸**：它会通过 R13 的跨度闸间接暴露 |
| R12 | gate | `−exp(A)·softplus(u)`（每个 token 的 g） | 每个 token 的量级可达 16·u | 只要 u 有限就有限 | 本身安全，量程问题落在 R13 |
| R13 | gate | chunk 内 cumsum，之后 `exp(±span/2)` | 64 项负数求和 | 这是**已有的门控跨度缺陷**：stable 实现前向 155、反向 105（A5 实测，`gate-span-still-bounded`） | 门控融合后，跨度检查改由 kernel 写出 `span`、host 在返回前判断（§4.3）。**A2 上的上限要重新测，不能沿用 155/105** |
| R14 | gate（decode） | `exp(g_t)`，g_t ≤ 0 | 恒 ≤ 1 | 下溢为 0 表示 state 被完全遗忘 | **下溢到 0 就是正确结果**，不需要闸 |
| R15 | beta | `sigmoid(b_raw)` | 无界 | 同 R7 | **安全**（按 R7 的写法） |
| R16 | l2norm（decode） | q 乘 `scale = 128^-0.5` | — | — | **安全** |
| R17 | proj | GEMM，K=2304（P1）或 128（P2），fp32 累加 | bf16 输入 | 上溢实际不可达 | **安全**。走 vendor matmul，所以**不受** A2 split-K FP32 cube 缺陷影响；如果改成自写 cube kernel，就会受影响（§4.4） |

全部 17 处中，**有硬性写法要求**的是 R2、R7、R10、R15（`exp` 上溢的方向）和 R5、R9（eps 不能省），
**需要装闸**的只有 R13（沿用已有的闸，但要换实现位置，并在 A2 上重测上限）。
其余是"实际不可达"或"下溢到 0 就正确"。

---

## 9. 本文没有做、留给后续任务的事

- 所有性能数字（launch 次数、每项耗时、图捕获和合投影的收益）→ A2-43 第④步，用 profiler 在 A2 上测。
- A2 的核数、UB 大小、`block_dim` 上限、算子包覆盖 → A2-10。
- A2 后端能不能发射 `@vf` / `Reg` 代码 → A2-41 动手前先查。
- 门控跨度在 A2 上的实际上限 → A2-42（或 A2-15）。
- 纯向量单元要不要从 A2-11 的总闸中放宽 → 用户或 PM 决定（§6）。
