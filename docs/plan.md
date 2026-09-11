# 构建规划

制定于 2026-09-10。本文记录**为什么这样建**；具体工作规则见根目录 `AGENTS.md`，
当前支持状态见 `docs/matrix/`。

## 1. 上游调研结论

### fla 的结构（693 个 py 文件 / 19.3 万行）

```
fla/
├── ops/        43 个算子族，每族固定五件套：
│   │           chunk.py(分块训练) fused_recurrent.py(递推/解码)
│   │           wy_fast.py(WY表示) naive.py(纯torch参考) __init__.py
│   ├── common/   跨族共享原语：chunk_h / chunk_o / chunk_delta_h / chunk_scaled_dot_kkt / gate
│   ├── utils/    cumsum / solve_tril / matmul / index / pooling
│   └── backends/ 通用后端分派系统（BaseBackend / BackendRegistry / @dispatch）
├── layers/     43 个 nn.Module
├── models/     41 个 HF 兼容模型
├── modules/    layernorm / l2norm / rotary / causal_conv1d / fused_cross_entropy
└── utils/      _device.py(IS_NPU)、ascend_ub_manager.py
```

**关键发现：fla 的昇腾适配不是空白，而是已经相当成熟。** `triton_ascend` 后端已有
约 17.4k 行，覆盖 modules、ops/common、ops/utils、gla、kda、gated_delta_rule、
attnres；另有 `ascend-a2-ci.yml` 真机 CI（CANN 9.1.0 / torch 2.9 / triton-ascend 3.2.2）
和 `.agents/skills/fla-ascend-performance` 方法论。

**所以本仓的价值不在"填补昇腾支持的空白"，而在两处**：① 43 个算子族只有 4 个有
昇腾后端，覆盖率是真缺口；② ascriptor 是指令级编译，性能上限应高于 triton-ascend。
我们靠性能和覆盖率立足，不靠"第一个做昇腾适配"。

### 为什么最终不接 fla 接口

曾考虑做成 fla 的 out-of-tree 后端插件（注册进 `BackendRegistry`，priority 高于
`triton_ascend`）。最终选择纯算子库。

> **本节的原始论据已被第 0 期调研部分推翻，如实记录。** 当初的理由是"省掉 layout
> 转换开销"：fla 用 `[B,T,H,K]`，而 ascriptor 用定尺 `[B,H,C,64,128]`，接 fla 接口
> 就得每次调用都 permute。这对 GDN 成立，**但对 KDA 不成立** —— `kda_fwd`/`kda_bwd`
> 的公开张量本来就是 token-major `BTHK`/`BTHV`，内部自行 permute，且 state 为 FP32。
> 既然 KDA 已定为首个目标，这条论据在第一期用不上。

真正站得住的理由是**契约的诚实性**：现成算子的定尺约束（`L=64`、`K=V=128`、无 varlen、
无 tail、fp16 被拒）远窄于 fla 公共 API 承诺的范围。做 fla 后端意味着要在 verifier
里拒绝掉大部分调用，使用者感受到的是"时而加速、时而不加速"的不可预测行为；做独立库
则可以把约束直接写进 API 契约，调用方一开始就知道自己在用什么。次要理由是不承担跟随
上游 19 万行演进的维护成本。

代价是失去 fla 的测试集与模型生态。用两件事补偿：`naive.py` 仍作 oracle；
`models/` 用注入方式对齐上游模型规格（见第 4 节）。

## 2. ascriptor 侧的现成资产

`kernels/projects/a5/` 下已有 fla 三个核心算子族的 fwd + bwd：

| 单元 | 内容 |
|---|---|
| `gdn_fwd` / `gdn_bwd` | GDN 定尺五阶段，BF16 cube，含全部反向保存值 |
| `kda_fwd` / `kda_bwd` | KDA 五 kernel 前向 + 反向 |
| `delta_rule_fwd` / `delta_rule_bwd` | DeltaNet 前向 + 反向 |

`gdn_fwd` 的五个 kernel 与 fla 的 GDN chunk 路径逐级对应：

| ascriptor kernel | fla 对应 |
|---|---|
| `preprocess`(g_cumsum / decay_mask / strict_lower) | `gated_delta_rule/gate.py` + cumsum |
| `inverse`(严格下三角求逆) | `ops/utils/solve_tril.py` + `chunk_scaled_dot_kkt` |
| `recompute`(value_wu / k_cumdecay) | `gated_delta_rule/wy_fast.py` |
| `scores` | `ops/common/chunk_o.py` |
| `recurrent` | `ops/common/chunk_delta_h.py` |

**第一期算子侧主要是接线，不是从零写 kernel。** 另有可直接复用的算法单元：
`chunk_row_scan`（按 chunk 边界复位的行扫描 = chunk cumsum）、`matrix_normalization`
的 `row_l2`（l2norm）、`gated_approximations`（SwiGLU / GELU）。

### 首个目标为什么是 KDA 而不是 GDN

第 0 期把两者的 ABI 逐项对出来后，结论很清楚：**KDA 的契约离 fla 的语义近得多**。

| 能力 | `kda_fwd/bwd` | `gdn_fwd/bwd` |
|---|---|---|
| GQA 分组 | ✅ `HV % H == 0` | ❌ domain 只有 `B,H,C` |
| 公开布局 | ✅ token-major `BTHK`/`BTHV` | ❌ `[B,H,C,L,D]` |
| 非零 `initial_state` | ✅ FP32 正式输入 | ❌ zero only |
| 初始 state 梯度 | ✅ 输出 `dh0`、接受 `dht` | ❌ ABI 中不存在 |
| `final_state` dtype | ✅ FP32（同 fla 惯例） | ❌ BF16 |
| `block_dim` 上限 | 4 | 2 |
| 本地验证证据 | ❌ 无 `validation.json` | ✅ 齐全 |

只有最后一行对 GDN 有利，而那是可以补的（在本机跑 `run.py reference` 与 `sim` 即可，
不需要 CANN）。其余六项都是 ABI 层面的硬差距，要在 GDN 上补齐就是改 kernel。

一个附带好处：KDA layer 只依赖两个 module（`FusedRMSNormGated(sigmoid)` 与
`ShortConvolution`），GDN 还多一个 `RMSNorm`。KDA 的 `A_log`/`dt_bias` 是 per
value-head，原生支持 GVA。

首要目标模型相应从 Qwen3-Next 换成 **Kimi-Linear-48B-A3B**（H=HV=32，K=V=128，BF16），
它的形状完全落在 `kda_fwd` 的定尺域内，零阻塞缺口。Qwen3-Next 受 `gdn-no-gqa` 阻塞，
随 GDN 扩族移到第四期。

## 3. 第一期的真实风险：runtime 桥

ascriptor 的执行模型全是"落盘 + 独立进程"（详见 `AGENTS.md` §4）。它是 kernel
开发/验证框架，不是可嵌入的运行时算子库。本仓必须自建"常驻化 + torch 绑定 +
产物缓存"这一层。

**风险已消除（2026-09-11）。** 原先的判断是"六个 a5 单元的 `compile` stage 全是
`untested`，所以本地 aclnn 编译路径从未验证"。这个推断**有误**：unit runner 的
`LAUNCHER_STAGES` 把 `board` / `aclnn` / `pypto` 三个 launcher 都记为 `board` stage，
而 contract 里的 `compile` 指的是 `ascriptor compile` CLI（纯发射+编译、不执行）。
不过结论仍需自证 —— 实测 `chunk_row_scan --launcher aclnn` 真机通过（19.6s），
随后 `runtime/binding.py` 的 ctypes 零拷贝调用与 harness 路径逐位一致。

## 4. 全链路支持矩阵

目标是形成**模型 → 模块 → 层 → 算子**的支持矩阵，并与 fla 主仓的模型规格对齐。
三条做法约束：

**① 开发顺序自底向上，形状约束自顶向下。** "模型→算子"是看问题的视角，不是
施工顺序。算子先行，但算子要支持什么形状由模型 config 反推决定 —— 否则算子会在
玩具形状上全绿，到真机模型上挂掉。这就是第 0 期存在的理由。

**② 窄切片，不照搬。** KDA 这条链路实测只需要：

```
model:   Kimi-Linear 规格（kimi_linear）
layer:   KDA                                                  1 个
modules: FusedRMSNormGated(sigmoid), ShortConvolution/causal_conv1d   2 个
ops:     chunk(训练/prefill) + fused_recurrent(decode，待写)    2 个入口
```

43 个 layers / 41 个 models 一个都不要照搬。

**③ models 层用注入，不用重写。** 不复制 `modeling_*.py`。直接用 HF transformers /
fla 的模型定义，只把我们的 layer 替换进去 —— 规格自动跟上游对齐，零维护成本。

**④ 矩阵机器可读、CI 生成。** 学 ascriptor 的 `catalog.json` / `validation.json`
（带 contract 哈希与证据指向），不学 fla 的 README 表格。手写的矩阵三周后就没人信。

## 5. 分期

| 期 | 内容 | 验收 |
|---|---|---|
| **0** ✅ | 形状清单反推 + 缺口表 + 矩阵 schema | 已完成：`docs/matrix/` 三份 json，19 项缺口显式列出 |
| **1** ✅ | ① aclnn 编译 ✅ ② runtime 桥 ✅ ③ `kda_fwd` 接线 ✅ ④ KDA 本地基线 ✅ ⑤ torch_npu 基线 ✅ | 见下「第一期实测结果」。自编译算子在 bd=4 下比 torch_npu 组合快 2.4~4.4x |
| **2** ✅ | `kda_bwd`（九 kernel）+ 九个前向检查点 + autograd + KDA layer（含 2 modules） | 见下「第二期实测结果」。层级梯度对齐，训练步比 torch_npu 组合版快 4.8~5.7x |
| **3** 进行中 | model 注入（Kimi-Linear）+ KDA `fused_recurrent`(decode) ◐ + 矩阵 CI 生成 | 端到端跑通一个模型；chunk↔recurrent 互验通过 |
| **4** | GDN 扩族（含 GQA、token-major 布局、非零初始 state）+ DeltaNet + 性能迭代 | Qwen3-Next 可用；兑现"高效率算子" |

### 第一期实测结果（A5 / Ascend950PR，CANN 9.1.0，2026-09-11）

| 项 | 结果 |
|---|---|
| ① aclnn 本地编译 | ✅ `chunk_row_scan` 经 `--launcher aclnn` 真机通过，19.6s |
| ② runtime 桥 | ✅ `runtime/{binding,compile}.py`。ctypes + `aclCreateTensor` 直吃 NPU `data_ptr`，零拷贝。单 kernel 与 ascriptor harness 路径**逐位相同**（`max_abs_diff=0`） |
| ③ `kda_fwd` 接线 | ✅ 五 kernel 串联。单 chunk / 多 chunk / GVA(HV=2·H) 三形状的 `o` 相对 L2 3.29e-03~3.38e-03、`final_state` 1.94e-03~2.93e-03，均在 contract 预算 0.05 内 |
| ④ KDA 本地基线 | ✅ `kda_fwd` reference+sim（各 4 case）、`kda_bwd` reference+sim（各 5 case）全 passed |
| ⑤ torch_npu 基线 | ✅ 换到一台 opp 带 `ascend950` 的机器（CANN 9.2.0）后跑通。原先判定的"SoC 不支持"其实是**算子包安装差异** |

四个值得记住的坑（细节见 `AGENTS.md` §5 与 §6）：

1. **`aclCreateTensor` 在 `libnnopbase.so`**，不在 `libascendcl.so`。
2. **aclnn 的浮点 attr 是 `double` 而不是 `float`。** 按 `c_float` 传 4 字节，被调方从
   8 字节槽里读垃圾值，表现为 `scale` 近 0 —— 于是**只有用到 scale 的输出归零、不用的
   输出照常正确**。这个 bug 一开始被误判成 kernel 或 dtype 问题；定位靠的是拿同一个
   kernel 对比 harness 路径做单点二分。判 attr 类型一律看生成的 `aclnn_*.h`。
3. **缓存键不能比缓存贵。** `compile_kernel` 的签名要 `inspect.getsource` + sha256，
   每 kernel ~2ms；`kda_fwd` 一次前向查 5 个，于是"算缓存键"吃掉了端到端耗时的 **95%**
   （设备侧只占 2.3%）。加备忘后 10.119→4.930ms。
4. **一个算子名，一个进程，一份 build。** `ASCEND_CUSTOM_OPP_PATH` 按算子名查、第一个
   命中的胜出、每进程只解析一次，第二份 build 被**静默**忽略。这让同进程的 `block_dim`
   扫描全都执行了 bd=1 的二进制，四个耗时完全相同 —— 我据此写过"block_dim 无效"的错误
   结论。分进程后 bd=4 比 bd=1 快 3.9x。

这两条都是"先有推断、后看数据"的产物，已写成 `AGENTS.md` §6 的性能测量铁律。

### 第二期实测结果（A5 / Ascend950PR，CANN 9.2.0，2026-09-11）

| 项 | 结果 |
|---|---|
| 前置：fwd/bwd dtype 不一致的代价 | ✅ 已量化。保存值降精度 1.4e-03~1.6e-03，输出舍到 bf16 ~1.65e-03，叠加 2.2e-03~2.4e-03 且 ≈ √(A²+B²)。与契约已声明的输出精度同量级，可推进 |
| 九个前向检查点 | ✅ 逐个对单元的 `build_saved_forward`，四形状误差 6e-05~2.9e-03 |
| `kda_bwd` 九 kernel 接线 | ✅ contract 五个 case 全通过（含 bd=2/3），预算取 contract 的 comparison |
| autograd + KDA layer | ✅ 层级输出相对 L2 4.9e-03，17 个参数梯度 3.6e-03~2.2e-02 |
| 训练步性能 | ✅ bd=4 下比 torch_npu 组合版快 4.83x（kimi）/ 5.06x（qwen）/ 5.67x（T=4096） |
| 门控跨度（前向） | ✅ 已根治。`kda_fwd_stable` 把可用跨度从 ~80 扩到实测 155.97，重叠域内与上游四位有效数字相同；默认初始化（~94）下整层前向相对 L2 4.697e-03 |
| 门控跨度（反向） | ✅ `kda_bwd_stable`：有限性到 169.76（upstream 89.35 起六项梯度全坏），契约五个 case 全过且与 upstream 反向三位有效数字全同 |

三个第二期新发现的缺口（细节见 `docs/matrix/gaps.json`）：

1. **前向 kernel 不产出反向要的检查点**（`fwd-caches-not-emitted`）。九个里只有六个是
   kernel 直接给的；`g_cumsum` / `h` / `v_new` 当前在 host 侧补，实测占训练步 21%（bd=4）。
   它是 torch 算子，**不随核数缩短**，所以 block_dim 上限抬高后占比会继续涨。
2. **按 fla 的默认初始化，算子在门控跨度上失效 —— 前向已根治，反向是独立的第二处**
   （`gate-range-beyond-declared` 已 resolved / `bwd-gate-range-overflow` 新开 P1）。
   详见下面「门控跨度：两处、两个方向」。
3. **kernel 不做 q/k 的 l2norm、门控变换、beta sigmoid**（`qk-l2norm-not-in-kernel`）。
   fla 把这三步放在 kernel 里（`use_*_in_kernel=True`），我们要调用方做。数值上合法的输入
   无法区分做过没做过，所以**门控挡不住** —— 目前唯一"错了不报错"的语义缺口。

### 门控跨度：两处、两个方向

这是第二期最花时间的一件事，也是唯一一个"窄域测试全绿但真实模型不可用"的缺口。记在这里
是因为它的排查过程有两条可复用的教训。

**事实。** KDA 的 chunk 内门控跨度 = `max(cumsum(g)) - min(cumsum(g))`（每 64 token 重置，
**不随 T 增长**）。fla 自己的初始化（`A_log = log(U(1,16))` → `exp(A_log) ≈ 15.35`，
`dt_bias` 为 Mamba inv-softplus → `dt ∈ [0.001, 0.096]`）给出跨度约 **94**，而契约声明的
输入域（`g_raw ∈ [-0.03, 0]`）只到 1.89 —— 窄约 50 倍；反向单元 `make_inputs` 的分布宽些，
它五个 case 落在 1.30~46.52，也只到失效线的一半。ascriptor 的 kernel 在两处撑不住，
**方向相反**：

| | 位置 | 算式 | 失效线 | 机制 |
|---|---|---|---|---|
| 前向 | `gate.py` → `intra.py` / `wy.py` | `eg = exp(gc)`，下游取 `k/eg`、`eg_last/eg` | `-ln(FLT_MIN_NORMAL) ≈ 87.3` | **下溢**到 0（非正规数被 flush），于是 `0×inf`、`0/0` |
| 反向 | `finalize_pre.py` / `finalize_post.py` | `rscale = exp(g − g_last)` | `ln(MAX) ≈ 88.72` | **上溢**到 inf（输出是 bf16 GM），配对的 `cscale` 同时下溢到 0，矩阵乘得 `inf×0` |

**修法（两处同一个思路，都不改数学）。** 成对衰减 `exp(g_i − g_j)` 必须分解成两个单边因子
才能用 matmul 在通道维求和，而分解的锚点可以任选 —— 锚点常数在配对时抵消。上游两处都把
锚点取在区间端点（前向隐含取 0，反向取 `g_last`），于是一个因子顶到 `exp(±span)`。
改取**中点**，两个因子各压到 `exp(±span/2)`，有限性的理论上限正好翻倍：前向
`2 × 87.3 ≈ 174.7`，反向 `2 × 88.72 ≈ 177.4`。

**但两条链的闸不能取同一个数，这是实测出来的**：前向的约束是有限性（到 155.97 精度完全不
退化），反向的约束是**精度，而且它先于有限性到来**（梯度到 169.8 都有限，但 `dq` 在 130 处
就越过契约预算 0.05）。所以 `MAX_GATE_SPAN` 是二维的 —— `{impl: {forward, backward}}`，
`stable` 下前向 155、反向 105，纯推理走前者、训练走后者。
让反向"有限但超预算"地跑过去就是静默降级（AGENTS.md §7），所以宁可报错。

**反向的 105 是两头夹出来的，而下界这头我一开始算错了。** 默认初始化的跨度**不是常量 94，
是随机变量** —— 它 ∝ `max_hv exp(A_log)`，而 fla 取 `A_log = log(U(1,16))`。实测 12 个 seed：
HV=1 给 21.5~96.2、HV=2 给 15.3~94.8、**HV=8 给 63.1~100.9**（头数越多越稳地顶到上界）。
上界是 `exp(A_log)≤16 × dt≤0.1 × 63 ≈ 100.8`。**闸必须覆盖 100.8**，否则默认初始化的层会被
我们自己的门控拒掉；上限那头是预算还成立的最深实测点（110 处 `dq` 只剩 0.2% 余量，不取）。
105 实测 `dq`=4.861e-02，余量 2.8%。

AGENTS.md §3 规定不改 ascriptor 仓，所以两处都在本仓建了派生单元：
`kernels/projects/a5/kda_fwd_stable`（3 个 kernel）与 `kda_bwd_stable`（2 个 kernel）。
`impl="stable"` 是默认值，**同时选前向与反向两条链**，不提供分开的开关 —— 混用会让门控
检查的上限对不上实际会失效的那一侧。

**反向实测**（Ascend950PR / CANN 9.1.0 / `block_dim=1`，B1/T64/H1/HV1，数六个梯度里非有限元素的个数）：

| 跨度 | 1.12 | 74.83 | 89.35 | 125.09 | 149.66 | 159.71 | 169.76 | 174.23 |
|---|---|---|---|---|---|---|---|---|
| upstream | 全有限 | 全有限 | **六项全坏** | 全坏 | 全坏 | — | — | — |
| stable | 全有限 | 全有限 | 全有限 | 全有限 | 全有限 | 全有限 | 全有限 | dq/dk/dbeta/dg 各 64 个坏 |

upstream 在 89.35 一步跨过去就是 `dh0` 16384/16384、`dk`·`dv`·`dg` 各 8192/8192 —— 与
`ln(MAX) = 88.72` 吻合。stable 到 **169.76** 仍全部有限，**174.23** 才开始坏，而且只坏 64 个
元素（`gate_span` 报的是**逐通道最大值**，边缘上只有最深的那个通道越界），与理论上限 177.4
吻合。声明的 160 落在实测通过的 159.71 与首次失败的 174.23 之间。

重叠域里 stable 与 upstream 的逐梯度相对 L2 是 **6e-04 ~ 7e-03**（随跨度缓升），而 `dv` 与
`dh0` 逐位相同 —— 它们不经过被重标的那条路。**不是 0 也不该是 0**：改锚点在实数上是恒等变形，
但中间量落在 bf16 GM 上，换锚点就换了舍入发生的位置。量级与已量化的 fwd/bwd dtype 降精度
（2.2e-03~2.4e-03）同档，远在契约预算（`dk` 0.15 / `dg` 0.25）内。

**精度（契约五个 case，全过）**：对单元 oracle 的相对 L2 与 upstream 反向**三位有效数字全同，
只第四位有差**，最大的差是 `gentle_decay` 的 `dbeta`（3.928e-03 对 3.854e-03，1.9%）。
逐项数见 `docs/matrix/ops.json` 的 `a5.kda_bwd_stable`。

**宽域精度（随跨度的曲线）**。契约五个 case 的跨度只到 46.5，而真实层在 94，所以必须单独测。
宽域下**单元自己的 oracle 不能用** —— 它在跨度 93.84 时 `dq`/`dk`/`dg` 分别有 3286/6507/16320
个非有限值（按 `exp2(g_i − g_j)` 算整个成对矩阵再掩码，非因果半边先变 inf）。换成 fp32 逐
token 递推参考 + autograd（`tests/test_kda_bwd_deep_npu.py`）：

| 跨度 | 46 | 60 | 80 | **94** | 100 | 110 | 130 | 150 | 169 |
|---|---|---|---|---|---|---|---|---|---|
| `dq` /0.05 | 2.89e-2 | 3.87e-2 | 4.25e-2 | 4.55e-2 | 4.73e-2 | 4.99e-2 | 6.17e-2 ✗ | 6.69e-2 ✗ | 6.88e-2 ✗ |
| `dg` /0.25 | 6.33e-2 | 1.27e-1 | 1.76e-1 | 1.51e-1 | 1.44e-1 | 2.32e-1 | 2.37e-1 | 2.40e-1 | 6.49e-1 ✗ |

`dv` / `dbeta` / `dh0` 全程 2e-03~1e-02，`dk` 全程 ≤9e-02 —— **`dq` 是约束项**。
跨度 94 六项全在预算内，110 过但 `dq` 只剩 0.2% 余量，130 起超。闸因此取 105。

> **一个被证伪的假设，留着免得别人重走。** 我曾以为 `dq` 的误差主要来自反向 ABI 把
> `g_cumsum` 定为 bf16（前向入口的 g 是 fp32），即"任何实现都只能这么准"，于是想把判据换成
> "喂了 bf16 往返 g 的 fp32 参考"。**实测证伪**：算子离 fp32 参考反而更近（跨度 46：对 fp32
> 的 `dq` 2.889e-02，对那个参考 4.169e-02）。原因是那个构造把 bf16 cumsum **差分回**
> per-token 增量，属于灾难性相消；而 kernel 是**直接用** cumsum 算 `exp(g_i − g_j)`，
> 从不差分回去。这段反过来解释了**为什么 kernel 必须直接用 cumsum**。

**前向实测**（`o` 对逐 token 递推 oracle 的相对 L2）：

| 跨度 | 1.11 | 11.14 | 44.56 | 66.84 | 89.12 | 111.40 | 133.69 | 155.97 |
|---|---|---|---|---|---|---|---|---|
| upstream | 2.956e-03 | 2.939e-03 | 2.975e-03 | 2.925e-03 | **NaN** | **NaN** | **NaN** | **NaN** |
| stable | 2.956e-03 | 2.939e-03 | 2.975e-03 | 2.925e-03 | 2.924e-03 | 3.191e-03 | 3.053e-03 | 2.850e-03 |

重叠域内四位有效数字相同，宽域里不退化。层级：默认初始化下整层前向相对 L2 **4.697e-03**。
反向五个契约 case 在 `impl="stable"` 默认生效后重跑，六项梯度与之前逐位相同。

静态侧另有：`ascriptor check` 两个 kernel 各 0 error / 0 warning，IR op 数 247→251 与
368→372（多出的乘法在 64 行循环外，每 chunk 只算一次）。

**最后那一项（整层反向）已经补齐**，在有 `ascend950` 算子包的 CANN 9.2.0 机器上跑的
（层里的投影/卷积/softplus/RMSNorm 全是 torch_npu 算子，缺算子包的机器做不了这项）。
门控跨度确定性校准到 94 后，对同一份权重的 CPU fp32 层（KDA 算子换成逐 token 递推）逐参数比对：

| 量 | output | dx | A_log | dt_bias | f_proj.0 | f_proj.1 | 其余 14 项 |
|---|---|---|---|---|---|---|---|
| 相对 L2（预算 0.25） | 4.694e-03 | 9.024e-03 | 1.551e-01 | 6.542e-02 | 5.040e-02 | 5.153e-02 | ≤1.090e-02 |

`A_log` 与 `dt_bias` 误差大一个量级是预期的：它们的梯度都要穿过 `exp`/`softplus`，
深衰减下对 `g` 的扰动放大最厉害 —— 正是算子级曲线里 `dg` 预算放宽到 0.25 的同一个原因。
同批次的算子级宽域三档：跨度 46 → `dq` 2.889e-02、94 → 4.550e-02、104 → 4.782e-02。
`tests/test_kda_layer_npu.py` 8 项 + `tests/test_kda_bwd_deep_npu.py` 4 项，**12 passed**。

**六条教训。**

1. **窄域全绿不代表算子可用。** 前向五个 case、反向五个 case 全部 passed，跨度却分别只到
   1.89 与 46.52（后者约是失效线 88.72 的一半 —— 通过，但离悬崖只有两倍）。
   真实初始化最深能到 100.8。测试覆盖的是契约声明的域，而契约声明的域可能根本不是模型会用的域 ——
   要主动去问"真实模型在这个参数上取什么值"。
2. **第一次归因错了，而且错在方向上。** 我最初把前向的失效写成 `ln(FLT_MAX)` 的**上溢**，
   逐 kernel 二分后才发现是 `-ln(FLT_MIN_NORMAL)` 的**下溢**（失效时 `eg` 最小值恰为 0.0，
   输出全 NaN、零个 Inf —— 上溢会先给 Inf）。方向决定修法，所以这条纠正留在
   `gaps.json` 里没有删。
3. **跨度是随机变量，不是常数。** 我把"默认初始化 ≈ 94"当成了确定值，于是
   整层反向那个测试自己的前置断言先挂了（断言跨度 >80，实测那个 seed 只有
   55.2）。跨度 ∝ `max_hv exp(A_log)` 而 `A_log = log(U(1,16))`，同一个 seed 换一下 RNG 的消耗
   顺序就从 64.55 变 94.0；HV=8 实测 8 个 seed 落在 54.2~100.6，上界 `16 × 0.1 × 63 ≈ 100.8`。
   两个后果：测试要**确定性地标定**跨度（`_calibrate_span` 平移 `A_log`）而不是指望默认初始化；
   **闸的下界由这个上界定** —— 反向从 100 改成 105，否则默认初始化的层会被我们自己的门控拒掉。
4. **校验要排在有副作用的步骤之前。** `chunk_kda_fwd_with_caches` 原本先编译反向链、后查门控，
   于是真机上报的是 `AclError: 已经执行过 aclnn 算子`（编译注册了新 vendor 树，撞上
   `opp-path-read-once`），把本该报的"门控跨度超限"整个盖掉 —— 我照着那个错信息去查 opp 路径，
   方向全错。
5. **修一处不等于修完**。前向根治之后我差点就收工了 —— 反向那处是读源码时顺手发现的，
   不是测出来的，而契约的五个 case 永远测不到它（跨度最高 46.52，不到失效线 88.72）。
   **凡是按「量程」失效的缺陷，都要把整条链上同类算式逐处列出来查一遍**，
   而不是修掉报错的那一处。这次列了 14 个 kernel，命中 5 个（前向 3、反向 2）。
6. **"不吐 NaN"不等于"能用"。** 反向的有限性上限是 169.8，而精度在 130 就超出契约预算 ——
   若按有限性定闸（我最初就是那么写的 160），调用方会在 130~170 之间拿到**有限但超预算**的
   梯度且没有任何提示，这正是 AGENTS.md §7 要避免的静默降级。
   **扩了可用域之后，要把"有限"和"准"分别测，闸按更严的那个定。**

另外修了一个桥层 bug（`opp-path-read-once`）：CANN 只在首次算子解析时读
`ASCEND_CUSTOM_OPP_PATH`，之后注册的 vendor 树失效，而它报的是"算子包未安装"。

### 真实形状精度验收：同一个教训的形状版

门控跨度那件事讲的是"契约声明的域 ≠ 模型会用的域"，维度是**量程**。收尾时发现同一句话
在**形状**维度上也成立，而且更糟：此前本仓**所有**精度数字的最大 C 是 2、最大 H/HV 是 2，
而性能一直在真实形状上测（`kimi_linear_layer` B1/H32/HV32/C16/T1024）。
于是"算子精度在预算内"的依据里，没有一个点落在模型真会用的形状上。

按 `models.json` 的形状补测，第一档就抓到一个 **P0 静默错误**
（`c1-multihead-o-corrupt`）：

| block_dim | HV=2 | HV=4 | HV=8 | HV=16 |
|---|---|---|---|---|
| 1 | 仅头 1 对 | 仅头 3 | 仅头 7 | 仅头 15 |
| 2 | 全对 | 头 1,3 | 头 3,7 | 头 7,15 |
| 4 | 全对 | 全对 | 头 1,3,5,7 | 头 3,7,11,15 |

**C=1 且一个 cube 核要连续处理多个头时，`o` 的内容是错的** —— 正确的恰好是每个核分到的
最后一个头。`pair_begin/pair_end` 按 `GetCubeIdx()/GetCubeNum()` 切 `B*HV`，而
`GetCubeNum() == block_dim`，所以安全条件是 `B*HV ≤ block_dim`。C≥2 全对：chunk 循环跑
第二遍时补上了缺的那次同步（kernel 源码里写着 `auto sync is not used here because the
nested for loops interfere with it` —— 同步是手写的，手写的那份假设了 C≥2）。

**这是最坏的一类缺陷**：没有 NaN、没有报错、`|got| ≈ |ref|` 范数还正常，只有逐元素比值
是乱的。`final_state` 不受影响，**反向也不受影响**（它不消费 `o`）。已按 §7 在
`_check` 里报错拦掉，闸的边界照实测表逐个钉进测试 —— 这类缺陷一旦闸被改松，没有别的
东西会报警。

**为什么藏了两期**：`kda_fwd` 四个 case 里 C=1 的三个都是 HV=1，唯一 HV=2 的那个是 C=2。
在 (C, HV) 平面上只覆盖了 (1,1) 与 (2,2) 两个点，而 (1,2) 就是坏的 ——
**两个维度各自全绿，不等于它们的组合全绿。**

其余各项都干净（跨度固定 46，把差异归因到形状）：

| 项 | 结果 |
|---|---|
| chunk 链长 | C=2→64（T=128→4096）`o` 3.200e-03→3.277e-03、`final_state` 2.345e-03→2.456e-03 —— **链长 32 倍，误差 ×1.02，不累积** |
| 核切分 | kimi 形状 bd=1 vs bd=4 的 `o`/`final_state`/六项梯度**全部逐位相同（8/8）** —— 不需要参考的硬判据 |
| GQA 分组 | qwen 形状 H16/HV32（16 组）`o` 3.264e-03；逐 v 头 3.193e-03~3.335e-03，离散度 1.04 倍 |
| 头独立性 | 第 0/8/15 组单独按 H=1/HV=2 跑，与全量跑的切片**逐位相同（0.000e+00）** |
| 真实形状反向 | kimi 形状 dq 3.021e-02/0.05、dk 3.813e-02/0.15、dv 3.432e-03、dbeta 3.478e-03、dg 6.386e-02/0.25、dh0 2.347e-03 —— 全部在预算内，**与玩具形状同跨度下几乎同值** |

反向的参考按 chunk 做 gradient checkpointing —— 1024 步的完整图在 HV=32 下要十几 GB。
checkpoint 是**重算**不是近似，fp32 的精确性不变，峰值内存降到一个 chunk 的量级。

过程里差点走错一步：`o` 经过主机侧的 `_from_bhcld` 重排而 `final_state` 不经过，第一反应
是怀疑那段代码。**直接比裸 kernel 输出（BHCLD，未重排）错得一样**，才把锅定到 kernel。
"只有这个量经过那段代码"不是证据，先测再归因。

### decode 路径：算子通了，瓶颈不在算子

第三期的 decode 先探了一轮。ascriptor 侧整族没有 recurrent 单元，所以
`kernels/projects/a5/kda_fused_recurrent` 是**本仓自写的第一个 kernel**（前两个本仓单元是
上游的带标注改写）。`ascriptor check` 0 error / 0 warning / 156 ops。

设计上两件事值得记：

* **两趟扫 state**：第一趟衰减并攒 `kᵀ·state_dec`（给 delta），第二趟做 rank-1 更新并
  **顺手攒 `qᵀ·state_new`**。两趟都在 UB 内，GM 只碰一次 state。
* **一个头一个核**：按 `B*HV` 切给向量核，每头整份 state（64KB）常驻一核的 UB，
  **核间不需要任何同步**。这是刻意避开 `c1-multihead-o-corrupt` 那一类 —— 上游 chunk 的
  融合尾部把 K 切给 sub-block 才要手写同步，而手写那份假设了 C≥2。本 kernel 的 kernel 级
  循环只有一层，`auto_sync()` 能覆盖。

实测（CANN 9.2.0，`benchmarks/verify_decode.py`）：

| 项 | 结果 |
|---|---|
| 精度 | 八个形状 `o` 9.6e-08~1.8e-07、`final_state` 3.2e-08~1.4e-07（对 fp32 递推参考，预算 1e-05）—— 比 chunk 路径紧四个数量级，因为全程 fp32 |
| state 串接 | 逐 token 调 16 次并串接 state 与一次调 16 token **逐位相同**（两种形状、bd=1/4）。decode 的正确性就是这条 |
| 门控跨度 | **无上限** —— 逐 token 只用 `exp(g_i)`（~1.5），没有 chunk 那种 `exp(累计跨度)` 的量程问题 |
| block_dim | 1/2/4/8/16/28 全通（28 = 56 个向量核，物理上限，未死锁） |
| 延迟 | 整次 53~67µs，host 布局 26~29%，**设备侧边际只有 2.7~4.8 µs/token**，固定成本约 48~58µs |

**最重要的结论是个反直觉的**：我原本预测 decode 是带宽瓶颈（按 state 64KB×2/头估下限约
2.5µs），实测是**每次调用的固定成本主导**。`block_dim` 从 1 到 28 总时长没有趋势，正是这个
的征兆 —— 若不拆开量，就会把它误判成"扩展性不行"，与第一期在 `block_dim` 上连错两次
是同一个坑（§6 铁律一）。所以下一步是 `decode-call-overhead`（桥侧约 25µs + host 布局
15.7µs），不是调 kernel。

还没做：**没接进 layer**（`mode="fused_recurrent"` 仍报错 —— 还要短卷积的逐 token 状态推进
与 cache 寻址）、没做 harness 集成、T>16 要调用方自己分批（入口报错而不是自动分批：
自动分批会把一次调用的语义悄悄变成多次）。

## 6. 性能基线

两条基线，同形状、同 dtype、同步计时：

1. **torch_npu 组合实现** —— 用原生算子拼出同语义的 KDA。它同时是第二个 oracle。
   ⚠️ 当前在 A5/Ascend950PR 上**不可用**（`npu-builtin-ops-missing`）；需要换一台有
   完整算子包的机器，或改用下面第 3 条。
2. **ascriptor 生成的 aclnn 算子** —— 本仓的产物。
3. **备选**：内置算子缺失时，退化为「自编译 kernel 之间」的对比（不同 `block_dim`、
   不同 kernel 版本），并用 CPU fp32 参考守精度。它回答不了"比 torch_npu 快多少"，
   要如实标注。

报性能必须声明：形状、dtype、是否含 bwd、warmup 与重复次数、是否 `synchronize()`。
不得用 ascriptor 模拟器时间充当设备延迟。

## 7. 开放问题

- ~~**`scale` 参数无处安放**~~ **（KDA 已证不成立）**：`kda_sub2_score_kernel` 与
  `kda_sub45_fused_kernel` 都有 `scale: f32` 标量入口，本仓 `chunk_kda_fwd` 直通它，
  不需要 host 预乘。contract `inputs` 里没有 scale 是因为它是 attr 而非张量。
  GDN / DeltaNet 是否同样有入口待接线时核实。
- ~~**KDA 的 fwd/bwd dtype 不一致**~~ **（已量化）**：`kda_fwd` 的
  `beta`/`initial_state`/`g_raw` 是 FP32，`kda_bwd` 的同名张量是 BF16，autograd 组装时
  必须降精度。拆成三组对照测出来：保存值降精度 1.4e-03~1.6e-03、输出舍到 bf16 ~1.65e-03、
  叠加 2.2e-03~2.4e-03 且 ≈ `√(A²+B²)`（即两者独立）。与契约已声明的输出精度同量级，
  不构成阻塞；`autograd.py` 里那一步 `.bfloat16()` 是**显式**的，不当无害的类型适配。
- **A2 何时启动**：取决于 ascriptor 侧 A2/A3 deferred 状态何时解除。
- **varlen 是否要做**：训练场景常用 packing；ascriptor 侧明确声明不支持 `cu_seqlens`。
  代价与收益待评估。

完整缺口清单见 `docs/matrix/gaps.json`。
