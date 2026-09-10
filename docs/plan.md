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
| **3** | model 注入（Kimi-Linear）+ KDA `fused_recurrent`(decode) + 矩阵 CI 生成 | 端到端跑通一个模型；chunk↔recurrent 互验通过 |
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

三个第二期新发现的缺口（细节见 `docs/matrix/gaps.json`）：

1. **前向 kernel 不产出反向要的检查点**（`fwd-caches-not-emitted`）。九个里只有六个是
   kernel 直接给的；`g_cumsum` / `h` / `v_new` 当前在 host 侧补，实测占训练步 21%（bd=4）。
   它是 torch 算子，**不随核数缩短**，所以 block_dim 上限抬高后占比会继续涨。
2. **按 fla 的默认初始化，算子会 fp32 上溢**（`gate-range-beyond-declared`）。实测前向在
   chunk 内门控跨度 ≤67 时完全正常、≥89 时吐 NaN，分界是 `ln(FLT_MAX)≈88.7`；而 fla 的
   KDA 初始化给出约 94。契约声明的域（≤1.92）比失效阈值还窄 46 倍 —— 此前所有精度数都
   测在真实模型不会出现的窄域里。已加 `MAX_GATE_SPAN=80` 的显式门控，但那只是把静默失败
   变成明确失败，**没有扩大可用域**。第三期注入前必须就此做决策。
3. **kernel 不做 q/k 的 l2norm、门控变换、beta sigmoid**（`qk-l2norm-not-in-kernel`）。
   fla 把这三步放在 kernel 里（`use_*_in_kernel=True`），我们要调用方做。数值上合法的输入
   无法区分做过没做过，所以**门控挡不住** —— 目前唯一"错了不报错"的语义缺口。

另外修了一个桥层 bug（`opp-path-read-once`）：CANN 只在首次算子解析时读
`ASCEND_CUSTOM_OPP_PATH`，之后注册的 vendor 树失效，而它报的是"算子包未安装"。

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
- **KDA 的 fwd/bwd dtype 不一致**：`kda_fwd` 的 `beta`/`initial_state`/`g_raw` 是 FP32，
  `kda_bwd` 的同名张量是 BF16。autograd 组装时这一步降精度不在任何一侧的契约预算内，
  影响幅度待测（第二期前置）。
- **A2 何时启动**：取决于 ascriptor 侧 A2/A3 deferred 状态何时解除。
- **varlen 是否要做**：训练场景常用 packing；ascriptor 侧明确声明不支持 `cu_seqlens`。
  代价与收益待评估。

完整缺口清单见 `docs/matrix/gaps.json`。
