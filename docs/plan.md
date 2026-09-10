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
`triton_ascend`）。最终选择纯算子库，换来的是 **ABI 自由度**：

fla 用 `[B, T, H, K]`、state 为 FP32、带 `cu_seqlens` 变长；ascriptor 现成算子用
定尺 `[B, H, C, 64, 128]`、state 为 BF16、定长。接 fla 接口就必须在每次调用时做
layout 转换，而那恰好是访存开销 —— 与"高效率算子"的目标直接冲突。不接接口，
定尺布局就能原样用，转换降级为 `compat/` 里的可选便利层。

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

## 3. 第一期的真实风险：runtime 桥

ascriptor 的执行模型全是"落盘 + 独立进程"（详见 `AGENTS.md` §4）。它是 kernel
开发/验证框架，不是可嵌入的运行时算子库。本仓必须自建"常驻化 + torch 绑定 +
产物缓存"这一层。

**前置风险已量化**：六个 a5 单元的 `compile` 与 `cannsim` stage 全部 `untested`
——真机 `board` passed 走的是 SSH 远程路径，而我们要用的本地 aclnn 编译路径
从未验证过。第一期的第一个里程碑就是证明这条路通。

## 4. 全链路支持矩阵

目标是形成**模型 → 模块 → 层 → 算子**的支持矩阵，并与 fla 主仓的模型规格对齐。
三条做法约束：

**① 开发顺序自底向上，形状约束自顶向下。** "模型→算子"是看问题的视角，不是
施工顺序。算子先行，但算子要支持什么形状由模型 config 反推决定 —— 否则算子会在
玩具形状上全绿，到真机模型上挂掉。这就是第 0 期存在的理由。

**② 窄切片，不照搬。** GDN 这条链路实测只需要：

```
model:   Qwen3-Next 规格（gated_deltanet）
layer:   GatedDeltaNet                                              1 个
modules: ShortConvolution/causal_conv1d, RMSNorm, FusedRMSNormGated  3 个
ops:     chunk(训练/prefill) + fused_recurrent(decode)               2 个入口
```

43 个 layers / 41 个 models 一个都不要照搬。

**③ models 层用注入，不用重写。** 不复制 `modeling_*.py`。直接用 HF transformers /
fla 的模型定义，只把我们的 layer 替换进去 —— 规格自动跟上游对齐，零维护成本。

**④ 矩阵机器可读、CI 生成。** 学 ascriptor 的 `catalog.json` / `validation.json`
（带 contract 哈希与证据指向），不学 fla 的 README 表格。手写的矩阵三周后就没人信。

## 5. 分期

| 期 | 内容 | 验收 |
|---|---|---|
| **0** | 形状清单反推 + 缺口表 + 矩阵 schema | `docs/matrix/` 三份 json：目标模型真实形状 × 所需算子，缺口显式列出 |
| **1** | runtime 桥 + `gdn_fwd` | 真实形状下进程内零拷贝调用，精度对齐双 oracle |
| **2** | `gdn_bwd` + autograd + GatedDeltaNet layer（含 3 modules） | layer 级梯度端到端对齐；首版性能数 vs torch_npu 组合版 |
| **3** | model 注入 + `fused_recurrent`(decode) + 矩阵自动生成 | 端到端跑通一个模型；矩阵由 CI 产出 |
| **4** | KDA / DeltaNet 扩族 + 性能迭代 | 兑现"高效率算子" |

## 6. 性能基线

两条基线，同形状、同 dtype、同步计时：

1. **torch_npu 组合实现** —— 用原生算子拼出同语义的 GDN。它同时是第二个 oracle。
2. **ascriptor 生成的 aclnn 算子** —— 本仓的产物。

报性能必须声明：形状、dtype、是否含 bwd、warmup 与重复次数、是否 `synchronize()`。
不得用 ascriptor 模拟器时间充当设备延迟。

## 7. 开放问题

- **`scale` 参数无处安放**：fla 的 `scale`（默认 `head_dim**-0.5`）在 ascriptor GDN
  ABI 里没有对应入口（contract 里的 `scale: 0.05` 是输入生成幅度，不是算子参数）。
  要么进 kernel，要么 host 侧预乘 q —— 后者多一次 elementwise 遍历，与性能目标冲突。
- **A2 何时启动**：取决于 ascriptor 侧 A2/A3 deferred 状态何时解除。
- **varlen 是否要做**：训练场景常用 packing；ascriptor 侧完全没有 `cu_seqlens` 概念。
  代价与收益待评估。

完整缺口清单见 `docs/matrix/gaps.json`。
