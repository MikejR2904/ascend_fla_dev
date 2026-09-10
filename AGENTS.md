# AGENTS.md

本仓库是 **fla 系列线性注意力算子在昇腾 NPU 上的高效实现库**，后端用
[ascriptor](https://github.com/ddddwee1/ascriptor)（指令级 Python 编译器）。

## 1. 定位

- **是什么**：独立的昇腾算子库。实现 GDN / KDA / DeltaNet 等 fla 系列算子的
  forward + backward，以 torch 可调用算子对外暴露，目标是**比现有方案更快**。
- **不是什么**：不是 fla 的 fork，也**不是 fla 的后端插件**。我们不接 fla 的
  `@dispatch` / `BackendRegistry` 机制，不在 `fla.ops.*.backends` 下注册。
  公共 API 由本仓自己定义。
- **fla 在这里的两个角色**：① 算子语义的权威定义；② 各算子族的 `naive.py`
  充当 CPU fp32 oracle。仅测试期依赖，运行时不依赖。

> 架构纪律：有人提议"顺便注册进 fla 的 dispatch"时，这是在改变仓库定位，
> 需要显式决策，不要顺手做。往 fla 方向回摆的代价是跟随上游的长期维护成本。

## 2. 三条已定的决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 目标 SoC | **A5 / 950 优先** | ascriptor 0.1.0 只正式声明 A5 支持，`projects/a5/` 下 GDN/KDA/DeltaNet 的 fwd+bwd 均已真机 passed |
| 与 fla 关系 | **纯算子库** | 不接 fla 接口，换来 ABI 自由度：可直接用定尺布局，省掉 layout 转换开销 |
| 首期范围 | **fwd + bwd** | 面向训练；反向资产已有，不做等于浪费 |

A2/A3 在 ascriptor 侧于 2026-09-06 被 deferred，且有未解决的 split-K FP32 cube
数值缺陷。`platform.py` 按多 SoC 设计，但 A2 是后续目标，**不要在 A5 还没打通时
分叉去做 A2**。

## 3. 与 ascriptor workspace 的关系

ascriptor 是 `library/` + `kernels/` + `agent/` 三个同级 checkout 组成的 workspace。
本仓通过 `$ASCRIPTOR_WORKSPACE` 定位它（未设置时按同级目录 `../ascriptor` 查找）。
进入那边工作前先读它自己的 `AGENTS.md`，
并按 `agent/compatibility.json` 选定 library/kernels 修订。

- 我们**复用**它的算子单元（`kernels/projects/a5/{gdn,kda,delta_rule}_{fwd,bwd}`）
  和算法单元（`chunk_row_scan`、`matrix_normalization`、`gated_approximations`）。
- 我们**不修改** ascriptor 仓。需要改 kernel 时，在本仓 `kernels/` 下按它的
  unit 协议建自己的单元（`unit.py` 导出 `make_inputs/reference/execute` +
  `contract.json` + `run.py`），保持可独立运行。
- 本仓实际使用的 ascriptor 修订记录在 `docs/matrix/ops.json` 的 `ascriptor_pin`。

## 4. 本仓要自建的关键能力：runtime 桥

这是第一期唯一的真实技术风险，也是整个仓的技术护城河。

ascriptor 现在的执行模型全是"**落盘 + 独立进程**"：`aclnn` launcher 写二进制参数
文件后跑独立的 `test_aclnnop`；`board`/`pypto` 通过 SSH 推源码和输入到远端；返回值
是 `torch.frombuffer` 重建的 **CPU** tensor。**它是 kernel 开发/验证框架，不是可嵌入
的运行时算子库。**

`ascend_fla/runtime/` 要补的就是这一段：把 ascriptor 生成的 CANN 自定义算子编译成
常驻 `.so`，经 `torch.library` 在进程内调用，直吃 NPU device tensor、零拷贝。
地基是 ascriptor 的 `runtime/aclnn/template/`（op_host + op_kernel + CMake 工具链）
和 `build_custom_op()`。

> ⚠️ 已知风险：六个 a5 单元的 `compile` 与 `cannsim` stage 全是 `untested`
> ——**真机 passed 走的是 SSH board 路径，不是我们要用的本地 aclnn 编译路径**。
> 第一期要先证明这条路通，再谈算子接线。

## 5. 硬件与远程环境

**本机是 macOS，没有 NPU。所有真机验证都在远程 Ascend 机器上执行。**

主机清单、SSH 方式、CANN 路径、conda 环境写在 git-ignored 的 `machine_specs.md`
（本仓尚未建立时，参照 ascriptor `agent/machine_specs.md` 与 ascriptor 的
`boards.json`）。

> 绝不把主机名、IP、端口、账号、路径写进任何会被提交的文件 —— 包括本文件、
> 脚本、注释、commit message。需要引用时写"见 `machine_specs.md`"。

- 按目标 SoC 选机器。**不要把一台机器的 CANN/torch 版本或结论套用到另一台。**
- 共享机器上跑任务前先看 `npu-smi info`；有别人的活跃任务就等，**绝不 kill
  或修改他人进程**。
- 远程工作副本通常是 `rsync` 的普通拷贝，不是 git checkout —— 不要在远端 `git pull`。

## 6. 验证方法论

### 双 oracle

每个算子的精度判定都对**两个**独立参考：

1. **fla 的 `naive.py`**（纯 torch，CPU fp32）—— 语义权威。
2. **torch_npu 组合实现**（同形状在 NPU 上用原生算子拼出来）—— 同时是性能基线。

两个 oracle 之间的差异本身就是有用信息，不要只报一个。

### 判定纪律

- **算子正确性一律在 fp32 下判定。** bf16 的逐元素比对没有判别力，只适合做端到端
  输出质量检查，不要用它判断算子对错。
- **chunk 与 recurrent 两条路径在数学上等价，互为最好的 oracle。** prefill/decode
  一致性（一次前向 vs 逐 token 递推 + state 传递）同理。
- **报数字，不报 "OK"**：给 `max_abs_diff` / 相对误差 / 相对 L2 残差。
- **算子级精度指标不能外推到任务精度**，反之亦然。要声称任务级影响，就得跑任务级
  实验，并且拆出中间对照组（"替换实现"与"改精度"是两件事，混在一起测会把账记错）。
- **失败要留证据**：贴真实输出和报错，不要用"应该没问题"收尾。日志留在 `tmp/<task>/`。

### 不继承 A2 的结论

同级的 `fla_infer` 工作区有 A2/910B3 上的 GDN 精度与 HF32 实测结论。
**那些数字属于 A2，不要搬到 A5 当预期。** 方法论可以借，阈值和结论必须在 A5 上重测。

## 7. 门控：不满足就报错

现成算子有硬性定尺限制（`L=64`、`K=V=128`、零初始 state、无 varlen、GDN 无 GQA 分组）。

- 这些限制必须在 `platform.py` / 算子入口**显式声明并在不满足时报错**，
  错误信息要说清哪一条约束没满足、实际值是多少。
- **绝不静默降级到 torch 兜底**，也不要用"近似等价"的路径悄悄替换。
  隐藏缺失能力比缺失能力本身更糟 —— 它让支持矩阵说谎。

## 8. 支持矩阵是单一事实源

`docs/matrix/` 下的 json 是**唯一权威**，markdown 由 `tools/gen_matrix.py` 生成。

- `models.json` — 目标模型的真实形状（带来源 URL 与获取日期）
- `ops.json` — 算子 ABI、定尺约束、各 stage 验证状态
- `gaps.json` — 缺口表，每条带影响面与建议处置

规则：**不要手写 `docs/matrix/*.md`**（手写的矩阵必然腐烂）。每一格的状态都应能
指向一次真实的 `check` / `profile` 运行记录。状态值沿用 ascriptor 的词汇：
`passed` / `untested` / `gap` / `failed`。

## 9. 目录约定

```
ascend_fla/
├── ascend_fla/
│   ├── platform.py      # SoC/CANN 探测、能力门控（不满足即报错）
│   ├── runtime/         # ★ ascriptor → 常驻 torch 可调用算子
│   │   ├── compile.py   #   kernel → CANN custom op .so
│   │   ├── cache.py     #   按 (kernel, 形状签名, SoC, 版本) 缓存
│   │   ├── binding.py   #   torch.library 注册，device tensor 零拷贝
│   │   └── autograd.py  #   fwd/bwd → autograd.Function
│   ├── ops/             # 算子层（目录名对齐 fla.ops 便于对照）
│   ├── modules/         # 窄切片：causal_conv1d / RMSNorm / FusedRMSNormGated
│   ├── layers/          # 窄切片：GatedDeltaNet 等
│   ├── models/          # 注入式：不重写 modeling_*.py，只替换 layer
│   ├── compat/          # 可选：fla 风格签名 wrapper（布局转换）
│   └── reference/       # torch oracle
├── kernels/             # 本仓自有的 ascriptor 单元（unit 协议）
├── tests/  benchmarks/
├── docs/plan.md         # 构建规划
├── docs/matrix/         # ★ 支持矩阵（json 权威，md 生成）
├── tools/               # gen_matrix.py 等
└── tmp/                 # 构建产物、日志、profiling（git-ignored）
```

**窄切片原则**：fla 有 43 个 layers、41 个 models，**一个都不要照搬**。按算子倒推，
用到哪个做哪个。`models/` 用注入而非重写 —— 直接用 HF/fla 的模型定义，只把我们的
layer 换进去，这样规格自动跟上游对齐、零维护成本。

新增目录时同步更新这一节。

## 10. 提交卫生

- 不提交：`machine_specs.md`、`boards.json`、模型权重、数据集、构建产物、
  profiling trace、`tmp/` 下任何东西。规则见 `.gitignore`。
- 提交前 `git status` 确认没有大文件和机器信息混入。
- commit message 中英文皆可，但不要写入主机或账号信息。
