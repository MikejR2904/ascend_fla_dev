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

## 2. 四条已定的决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 目标 SoC | **A5 / 950 优先** | ascriptor 0.1.0 只正式声明 A5 支持，`projects/a5/` 下 GDN/KDA/DeltaNet 的 fwd+bwd 均已真机 passed |
| 与 fla 关系 | **纯算子库** | 定尺约束远窄于 fla 公共 API 的承诺范围；做独立库才能把约束写进契约，而不是塞进 verifier 的拒绝理由 |
| 首期范围 | **fwd + bwd** | 面向训练；反向资产已有，不做等于浪费 |
| 首个算子族 | **KDA**（非 GDN） | 第 0 期 ABI 对比的结论，见下 |

**首个目标是 KDA 链路，不是 GDN。** 第 0 期把两者 ABI 逐项对出来后发现，KDA 在六项
能力上都更接近 fla 语义：GQA 分组、token-major 公开布局、非零 `initial_state`、
backward 产出 `dh0`、`final_state` 为 FP32、`block_dim` 上限 4。GDN 只在"本地验证证据
齐全"一项上占优，而那是可补的。对照表见 `docs/matrix/README.md` 的"为什么首个目标是
KDA"，依据见 `gaps.json` 的 `summary.kda_vs_gdn`。

首要目标模型相应是 **Kimi-Linear-48B-A3B**；Qwen3-Next 受 `gdn-no-gqa` 阻塞，随 GDN
扩族移到第四期。**不要因为 GDN 更知名就调回去** —— 换回来要先解决六项 ABI 缺口。

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
- **装包用自己的 venv**。共享 conda 环境可能同时服务别的项目，往里 pip install
  会污染别人。做法：`python -m venv --system-site-packages <工作区>/venv`，
  复用宿主的 torch/torch_npu，自己的包只进 venv。
- 传大文件要留意带宽：整包 `git archive` 往往有大量 examples/docs，
  只打包 `ascriptor` 包 + `pyproject.toml` 能把 29MB 压到 2.7MB。
  `scp` 中断会留下**不完整**的文件且不报错 —— 传完一定对 `md5sum`。

### 开机必查：opp 有没有 `ascend950` 算子包

**同为 Ascend950PR，不同机器的内置算子包覆盖不同 —— 这决定了你能做什么。**
进任何 A5 机器先跑：

```bash
ls $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/
```

- 有 `ascend950` → torch_npu 的计算算子可用（实测 randn / zeros / fp32+bf16 matmul /
  cast / contiguous / einsum / cumsum 全通）。性能基线与 layer 级验证都能做。
- 只有 `ascend910*` → **torch_npu 的计算算子全不可用**，下表为实测可用面。

实测：CANN 9.2.0（innerversion V100R001C25B046）的机器**有** `ascend950`；
CANN 9.1.0 的机器**只有** 910 系列。这是算子包安装差异，不是 SoC 级缺陷。

### 内置算子包缺失时的可用面（实测）

| 操作 | 可用 | 说明 |
|---|---|---|
| `torch.empty(device="npu")` | ✅ | 纯分配，不走算子 |
| `.to("npu")` / `.cpu()` | ✅ | H2D / D2H memcpy |
| `data_ptr()` / `current_stream()` | ✅ | runtime 桥需要的就是这些 |
| `torch.zeros` / `randn` | ❌ | 需要 ZerosLike / StatelessNormal |
| 任何 dtype 转换（`.float()`、bf16↔fp32） | ❌ | 需要 Cast |
| `permute().contiguous()`（NPU 上） | ❌ | 需要 d2d copy |
| 任何 matmul / einsum | ❌ | |

**实践后果**（仅在缺 `ascend950` 的机器上）：取值、比较、layout 重排一律**先 D2H
再做**（`t.cpu().float()`，不是 `t.float().cpu()`）。造零张量在 CPU 上造再 H2D。
`ops/kda/chunk.py` 的 `layout_device="auto"` 会自动探测并绕路。

**我们自己编译的 kernel 在两种机器上都不受影响** —— 计算都在自编译算子里。
这正是 runtime 桥的价值：它让算子在内置算子包不全的机器上照样可用。

**aclnn 相关的硬事实**（写 runtime 代码时会用到）：

- `aclCreateTensor` / `aclDestroyTensor` 在 **`libnnopbase.so`**，
  `libascendcl.so` 里没有这个符号。
- aclnn 参数顺序 = `inputs… + scalars… + outputs… + &wsSize + &executor`。
- aclnn 接口层**放宽** attr 类型：整型 attr 一律 `int64_t`，浮点 attr 是
  **`double`**（不是 `float`）。按 `c_float` 传 4 字节会让被调方从 8 字节槽里读到
  垃圾值 —— 实测表现为 `scale` 近 0，于是**只有用到它的输出归零、别的输出照常正确**，
  极其隐蔽。判类型一律看生成的 `aclnn_*.h`，不要照搬 ascriptor 的 `SCALAR_C`
  （那是 kernel 侧的 C 类型）。
- ACL dtype 枚举：f32=0、f16=1、i32=3、i64=9、bool=12、bf16=27；`ACL_FORMAT_ND=2`。
- aclnn 的 **HostSpec 标量列表包含 GM 形状里出现的全部符号维**，不只是 kernel 签名里
  显式声明的标量。`kda_bwd` 的九个 kernel 都因此多一个 `T`；`kda_fwd` 的五个恰好把符号
  都显式声明了，所以没踩到。**标量名一律以 `CompiledKernel.scalar_names` 为准，不要从
  kernel 签名推断** —— 漏传会报"缺少参数"（这个还算好查），多传或错序则不一定报错。
- 必须让 `ASCEND_CUSTOM_OPP_PATH` 指向 vendor 树，CANN 才找得到算子的 JSON 配置。
  **而且它只在首次算子解析时被读一次** —— 之后追加的路径 CANN 看不见，调用时报
  `rc=161001`，plog 里说的却是"SoC version ascend950 verification failed / 算子包未安装"。
  **这个报错是误导的**：构建产物完好也会这样。所以一个进程要用到的 kernel 必须在第一次
  执行之前全部编译完（`ascend_fla.ops.kda.prepare()`）。
- `ascriptor` 的 `a5` → `950` profile（32 cube / 64 vec），而 Ascend950PR 物理上
  只有 **28 cube / 56 vec**。`block_dim` 超过物理核数会在硬件 barrier 上死锁。

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

### 性能测量的三条铁律（都是踩出来的）

1. **profile 之前不要相信任何性能推断。** 我在 KDA 上对 `block_dim` 连续判断错两次：
   第一次结论"无效"是因为桥的缓存键每次调用都算一遍（`inspect.getsource` + sha256，
   5 个 kernel 共 ~10ms），把设备侧 0.23ms 淹没在 95% 的 host 开销里；第二次结论
   "无效"是因为同进程的四份 build 互相覆盖。两次都是先有推断、后看数据。
2. **一个算子名，一个进程，一份 build。** `ASCEND_CUSTOM_OPP_PATH` 是搜索路径，CANN
   按算子名查，第一个命中的 vendor 树胜出，解析每进程只发生一次 —— 第二份 build 被
   **静默**忽略。扫 `block_dim` 或标量绑定必须一份 build 一个进程。`runtime/binding.py`
   的 `_claim_op_name()` 会在越界时报错。
3. **缓存键不能比缓存贵。** 凡是放在每次前向热路径上的缓存查询，键的计算必须是 O(1)
   的字典查找级别。读源码、算 hash、遍历文件系统都不行。

`block_dim` 对 KDA 的实际效果（分进程实测，kimi_linear_layer）：三个重 kernel 从
bd=1 的 1.583/1.354/1.311ms 降到 bd=4 的 0.407/0.361/0.329ms，**近乎完美的 4 倍扩展**。
kernel 内部用 `GetVecIdx()/GetVecNum()` 自行切分，而 `GetVecNum() == 2 * block_dim`，
所以 bd=1 只用到 2 个向量核。契约只声明到 4。

### 卡会中途挂掉

179 那台的 NPU 7 在本次调试中途从 OK 变成 `Critical` / 0.0W，表现为 `TsdOpen failed,
devId=7` + `error code is 507033`。**先看 `npu-smi info` 的 Health 列再怀疑自己的代码。**
共享主机上不要尝试复位别人也在用的卡；换一张 Health=OK 且 `npu-smi info -t proc-mem -i N`
无进程的卡，并把换卡理由写进环境脚本的注释。

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
