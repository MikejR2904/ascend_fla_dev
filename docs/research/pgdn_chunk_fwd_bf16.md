# PGDN 原生 BF16 前向

BF-03 将 BF16 q/k/v 直接交给六阶段 Vector kernel；归一化、ATK 与递推
保持 FP32，末端直接写 BF16 o，主状态和 ATK 状态均为 FP32。公共 ABI 与
PK-03 一致，不扩展初态、归一化开关、center、形状或训练支持。原 FP32 单元不改。
PM 已明确 Cube 不是本任务要求；不作 Cube 吞吐声称。

## 实现前固定的比较契约

A 是 FLA `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2` 的
`fla/ops/precond_gated_delta_rule/naive.py`，文件 SHA256 为
`3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8`。
B 是本单元独立 CPU block solve：ATK 使用因果加权和，主状态使用三角系统求解。
B 不导入 A、生产图、设备 kernel 或 simulator。两者均接收实际 BF16 输入值，
在 CPU FP32 中计算；近零输入在设置分量后才舍入到 BF16。

对每个 case、每个 oracle、每个输出分别计算
`F = ||BF16(reference).float() - reference||₂ / ||reference||₂`。
BF16 输入的 o 和 final_state 预算在实现前固定为 `min(1e-2, 3F)`；
final_state 的实际存储仍是 FP32。F=0 必须精确一致，零参考范数不加 epsilon。
final_A_state 与 FP32 公共输出的相对 L2 预算为 `1e-4`。
FP32 阶段保留原 `atol=2e-5, rtol=2e-4` 且相对 L2 ≤ `1e-4` 的双重检查。
BF16 o 还检查 `atol=2e-5, rtol=1e-2`；不能只用该 allclose 替代 3F。
报告同时保留 max_abs、dtype、shape、非有限值与输入不变检查。

固定用例为 PK-03 的 30 个 shape/seed（两种 dtype 都测）加上
norm=`1e-13,1e-12,2e-12` 的三个用例；每种 dtype 都覆盖 bd1/2。
规范 BF16 单元有 66 个 case；额外 FP32 回归也有 66 个。
原比例 `{1,2,4,8}` × chunk 数 `{1,2,3,64}`、B2/H3/HV12、
T4096/H8 与 H14、零与近零行、强/弱/零门控、beta/beta_atk 端点均保留。

校准入口为单元内 `python -m ref.calibrate --output <report.json>`。
`evidence/calibration-pre-kernel.json` 与 `.log` 保存实现前的 66 条 CPU
记录（33 个 shape × 两种 dtype）；设备运行仍须现场生成输入与独立参考，
按同一公式重新求 F，不能跨环境搬运随机输入或把这份校准当真机验收。

## 运算与存储边界

BF16 q/k 在 ATK kernel 内精确加宽后按 `x/max(norm(x),1e-12)` 归一化。
`k_read` 用于修正 `v-k_readᵀD`；预条件的 `k_write` 用于状态写入与因果 score。
归一化后的 q/k 不再舍入为 BF16。ATK 状态由每个 `(B,H)` worker 独占，
主状态由 `(B,HV)` worker 独占；连续的 `HV/H` value heads 共享一个 key head。

六阶段为 ATK → prepare → scores → WY → scan → output。
所有中间 GM 边保持 FP32，独立分配且完整写入；仅末端 o 使用 BF16 存储。
q/k 的 BF16 staging 加宽到原 FP32 tile，v 在 prepare 内同样处理，
后续保留原 FP32 运算次序。一个局部 slot 按顺序复用，保留 STORE→LOAD
屏障并核验 auto_sync 的 DMA/VF 依赖与末次读者退休；不添加流水重叠。

FP32 公共路径继续执行原六个 kernel。公共入口移除输入输出 host cast，
保留原只读数值校验并在审计中单列，校验同步开销计入完整公共调用测量。
首次 CANN 算子解析前编完两个 dtype 的全部 entry，每个 bd 用独立进程。

## 验证状态

本节随新证据更新。首次完整 workload 与 bd1/bd2 全网格已通过；性能验收仍在进行，
未执行模拟或管线模拟。
主机全套为 918 passed / 9 skipped / 5 warnings；其中原 PGDN 测试 104 项、
新增 BF16 测试 38 项均通过，规范 reference 阶段 66 项通过。
五条警告来自已有 NPU 测试在明确的 CPU 环境中遇到 `torch_npu` ImportError
时的 pytest 弃用提示，原日志保留；这些 skip 不代表设备通过。

原入口负对照为两个 BF16 marshalling case 失败、两个 FP32 case 通过；
除获批的单个测试函数外，其余旧测试顶层 AST 不变。
规范 reference 首次因 `reference_dtype` 元数据放错字段而拒绝；已按 runner
契约放入 comparison rule，保留失败记录，不改计算、输入、误差指标或预算。

六个 entry × bd1/2 的 lowering 与 CCE 源码生成通过，无 lint 或警告。
从实际分配地址核对的 UB 峰值分别为 ATK 135712、prepare 217088、scores 163840、
WY 180224、scan 229376、output 196608 字节；分配不重叠、32 字节对齐，
生成的 get/rls 成对。它们属于静态检查，不能代替硅片上的同步验收。

A5 SSH 重连后完成隔离源码同步；首次 B1/T4096/H=HV8/bd2 完整 workload
的 BF16/FP32 两条路径均已通过实际设备验收。两类 dtype 的生产 host 事件均为
1 次 `aten.zeros.default` 和 17 次 `aten.empty.memory_format`；原只读校验单列。
BF16 o 对 A/B 相对 L2 为 0.0016589061/0.0016589062，预算分别约 0.004976718；
主状态为 6.86e-7/8.06e-7，ATK 状态为 3.50e-7/4.13e-7。FP32 公共输出与 pristine
入口逐位相同。全部 17 个组合数组和独立阶段、NaN 预填、输入不变检查均通过。
279 次设备上下文采样中 253 次在两种驱动注册表观测到本作业，外来进程为零，
运行前后为空、结束健康；锁保持至上下文排空。原始日志、源码清单与环境回执见
`evidence/full-bd2-v1*`，CANN 9.2.0 的 compiler/opp timestamp 为 20260805_101249091。
历史设备结果不用于本次资格声明。
真机先执行 B1/T4096/H=HV8 的完整 workload，再完成 bd1/bd2 的全部网格：
132 条记录（BF16/FP32 各 66），每条 17 个数组的独立阶段与组合检查、
NaN 预填、输入字节不变与两种 dtype 的真实 NPU host 算子审计均通过。
66 个 case/dtype 跨 bd 的 1,122 对独立阶段数组字节哈希一致，公共输出另核对
198 对（与阶段输出重叠，不重复计算独立数组）；66 条 FP32 公共结果与原入口逐位相同。
首次完整运行与网格重复运行的全部输入/输出/阶段哈希也相同。

全网格结果汇总见 `evidence/native-summary.json`，原始记录见 `grid-bd1-v1*`
和 `grid-bd2-v1*`。各输出对两个 oracle 合计的最大值如下；每个最大值可能
来自不同 case，原始记录保留具体形状与逐项预算。

| dtype | 输出 | 最大相对 L2 | 最大绝对误差 |
| --- | --- | --- | --- |
| BF16 | o | 0.00168058746 | 4.41074e-5 |
| BF16 | final_state（FP32） | 3.22841e-6 | 1.60187e-7 |
| BF16 | final_A_state（FP32） | 6.01047e-7 | 1.33514e-5 |
| FP32 | o | 2.45481e-6 | 4.75557e-8 |
| FP32 | final_state | 3.20236e-6 | 1.67639e-7 |
| FP32 | final_A_state | 5.81354e-7 | 1.33514e-5 |

16 项零预算检查全部通过；非零预算项最大 error/budget 为 0.3333337621。
两种 dtype 的全部生产 host 事件均为 zeros1 + empty17。
原 `_validate` 单列：isfinite9、all9、scalar13、gt4、any4、view2、sum2、lt2、bitwise_or2。
它们是原有只读检查，完整公共调用的性能测量包含这些同步开销。

24 份实际构建（原 FP32 六阶段与新 BF16 六阶段，各 bd1/2）的完整日志和
2,056 个源码/编译产物文件的字节数与 SHA256 见 `evidence/builds/`。
432 条含 warning 的日志行均保留：CMake 未使用跨编译变量（实际选 native x86_64），
CANN 头文件计划于 2027-06 后移除的 pragma 提示，以及 Python 3.12 对外层模板
字符串中 `\\d` 的转义提示。逐份模板与选定 library 字节相同，嵌套生成的 raw regex
保留原意；未改库、厂商文件、编译选项或生成代码。没有未解释的 kernel/同步警告。

同卡测量使用 pristine main 的完整 FP32 公共入口作为 baseline，
BF16 输入舍入值在测量外生成。T1024/4096、H8、bd2 各做三轮
FP32/BF16/FP32，每段 warmup10/repeat50，每次调用前后同步；不设速度门槛。
未运行的 CUDA/Triton 或权重验证不作声称。

实现前校准已完成：66/66 个 CPU A/B 对照通过。最大相对 L2 分别为
`o` 1.89193411e-06, `final_state` 1.2260166e-06, `final_A_state` 3.19019258e-07。

| 输出 | BF16 非零 F 范围（A/B 合计） | F=0 条数 |
| --- | --- | --- |
| o | 0.00163924664–0.00168026196 | 4 |
| final_state | 0.00152873728–0.00186271304 | 4 |

CPU 输入量化的 20 个独立记录见 `evidence/input-quantization.json`：
norm=1e-13 舍入成 9.992007221626409e-14，归一化首分量为
0.09992007166147232；ATK 相对未舍入输入的变化约 0.001598。
norm=1e-12 舍入成 1.0018652574217413e-12，跨过 clamp 点，归一化结果仍为 1。
这比较的是不同输入值，不是 kernel 对 oracle 的误差，不能拿来放宽 ATK 的 1e-4。
原生 harness 将近零 case 的实际 BF16/FP32 路径差与两份参考的输入量化差分列；
bd2 的原生比较已执行：norm=1e-13 时，实际 BF16 对未舍入 FP32 的
o/main/ATK 相对差分别为 0.00276725 / 0.00170575 / 0.00159777；A 的纯输入量化
贡献分别为 0.00224857 / 0.00170574 / 0.00159777，B 的结果同量级。
norm=1e-12、2e-12 时两种输入归一化结果相同，ATK 状态的实际跨 dtype 差为零。
完整逐项值保存在 `evidence/grid-bd2-v1.json` 的 `bf16_vs_unrounded_fp32`，
该差异列不参与每条路径相对于自身舍入输入参考的精度判断。

bd2 全网格 66/66 条记录通过，33 个 FP32 公共结果与旧入口逐位相同。
BF16 对 A/B 的最大 relL2：o=0.0016805783/0.0016805875，
主状态=2.21529e-6/3.22841e-6，ATK=6.01047e-7/4.78898e-7。
原始 JSON 逐 case/oracle 保留 F 与预算，不用这些最大值构造统一预算。
89 次设备上下文采样中 76 次在两注册表观察到本作业，外来进程为零；
原始证据与环境回执见 `evidence/grid-bd2-v1*`。
