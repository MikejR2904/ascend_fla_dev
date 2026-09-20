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

本节随新证据更新。当前未声称 BF-03 设备执行、模拟、管线模拟或性能通过。
真机先执行 B1/T4096/H=HV8 的完整 workload，再完成全部网格、17 个数组的
独立阶段与组合检查、NaN 预填、输入字节检查、FP32 改前改后与跨 bd 字节比较。
两种 dtype 均需真实 NPU host 算子审计。

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
