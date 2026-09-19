# BF16 必须 kernel 侧（用户 2026-09-19，D-PM-35）

> 用户原话：「bf16计算必须是kernel侧的计算，不能再host侧完成。追加任务把这个改正。后续bf16优先」

BF 系列任务（`BF-01` …）和之后所有涉及 BF16 的任务都以本文为通则；任务规格只写各自特有的部分。

## 什么算「kernel 侧」

- BF16 的 `q/k/v`（反向还有 `do` 等）**以 BF16 张量直接进自编译 kernel**：dtype 转换、归一化 / 激活 / 门控变换等算术、累加精度的选择，全部在 kernel 里做；
  输出（`o`、反向梯度）由 kernel **直接写成 BF16**。cube 累加到 FP32、`state` / 门控前缀 / 检查点用 FP32，是 kernel 内部的精度选择，允许（KDA 的 BF16 kernel 就是这样）。
- host 侧（Python 包装、PyTorch 算子）**只允许零算术、零 dtype 转换的布局操作**：分配输出（`empty`）、`view` / `reshape`、`expand`、已连续时的 `contiguous`、取指针、launch。
  **禁止**：`x.float()` / `x.to(dtype)` / `.bfloat16()`、`F.normalize`、`softplus` / `exp` / `sigmoid`、任何 elementwise / reduce / matmul——包括「先加宽成 FP32 再进 FP32 kernel、输出再转回」。
- 约束的对象是 **BF16 路径**。FP32 路径不变，也不得因此改变（FP32 路径逐位不变是硬判据，改前改后各跑一遍字节比较）。
- 本仓现存的不合规项（首页表里标 `🔁 API 加宽`）：GDN 前向、PGDN 前向、GDN-2 chunk 前向、KDA decode 的 BF16 入口在 host 侧加宽成 FP32；PKDA 目前直接拒绝 BF16。
  改正任务：`BF-01`（GDN 前向）、`BF-02`（GDN 反向）、`BF-03`（PGDN 前向）、`BF-04`（PKDA 前向）、`BF-05`（GDN-2 chunk 前向）、`BF-06`（KDA decode）；
  `BF-07`（KDA 前向里融合 q/k l2norm、门控、beta sigmoid）要 kernel 批次批准，gated。已合入的历史 PR 不回滚，标为待改。

## 统一验收（每个 BF 任务都要）

1. **host 算子审计（硬判据）**：BF16 公共入口在真机上跑一遍，用 `torch.utils._python_dispatch.TorchDispatchMode`（或等价手段）记录入口内发出的全部 aten 算子；
   只允许出现分配 / view / expand / 无操作 contiguous 这类布局算子，出现任何算术或 dtype 转换算子即失败。算子列表放进证据。入口源码里不得再有 `.float()` / `.to(dtype)` 一类的 BF16 路径转换。
2. **正确性口径**：仍是 fp32 判定——对 pin 住的 fla naive（CPU FP32，喂**已按 BF16 舍入的输入**）与独立 CPU 参考各报一组相对 L2 / `max_abs`。
   先在 CPU 上用「BF16 输入舍入 + FP32 递推」的模拟器**校准误差地板 F**（FP32 参考输出与其 BF16 舍入之间的相对 L2，通常 2–4e-3），
   **预算在实现前写定、事后不得放宽**；实测超过预算，或超过 3F，先发 `RISK` 再动。
3. **真机**（用户 D-PM-34）：完整 workload 先跑；分组比 / 奇偶 C / 多 batch / 尾块等边界按各任务列的网格；`bd` 之间输出逐位相同（一个 `bd` 一个进程，所有 kernel 在首次执行前编完）；
   输入不被修改；输出用 NaN 预填以识别漏写。证据带 SoC、CANN、算子包目录与**未过滤的原始日志**，PM 从原始 JSON 复算。
4. **量程 / 门控重测**：BF16 改变了精度。凡有门控跨度、指数量程的算子（KDA / PKDA / GDN-2 的 gate 前缀差等），闸要**在 BF16 路径上重测**，双边都有实测依据（`AGENTS.md` §6），不得沿用 FP32 路径的数。
   BF16 的可用域比 FP32 窄，属于域变化，先发 `RISK` 并等 PM / 用户确认。
5. **性能**：同卡三轮 baseline / candidate / baseline 三明治，baseline 用同一算子现有的 FP32 路径（如实标注身份），不设速度门槛，报数字——BF16 的意义之一就是 cube 吞吐，要能看出来。
6. **派生单元**：需要新 kernel 就建**新的派生单元**（`kernels/projects/a5/<unit>_bf16/`，`unit.py` + `contract.json` + `run.py`，可独立运行，见 `AGENTS.md` §3），FP32 单元源码不动。
7. **失败留证据**：不删 case、不降阈值、不放宽闸。

## 常见陷阱

- 用「BF16 值的 FP32 张量」冒充 BF16 kernel：kernel 的 GM 输入必须真的是 BF16 dtype。
- 把 `q.float()` 换成 `q.to(torch.float32)` 或 `torch_npu` 的 cast 算子——一样是 host 侧转换。
- 只测形状不测边界：奇偶 C、`HV/H` 组合、跨度端点；BF16 下失效的边界与 FP32 不同。
- 输出用 `.to(bf16)` 收尾：kernel 必须直接写 BF16。
- 内部 kernel 形参名避开 C++ 关键字（如 `do`，GDA-03 首次真机构建栽在这上）。
