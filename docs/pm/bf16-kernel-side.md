# dtype 与格式转换必须 kernel 侧（BF16 优先）

> D-PM-35（用户 2026-09-19）：「bf16计算必须是kernel侧的计算，不能再host侧完成。追加任务把这个改正。后续bf16优先」
> D-PM-37（用户 2026-09-19）：「禁止在host转数据类型，格式。需要在kernel内完成。」

文件名沿用 `bf16-kernel-side.md`（引用太多，不改名）；**D-PM-37 把范围从 BF16 扩到所有 dtype 与格式**。BF 系列、FMT 系列任务和之后所有算子任务都以本文为通则，任务规格只写各自特有的部分。

## 什么算「kernel 侧」

- 公共算子入口里，**dtype 转换与格式转换一律在自编译 kernel 里完成**：kernel 直接吃调用方给的 dtype 与布局（token-major、连续），直接写调用方要的 dtype 与布局。
  BF16 的 `q/k/v`（反向还有 `do` 等）以 BF16 张量直接进 kernel；cube 累加到 FP32、`state` / 门控前缀 / 检查点用 FP32，是 kernel 内部的精度选择，允许（KDA 的 BF16 kernel 就是这样）。
- host 侧（Python 包装、PyTorch / torch_npu 算子）**只允许**：分配输出（`empty*`；缺省 state 的零填充 `zeros` 可以，能让 kernel 自己写零更好）、
  不搬数据的元数据操作（`view` / `reshape` / `unsqueeze` 等，前提是不触发拷贝）、检查形状 / dtype / 连续性（不满足就**显式报错**，不悄悄修）、取指针、launch。
  **PM 暂定，待用户确认（2026-09-19，BF-04 / BF-01 申领人的提问）**：① **只读数值校验**（有限值 / 范围 / 范数 / 门控跨度的判定，不产出进入计算的张量）允许留在 host，
  也可以放进自编译 kernel、由 kernel 写一个小状态字、host 回读该状态字后显式报错（「控制回读」，是控制元数据不是数据转换）；这两种在审计里单列为「校验」类，不算违规，但要列出并写明带来的同步开销；
  ② 带常数填充的分配（`zeros` / `full`）算分配，允许；③ 显式 CPU 诊断 launcher（sim / board / CPU aclnn）里的输入校验保留，不属于 NPU 生产路径的审计范围。
- **禁止**：
  - dtype 转换：`.float()` / `.to(dtype)` / `.bfloat16()` / `.half()` / cast 算子，包括「先加宽成 FP32 再进 FP32 kernel、输出再转回」；
  - 格式转换（搬数据的重排或拷贝）：`permute` / `transpose` 之后 `contiguous`、token-major ↔ head-major 重排、`repeat` / `repeat_interleave` / `expand` 之后再拷贝（如 GQA 的 q/k 复制）、`cat` / `stack` / `pad`、`clone`、
    NZ / ND 等 NPU 私有格式转换（`npu_format_cast`）、以及「先绕 CPU 重排再传回」（`layout_device='cpu'` 一类的旁路）；
  - **产出进入计算的数据的算术**：`F.normalize`、`softplus` / `exp` / `sigmoid`、任何 elementwise / reduce / matmul（预处理、归一化、激活、默认值的计算）；只读校验除外（见上）。
- 「在 kernel 里完成」= **自编译 kernel** 完成，不是 torch 算子：可以融进主 kernel（首选），也可以是一个独立的自编译布局 / 转换 kernel（一次 launch），但要在 DONE 里报多出的 launch 数与耗时。
- 约束的对象是**所有 dtype 的路径**，FP32 也一样。去掉 host 转换之后，输出要与去之前**逐位相同**（转换是纯数据搬运，不应改数值；有意改变精度的另说，先报 `RISK`）。
- 范围（PM 的读法，用户可纠正）：算子公共入口及其包装（`ascend_fla/ops/**`）。模型层 / 模块（`ascend_fla/layers`、`modules`：投影、卷积、norm 等本来就是 torch 算子组成）暂不在内，除非用户另说。
- 本仓现存的违规（静态清单，正式清单由 `FMT-01` 的真机审计给出）：KDA 前 / 反向与 decode 的 token-major ↔ head-major 重排与 dtype 转换（含 `layout_device='cpu'` 旁路）；
  GDN / PGDN / GDN-2 chunk 前向的 BF16 加宽与 GQA 复制；A2-44 的 raw flags 预处理（FP32 host 算子）。
  改正任务：`BF-01` … `BF-06`（dtype + 格式，两条 dtype 路径）、`FMT-01`（真机审计工具与清单）、`FMT-02`（KDA 的 layout / dtype 进 kernel）、`BF-07`（KDA 前向融合预处理，要 kernel 批次批准，gated）。已合入的历史 PR 不回滚，标为待改。

## 统一验收（每个 BF / FMT 任务都要）

1. **host 算子审计（硬判据）**：公共入口在真机上**每条 dtype 路径各跑一遍**，用 `torch.utils._python_dispatch.TorchDispatchMode`（或 `FMT-01` 交付的审计工具，它落地后统一用它）记录入口内发出的全部 aten 算子；
   只允许出现分配（`empty*`、`zeros*`）与不拷贝的元数据算子（`view`、`unsqueeze`、`alias` 等），出现任何 dtype 转换、拷贝 / 重排、产出计算数据的算术算子即失败；只读校验与状态字回读单列为「校验」类（不算失败，但要列出）。算子列表放进证据。入口源码里不得再有 `.float()` / `.to(dtype)` / `.permute()+.contiguous()` 一类的转换。
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
- 把 `q.float()` 换成 `q.to(torch.float32)` 或 `torch_npu` 的 cast 算子——一样是 host 侧转换；把 `permute().contiguous()` 换成 `torch_npu` 的格式 / 转置算子——一样是 host 侧格式转换。
- 只测形状不测边界：奇偶 C、`HV/H` 组合、跨度端点；BF16 下失效的边界与 FP32 不同。
- 输出用 `.to(bf16)` 收尾：kernel 必须直接写 BF16。
- 内部 kernel 形参名避开 C++ 关键字（如 `do`，GDA-03 首次真机构建栽在这上）。
