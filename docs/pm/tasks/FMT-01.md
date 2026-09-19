# FMT-01 host 算子审计：可复用工具 + 全仓公共算子入口清单（A5 真机）

- 波次 / SoC：W0 / a5 · 优先级：P0（用户 D-PM-37）· 时限：16h · 依赖：无
- 需要：ascriptor、fla、a5 真机（D-PM-34）
- 写集：`benchmarks/host_op_audit.py`、`tests/test_host_op_audit.py`、`docs/research/host_op_audit.md`、`benchmarks/evidence/host_op_audit/**`
- **先读 `docs/pm/bf16-kernel-side.md`**（什么算 kernel 侧、允许 / 禁止清单、统一验收）。

## 这条任务是怎么来的

用户 2026-09-19（D-PM-37）：「禁止在host转数据类型，格式。需要在kernel内完成。」通则把「host 算子审计」定为硬判据，但目前每个任务各写各的；而仓里到底哪些公共入口还有 host 侧 dtype / 格式转换，
只有 PM 的静态 grep 粗清单（KDA chunk / autograd / decode 有 permute+contiguous 与 dtype 转换，GDN / PGDN / GDN-2 chunk 有 BF16 加宽与 GQA 复制，PKDA 看起来是干净的）。
本任务做**可复用的审计工具**并给出**真机清单**：之后各任务统一用这个工具，PM 也据此拆改正任务。

## 范围

- 工具 `benchmarks/host_op_audit.py`：把一个可调用对象（公共入口）在 NPU 张量上跑一遍，用 `TorchDispatchMode` 记录其间发出的全部 aten 算子（算子名、输入输出 dtype / 形状 / 是否连续、是否分配了新 storage），
  按通则**用显式分类表**（不用启发式猜）分为：允许（分配 `empty*` / `zeros*`、不拷贝的元数据 `view` 等）/ 禁止—dtype 转换 / 禁止—格式转换（拷贝、重排、复制、`cat` / `stack` / `pad`、格式 cast）/ 禁止—算术；
  并给出是**哪行 Python 源码**发出的（栈回溯到 `ascend_fla/ops/**` 的 文件:行）。输出 JSON 与人读表。分类器要能在纯 CPU 上用合成函数测（`tests/test_host_op_audit.py`，不需要 NPU）。
- 清单：对**每个公共算子入口 × 每条 dtype 路径**在真机上跑审计——KDA `chunk_kda`（前向 / 带缓存前向 / 反向，经 autograd）与 `fused_recurrent_kda`、GDN `chunk_gdn`、GDN 反向（GDA-03 合入后）、PGDN `chunk_pgdn`、PKDA `chunk_precond_kda`、GDN-2 `chunk_gdn2`（函数名以仓里实际为准）。
  每个入口给出：发出的禁止类算子、对应源码 文件:行、分类。**只读**：不改任何 `ascend_fla/ops/**` 与 kernel 源码。
- 仓主轨道的保留路径（`board.json` 的 `reserved_paths`）只跑、不改；结果照样列进清单，标「仓主轨道」。
- 交付 `docs/research/host_op_audit.md`：清单 + 建议的改正任务拆分（PM 据此建任务）。

## 验收

- [ ] 工具的 CPU 单测（分类表、合成函数的正反例）；DONE 里给命令与结果。
- [ ] 真机：清单覆盖上面每个入口 × 路径，附原始 dispatch 追踪（`benchmarks/evidence/host_op_audit/`，每个入口一份 JSON + 未过滤日志），带 SoC、CANN、算子包目录；PM 从原始追踪复算清单。
- [ ] 阳性对照：对已知违规（KDA 的 layout 与 dtype 转换、GDN / PGDN / GDN-2 的 BF16 加宽与 GQA 复制）工具必须查出；阴性对照：对已知干净的入口（PKDA FP32）工具必须报干净。
- [ ] 没有改任何入口 / kernel 源码；`git diff --stat` 对 `reserved_paths` 为空。

## 已知陷阱

- `TorchDispatchMode` 只看得到 aten 层；`torch_npu` 的自定义算子（`npu_*`）也会经过 dispatch，要单独列出；runtime 桥用 ctypes 直接调 aclnn 的那一步看不到 aten——那是我们自己的 kernel 调用，允许，清单里标「桥内 launch」。
- 「元数据 view」与「拷贝」要靠是否分配了新 storage 判断，别只看算子名（`contiguous` 在已连续时是 no-op）。
- 别为了过审计在被测入口里加东西；审计只观察，不改行为。
