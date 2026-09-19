# FMT-02 KDA：token-major ↔ head-major 重排与 dtype 转换进自编译 kernel（A5 真机）

- 波次 / SoC：W0 / a5 · 优先级：P0（用户 D-PM-37）· 时限：24h · 依赖：无（`FMT-01` 的审计工具落地后统一用它；没落地前自写审计）
- 需要：ascriptor、fla、a5 真机（D-PM-34）
- 写集：`kernels/projects/a5/kda_layout/**`、`ascend_fla/ops/kda/chunk.py`、`ascend_fla/ops/kda/chunk_bwd.py`、`ascend_fla/ops/kda/autograd.py`、`tests/test_kda_layout_kernel.py`、`docs/research/kda_layout_kernel.md`
- **先读 `docs/pm/bf16-kernel-side.md`**。

## 这条任务是怎么来的

用户 2026-09-19（D-PM-37）：「禁止在host转数据类型，格式。需要在kernel内完成。」KDA 是 Kimi-Linear 的主路径。公开 ABI 是 token-major，kernel 内部是 head-major / chunk-major，
现在的 `ascend_fla/ops/kda/chunk.py` 在 host 侧用 `permute().contiguous()` 重排（`_npu_supports_d2d_copy` 探测、`layout_device={auto,npu,cpu}`，缺内置算子包的机器还会**绕 CPU 重排再传回**），并有若干 `.to(dtype)` 转换。
这些都是 host 侧格式 / dtype 转换，要挪进自编译 kernel。

## 范围

- 新建自编译单元 `kernels/projects/a5/kda_layout/`（`unit.py` + `contract.json` + `run.py`，可独立运行）：把公开 token-major 张量搬成 kernel 要的布局、把输出搬回 token-major（前向 `o`、反向 `dq/dk/dv/dg/dbeta` 与 `dh0` 等公开输出），以及现存的 dtype 转换。
  首选融进现有 kernel 的读写寻址（若这样要改 `kda_fwd_stable` / `kda_bwd_stable` 的 kernel 源码，**先发 `RISK kernel-change-needed`**，属 kernel 批次，PM 不批）；否则做成独立的自编译布局 kernel（一次 launch），并在 DONE 里报多出的 launch 数与耗时。
- `chunk_kda` / `chunk_kda_fwd` / `chunk_kda_fwd_with_caches` / `chunk_kda_bwd` 的 host 侧不再有 `permute` / `transpose` + `contiguous`、`.to(dtype)`、CPU 旁路。
  `layout_device` 参数**保留但只接受 `"auto"` / `"npu"`（都表示 kernel 侧，弃用的空操作）**，`"cpu"` 显式报错并指向 D-PM-37——这样 `benchmarks/**` 里现有调用不必改。
- 允许的 host 操作只有：分配输出（用 `empty*`，让 kernel 写满；缺内置算子包的机器上 `torch.zeros` 不可用）、不拷贝的元数据操作、检查并显式报错、launch。
- **A2-44 的 raw flags 预处理不在本任务**（那是 `BF-07`，要 kernel 批次批准）：审计时 flags 为假的路径必须干净；flags 为真的路径标「BF-07 待改」。
- 不含：decode（`BF-06`）、性能优化。

## 验收

- [ ] 通则 §1：`chunk_kda` 前向 / 带缓存前向 / 反向（BF16 路径，flags 为假）的 host 算子审计干净，附算子列表；入口源码里不再有上面列的转换。
- [ ] **输出与改前逐位相同**（重排 / 转换是纯数据搬运）：全部契约 case、Kimi 真实形状、`(C, HV)` 网格的边界（`C ∈ {1,2,3}` 奇偶、`HV/H ∈ {1,2,4,8}`）、`bd=1..4`；前向 `o` / `final_state` 与九个检查点、反向六个梯度都要比。
  不一致就是转换有 bug 或数值被悄悄改了，发 `RISK`，不要自己解释掉。
- [ ] 通则 §3：真机、完整 workload 先跑、`bd` 间逐位相同、输入不被修改、NaN 预填、每个 `bd` 一个进程且所有 kernel 首次执行前编完；门控闸（前向 155 / 反向 105）不变。
- [ ] 通则 §5：同卡三轮 baseline / candidate / baseline，baseline 是改前路径（如实标注），报多出的 launch 数与耗时，不设速度门槛。
- [ ] 写集之外先发 `RISK write-set-expansion`；`git diff --stat` 对 `reserved_paths` 为空；没有改 `kda_fwd_stable` / `kda_bwd_stable` kernel 源码（除非先发 RISK 并获批）。

## 已知陷阱

- 「先 `.contiguous()`」在缺内置算子包的机器上做不到，这正是现在 `layout_device` 绕 CPU 的原因；kernel 侧搬运顺带去掉了这个对内置算子的依赖，DONE 里如果测了缺算子包的机器，可以写，但不是验收项（机器归 assignee 管，D-PM-28）。
- 一个算子名一个进程一份 build；所有 kernel 在首次执行前编完（`AGENTS.md` §5）；`GetVecNum() == 2 * block_dim`。
- 反向要九个前向检查点，检查点的布局只被我们自己的 kernel 消费，不需要转回 token-major——别多搬。
