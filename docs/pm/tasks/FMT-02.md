# FMT-02 KDA：token-major ↔ head-major 重排与 dtype 转换进自编译 kernel（A5 真机）

- 波次 / SoC：W0 / a5 · 优先级：P0（用户 D-PM-37）· 时限：24h · 依赖：无（`FMT-01` 的审计工具落地后统一用它；没落地前自写审计）
- 需要：ascriptor、fla、a5 真机（D-PM-34）
- 写集：`kernels/projects/a5/kda_layout/**`、`ascend_fla/ops/kda/chunk.py`、`ascend_fla/ops/kda/chunk_bwd.py`、`ascend_fla/ops/kda/autograd.py`、`tests/test_kda_layout_kernel.py`、`docs/research/kda_layout_kernel.md`
- **先读 `docs/pm/bf16-kernel-side.md`**。

## 这条任务是怎么来的

用户 2026-09-19（D-PM-37）：「禁止在host转数据类型，格式。需要在kernel内完成。」KDA 是 Kimi-Linear 的主路径。公开 ABI 是 token-major，kernel 内部是 head-major / chunk-major，
现在的 `ascend_fla/ops/kda/chunk.py` 在 host 侧用 `permute().contiguous()` 重排（`_npu_supports_d2d_copy` 探测、`layout_device={auto,npu,cpu}`，缺内置算子包的机器还会**绕 CPU 重排再传回**），并有若干 `.to(dtype)` 转换。
这些都是 host 侧格式 / dtype 转换，要挪进自编译 kernel。

## 2026-09-20 范围裁定（PM，依排队申领人在 #109 的只读规格核对——它发的 RISK 因任务未派、被工具按「assignee 不符」不采信，但内容成立，PM 当数据核实后采用）

本规格原来的验收写「带缓存前向 / 反向的 host 算子审计干净」，那是我静态盘点时只当成了纯搬运。申领人对源码逐段核对指出：`chunk.py::_scan_states`（约 599–639 行）用 host Torch 逐 chunk 做
FP32 bmm / 减 / 乘 / stack 来补 `h` / `v_new`（稳定 recurrent 本身不导出它们，即 `gaps.json` 的 `fwd-caches-not-emitted`）；`chunk.py` 约 720 行对 stable 的 `g_cumsum` 乘 `1/ln 2`，upstream 分支做 `log2(eg)`
再降 BF16；`chunk_bwd.py` 约 294 行用 host 取负生成 `dw`；域 / 门控跨度校验用只读 Tensor 算术。这些**不都是 layout / dtype**，只换布局与转换无法让完整带缓存前向的审计干净。裁定（D-PM-42）：

1. **FMT-02 的范围 = D-PM-37 的字面：host 的 dtype 转换与格式（布局）转换搬进自编译 kernel（新 `kda_layout` 派生单元）**，输出与改前**逐位相同**。这条不变。
2. **不属于 FMT-02、也不许顺手搬进新单元的「已登记存量算术例外」**——审计里单列、给出源码行、**不宣称合规**：
   - `_scan_states` **整体**（h / v_new 的 host 复算：bmm / 减 / 乘 / stack，以及夹在里面的 FP32 Cast）。里面的转换与算术缠在一起，不改前向 kernel 让它导出 `h` / `v_new` 就拆不开；
     消除它要改已有派生单元 `kda_fwd_stable` 的 kernel，属 **kernel 批次**，只有用户能批准（`fwd-caches-not-emitted`，P2，`requires_kernel_change`）。
   - upstream 分支的 `log2(eg)`（超越函数，kernel 里的实现不保证与 torch 逐位相同）。
   - 只读的域 / 门控跨度校验：审计里归「校验」类（暂定裁定，待用户确认），与数据预处理分开。
3. **允许（可选）**：只有当它是**位精确**的简单逐元素运算、能与某个 layout / dtype 搬运并进同一次 launch、且验收里输出与改前逐位相同时——`dw` 的取负（符号翻转）、`g_cumsum` 乘同一个 FP32 常数 `1/ln 2`——才可以并进新单元的 kernel；
   有任何一格不逐位相同就不并，退回「已登记例外」。**不许**为了并进去而引入 FMA / 不同的舍入路径。
4. **验收的审计口径相应改成**：BF16 路径、flags 为假——**推理前向（无缓存）与 decode**：host 算子审计干净（只有 empty / view / 校验 / launch）；**带缓存前向与反向**：**在已登记例外之外，不得再有任何 host dtype / 布局转换算子**，
   例外逐条列出（文件、行、算子名）并标注归属（`fwd-caches-not-emitted` 的 kernel 批次 / 校验类）。九个检查点与六个梯度改前改后逐位相同、门控闸（前向 155 / 反向 105）不变、raw flags 不扩大的要求都不变。
5. 这**不放宽**任何逐位判据 / 闸 / 域，也不给新代码开例外：新单元自己写的入口与 kernel 里不能有 host 转换 / 算术。存量算术留到 kernel 批次（用户批准）之后才能消除，PM 会把它列进要用户决定的事项。

## 2026-09-20 验收口径（PM 裁定 D-PM-44，依申领人的 RISK `bitwise-mismatch` / `kernel-change-needed` 与 STATUS）

> **06:05Z 补：PM 已从 PR #116 头 964cc8e 里提交的原始 JSON（`evidence/diagnosis/old-public-repeats.json`、`scan-probe-comparison.json`）自己复算**：旧路径 12 次 h0 五个哈希 7/2/1/1/1、其余 7 项各 1 个哈希、h0 相对 L2 与「超 0.05 的 4 次」与自述一致；第一设备 12 次直接重放 11 次 dh0 不同、dAqk / dh / dv 全同；候选 12 次同一哈希。
> 下面「事实」里的统计现在是 PM 核实过的；**设备 / 环境标识（哪张卡、CANN 9.1.0-beta.1）与第二设备的结果仍是自述**，PM 没有该环境、未复现。

**事实（自述）**：在申领人的替代环境（只读既有 CANN 9.1.0-beta.1）里，**旧公共路径自己**对同一 B2/T192/H2/HV4 输入重复 12 次，FP32 h0 梯度出现 5 个不同哈希（频次 2/1/7/1/1），其余 7 项输出 / 梯度各只有 1 个哈希；
h0 对 CPU FP32 的相对 L2 是 0.002337 ×7、0.008439、0.153270、0.357131、0.358643 ×2，**4 次超过原 0.05 预算**。差异隔离到只读上游 `scan_fused` 的 dh0 输出（`scan_fused.py:358-362` 一带，机制未定位）：固定输入直接重放已编译的原始 scan vendor 12 次，
第一张卡上 11 次 dh0 偏离第一次（1917–8184 个元素），另一张健康空闲的卡上 12 次全同；BF16 dh0 在任何 widen 之前就已不同，新 kernel widen 与旧 Torch widen 各自逐位等于 CPU FP32 转换；候选 12 次同哈希（等于旧路径的多数哈希）。

**裁定**（不放宽任何预算、不改判据的方向，只是让「逐位相同」在基线自己不稳定时仍可判）：

1. **转换边界（硬、逐位、每个样本）**：捕获真实运行里 scan 输出的 BF16 dh0 张量（含不确定的那些），旧 host widen、新 kernel widen、CPU FP32 转换三者对**同一张量**逐位相同。这是 FMT-02 自己改动的判据，不依赖设备是否稳定。
2. **端到端逐位对旧路径**：其余 7 项输出 / 梯度在**每个**跑过的设备上照旧要求逐位相同；dh0 / h0 梯度只在「旧路径自身重复 ≥12 次哈希全同」的设备上比，并写明是哪一类设备（用 dev-A / dev-B 这类匿名标签，不写设备编号 / 主机）。不稳定设备上 h0 的端到端结果**不作为通过或失败依据**，只作基线缺陷记录，不跨设备继承。
3. **预算不动**：h0 梯度原预算（对 CPU FP32 相对 L2 ≤ 0.05）在稳定设备上候选必须满足；不稳定设备上旧路径超预算的那几次是**基线缺陷**（已记 `gaps.json` 的 `kda-bwd-scan-dh0-nondeterministic`），不是候选的通过依据。
4. **候选在不稳定设备上 12 次同哈希不算「修好了」**：同步 / 捕获边界会改变复现概率，不许加 host 同步或改 kernel 去掩盖；scan 不动、写集不扩、上游只读不变。
5. **不承接根因与修复**：机制没定位（§6.5：不许照症状改），修在本仓派生 scan 单元、进 kernel 批次，**要用户批准**（PM 已记账并请示）。DONE 里交一个**最小复现包**（固定输入、直接重放脚本、两张卡各自的哈希分布、原始张量的哈希、CANN 版本 + compiler timestamp + opp 目录）供后续任务使用，不需要再往根因挖。
6. **环境标识**：DONE 的每份原始回执带 CANN 版本 + compiler timestamp 与 opp 算子包目录（是否有 `ascend950`）；本任务的环境（CANN 9.1.0-beta.1）与前面 A5 证据（9.2.0）不同，结论只对那一套成立。有条件的话，在原 9.2.0 环境上补跑同一组直接重放能帮着区分「环境 / 卡」与「kernel 竞态」，不强制（机器归你管，D-PM-28）。

## 范围

- 新建自编译单元 `kernels/projects/a5/kda_layout/`（`unit.py` + `contract.json` + `run.py`，可独立运行）：把公开 token-major 张量搬成 kernel 要的布局、把输出搬回 token-major（前向 `o`、反向 `dq/dk/dv/dg/dbeta` 与 `dh0` 等公开输出），以及现存的 dtype 转换。
  首选融进现有 kernel 的读写寻址（若这样要改 `kda_fwd_stable` / `kda_bwd_stable` 的 kernel 源码，**先发 `RISK kernel-change-needed`**，属 kernel 批次，PM 不批）；否则做成独立的自编译布局 kernel（一次 launch），并在 DONE 里报多出的 launch 数与耗时。
- `chunk_kda` / `chunk_kda_fwd` / `chunk_kda_fwd_with_caches` / `chunk_kda_bwd` 的 host 侧不再有 `permute` / `transpose` + `contiguous`、`.to(dtype)`、CPU 旁路。
  `layout_device` 参数**保留但只接受 `"auto"` / `"npu"`（都表示 kernel 侧，弃用的空操作）**，`"cpu"` 显式报错并指向 D-PM-37——这样 `benchmarks/**` 里现有调用不必改。
- 允许的 host 操作只有：分配输出（用 `empty*`，让 kernel 写满；缺内置算子包的机器上 `torch.zeros` 不可用）、不拷贝的元数据操作、检查并显式报错、launch。
- **A2-44 的 raw flags 预处理不在本任务**（那是 `BF-07`，要 kernel 批次批准）：审计时 flags 为假的路径必须干净；flags 为真的路径标「BF-07 待改」。
- 不含：decode（`BF-06`）、性能优化。

## 验收

- [ ] 通则 §1：`chunk_kda` 前向 / 带缓存前向 / 反向（BF16 路径，flags 为假）的 host 算子审计干净，附算子列表——**「干净」按 2026-09-20 范围裁定的口径：推理前向与 decode 完全干净；带缓存前向与反向在已登记的存量算术例外（`_scan_states`、`log2(eg)` 分支、校验类）之外不得再有 host dtype / 布局转换算子，例外逐条列出**；入口源码里不再有上面列的转换。
- [ ] **输出与改前逐位相同**（重排 / 转换是纯数据搬运）：全部契约 case、Kimi 真实形状、`(C, HV)` 网格的边界（`C ∈ {1,2,3}` 奇偶、`HV/H ∈ {1,2,4,8}`）、`bd=1..4`；前向 `o` / `final_state` 与九个检查点、反向六个梯度都要比。
  不一致就是转换有 bug 或数值被悄悄改了，发 `RISK`，不要自己解释掉。
- [ ] 通则 §3：真机、完整 workload 先跑、`bd` 间逐位相同、输入不被修改、NaN 预填、每个 `bd` 一个进程且所有 kernel 首次执行前编完；门控闸（前向 155 / 反向 105）不变。
- [ ] 通则 §5：同卡三轮 baseline / candidate / baseline，baseline 是改前路径（如实标注），报多出的 launch 数与耗时，不设速度门槛。
- [ ] 写集之外先发 `RISK write-set-expansion`；`git diff --stat` 对 `reserved_paths` 为空；没有改 `kda_fwd_stable` / `kda_bwd_stable` kernel 源码（除非先发 RISK 并获批）。

## 已知陷阱

- 「先 `.contiguous()`」在缺内置算子包的机器上做不到，这正是现在 `layout_device` 绕 CPU 的原因；kernel 侧搬运顺带去掉了这个对内置算子的依赖，DONE 里如果测了缺算子包的机器，可以写，但不是验收项（机器归 assignee 管，D-PM-28）。
- 一个算子名一个进程一份 build；所有 kernel 在首次执行前编完（`AGENTS.md` §5）；`GetVecNum() == 2 * block_dim`。
- 反向要九个前向检查点，检查点的布局只被我们自己的 kernel 消费，不需要转回 token-major——别多搬。
