# GDA-03 GDN chunk 反向：GQA/GVA 分组，A5 真机验收

- 波次 / SoC：W0 / a5。**真机验收是必须项**（申领人在 #89 的验收里自己要求，PM 同意，口径同 PK-03 / D-PM-26）：
  DONE 之前必须在 A5 真机完成测试，通过后 PM 审查；合入需要用户明确授权
- 优先级：`-`（D-PM-22 排期任务，不靠优先级）· 时限：24h（真机部分若因机器/卡自身受阻，assignee 在 STATUS 里如实说明，
  PM 据此调整，不当作超时）· 依赖：`GDA-02`（已完成）
- 需要：A5 真机、ascriptor workspace（`AGENTS.md` §3 的 gitcode 来源，按 `agent/compatibility.json` 的 pin）、
  fla（pin `e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2`）
- 写集：`kernels/projects/a5/gdn_chunk_bwd/**`、`ascend_fla/ops/gdn_chunk_bwd.py`、
  `tests/test_gdn_chunk_bwd.py`、`docs/research/gdn_chunk_bwd_gate_range.md`

## 这条任务是怎么来的

用户给的 GDN/PGDN 排期（D-PM-22）：GDN 前向（GDA-01）→ GQA 分组（GDA-02）→ PGDN 前向（PK-03）→ **backward** → decode →
性能。前三步已完成并合入。本任务是 backward 阶段的第一个切片：**分组 GDN 的 chunk 反向**，覆盖范围与前向逐项一致。
它来自 #89（申领人提案，PM 分诊后采纳）；来源是外部需求，所以在用户批准前是 gated。PGDN 的反向（`PK-05`）依赖本任务，单独做。

`docs/matrix/ops.json` 里的上游 `a5.gdn_bwd`（真机 2026-09-05 通过）**不能直接接**：它没有独立 value-head 维度
（`gdn-no-gqa`）、初始 state 只能为零、不产出初始 state 的梯度，而我们的前向是分组的。本任务在本仓建派生单元，
不改 ascriptor 仓（`AGENTS.md` §3）。

## 范围（窄）：反向域与前向域逐项一致，其余在发射 kernel 之前显式报错（`AGENTS.md` §7）

| 项 | 支持 | 其余 |
|---|---|---|
| 前向输入域 | 与 `ascend_fla/ops/gdn_chunk_fwd.py` 现有门控逐项一致：token-major 连续 `q,k:[B,T,H,128]`、`v:[B,T,HV,128]`、`g,beta:[B,T,HV]`（FP32；`g≤0`、`beta∈[0,1]`、有限）、T 是 64 的倍数且 ≤4096、`HV%H==0`（连续分组）、`block_dim∈{1,2}`、q/k/v 同为 FP32 或 BF16、`scale=128**-0.5` | 报错，**不悄悄扩大前向域** |
| 上游梯度 | `do:[B,T,HV,128]`；若前向返回 `final_state`，还有 `dht:[B,HV,128,128]`（FP32）。**三种组合都要覆盖**：只给 `do`、只给 `dht`、两者都给 | 其它形状/dtype 报错 |
| 输出梯度 | `dq,dk:[B,T,H,128]`——**组内 HV/H 个 value head 对同一个 key head 的贡献必须求和**（分组归约）；`dv:[B,T,HV,128]`；`dg,dbeta:[B,T,HV]`（FP32） | — |
| 初始 state | 仅 None（零）。因此**没有 `dh0` 输出**（同前向，也同上游 `a5.gdn_bwd` 的 "d_initial_state absent"） | 传张量报错 |
| 归一化 | **无**：前向和 pin 住的 `naive_recurrent_gated_delta_rule` 都不对 q/k 归一化（naive 只有 `q = q * scale`），反向不含归一化链式项；`dq` 带 `scale` 因子 | — |
| 前向检查点 | assignee 冻结 ABI 时决定，**默认在 backward 单元内自含地重算前向所需检查点**，不改 `gdn_chunk_fwd/**`（不在写集内） | 若必须让前向额外输出检查点或接 autograd：先发 `RISK write-set-expansion`，PM 查完冲突再批，批之前别动 |
| 其它 | 仅反向；decode、varlen、CP、`transpose_state_layout`、`head_first` 不在本任务 | 显式报错 |

## 口径决定（PM 对申领人提问的回答，预算在实现之前定死）

1. **oracle**：**A** = 对 pin 住的 fla `naive_recurrent_gated_delta_rule` 做 CPU FP32 自动微分（语义权威）；
   **B** = 独立推导的解析反向递推/块公式，**不复用 A 的实现**（不 import A、不用 autograd）。两者都必须先在 toy 形状上过一次
   **FP64 数值微分（gradcheck）**，证明 A 与 B 自己是对的。B 自带负控制：零梯度、取反、放大 1.25 倍必须被预算拒绝。
2. **损失**：标量损失 `L = <do, o> + <dht, final_state>`；`do`、`dht` 用固定种子运行时生成、非全 1、非对称。
3. **精度预算（FP32 判定）**：`dq`、`dk`、`dv`、`dg`、`dbeta` **每一个**对 A 与 B 的相对 L2 ≤ 1e-4，**逐梯度报数字**，
   不只报最大。**先校准再实现**：冻结 ABI 时先在 CPU 上报 A 对 B 的一致性；若二者自己就 > 1e-5（说明 1e-4 不可判别），
   assignee 必须在写 kernel 之前报给 PM，由 PM 依数据调整预算；**事后不得为了通过而放宽**。BF16 只作质量/存储边界检查
   （≤ 5e-3，报数，不作对错判定），不放宽 FP32 预算。
4. **量程**（`AGENTS.md` §6）：反向引入的新指数项（如 `exp(g_last−g)`、`exp(−g)` 一类）要**逐处列表判量程**，包括判为
   "恒 ≤1、下溢即正确"的。前向门控域沿用 GDA-01/02；若反向需要比前向更严的跨度闸，实测定，闸是双边约束：上边界=预算仍
   成立的最深实测点，下边界=调用方真实会送的值，两边都要有实测依据。
5. **真机（必须）**：完整选定 workload 先跑——`B1/T4096/H=HV8/K=V128`（FP32 与 BF16）、`HV/H ∈ {1,2,4,8}` ×
   `C=T/64 ∈ {1,2,3}` 与 `T=4096` 上界、多 batch、`block_dim ∈ {1,2}`；输入运行时生成，检查**实际返回的梯度**，
   叶子/组合检查，输入不被修改；`bd=1` 与 `bd=2` 输出逐位相同；sim/pipesim 只做定点诊断，不替代真机。
6. **同卡测量**：三轮 baseline/candidate/baseline 三明治（T=1024、T=4096）；基线由 assignee 选并**如实标注身份**
   （不要把"方程成本基线"写成"与某实现对比"）；**不设速度门槛**，慢也如实写；原始样本保留。
7. **机器与卡归 assignee 自己管理**（D-PM-28），PM 不设关卡；PM 只看进度与代码/证据质量：每个真机数字带 SoC、CANN、
   内置算子包目录和原始日志行，读未过滤的原始日志。

## 步骤

1. **冻结 ABI（对着 pin 住的源码，不凭参数名猜）**：`chunk_gated_delta_rule_bwd` 的梯度与 cotangent、分组归约、`scale` 因子、
   检查点/重算边界、公共梯度 dtype；写进 `docs/research/gdn_chunk_bwd_gate_range.md` 开头。
2. **参考先行**：A、B 与 FP64 gradcheck，报 A↔B 一致性，**在写 kernel 之前**（见口径 3）。
3. **量程复核**：反向整条链的指数项逐处列表。
4. **建单元**：`kernels/projects/a5/gdn_chunk_bwd/`（`unit.py` / `contract.json` / `run.py`），复用 `gdn_chunk_fwd` 的分解思路
   （只读引用，不改）。
5. **验证网格**（`AGENTS.md` §6：组合逐格扫，含奇偶）：见口径 5。
6. **入口门控**：`ascend_fla/ops/gdn_chunk_bwd.py` 对范围外组合与形状/状态不匹配显式报错，测试覆盖每一行"其余"。
7. 交付：`task/GDA-03`，PR 标题 `[GDA-03] …`，正文 `Refs #<issue 号>`，然后发 DONE。

## 验收

- [ ] `gdn_chunk_bwd_gate_range.md`：ABI 冻结表、A↔B 一致性与 gradcheck、整链量程表。
- [ ] `ascriptor check` 0 error；`run.py reference` 全过；sim/pipesim 在缩小形状边界点上过。
- [ ] **真机**：完整 workload 先跑；`dq/dk/dv/dg/dbeta` 逐梯度 FP32 相对 L2 ≤ 1e-4（对 A、B 各一组）；三种 cotangent 组合都覆盖；
      `HV/H∈{1,2,4,8}`、奇偶 chunk、多 batch、`bd=1/2` 逐位相同；输入未被修改。每个数带形状、dtype、SoC/CANN/算子包与原始日志行。
- [ ] 同卡三轮三明治的原始样本与如实的基线标注。
- [ ] 入口拒绝范围外组合，测试覆盖每一行"其余"。
- [ ] `git diff --stat` 对 `reserved_paths` 为空，且没有改动写集之外的文件；没有改 `gdn_chunk_fwd/**`。
- [ ] **没有声称**：CUDA/Triton、真实 checkpoint、任何没跑过的真机结果。

## 已知陷阱

- **分组求和**：`dq/dk` 是组内所有 value head 的求和；漏了会在 `HV>H` 时静默出错，而 `HV==H` 全对。
- **`dq` 带 `scale`**：naive 有 `q = q * scale`。
- 上游 `a5.gdn_bwd` 的约束（无 value-head、无 `dh0`）不能照搬；本任务是派生单元。
- **别把 GDN 的结论套给 PGDN**：PGDN 的 naive 做归一化、有 ATK 与读/写 key 不对称（`docs/research/pkda_semantics.md` §7），
  它的反向是 `PK-05`。
- 一个算子名一个进程一份 build；所有 kernel 在首次执行前编完。
- GDN-2 保留路径（`board.json` 的 `reserved_paths`）一个字都不要动。
