# GDA-01 GDN（gated_delta_rule）chunk 前向：A5 真机开发 + 优化（非 GQA）

- 波次 / SoC：W0 / a5 —— **例外任务，不属于 A2→A3→A5 波次顺序**，见 `AGENTS.md` §2 的
  D-PM-20 记录
- 优先级：P2 · 时限：24h · 依赖：无
- 需要：A5 真机、ascriptor workspace（`AGENTS.md` §3 的 gitcode 来源）
- 写集：`kernels/projects/a5/gdn_chunk_fwd/**`、`ascend_fla/ops/gdn_chunk_fwd.py`、
  `tests/test_gdn_chunk_fwd.py`、`docs/research/gdn_chunk_fwd_gate_range.md`

## 这条任务是怎么来的、边界在哪

issue #70/#71 提议做 GDN（`gated_delta_rule`，**不是** GDN-2）的 chunk 前向开发与优化，
在已配置好的 A5 机器上跑真机。用户批准（D-PM-20），但这是一条**窄范围例外**，不是把
GDN 调回第一优先级——读 `AGENTS.md` §2 那条更正：

- **KDA 仍是首个目标算子族**，本任务不影响这个排序。
- **Qwen3-Next 仍卡在 `gdn-no-gqa` 等六项 ABI 缺口上**，本任务不解这些缺口——
  **显式跳过 GQA**（`HV > H` 的分组），不支持就报错，不是想办法支持它。
- 本任务只覆盖"非 GQA 的 GDN chunk 前向在 A5 上跑通 + 优化"这一件事，动机是有人已经配好
  A5 真机访问、愿意做，不是因为发现了新的目标模型需求。

## 现成资产：`a5.gdn_fwd`（只读起点，不是要复用的代码，是要对照的 ABI）

ascriptor 上游已有 `kernels/projects/a5/gdn_fwd`（`AGENTS.md` §3 的 gitcode 来源），
契约声明的定尺域：

- `B,H,C` 是运行时维度，**`L=64、D=128` 固定**，无 tail 路径。
- **无独立 value-head 维度**（`gdn-no-gqa`，只有 `B,H,C`，没有 `HV`）——这正是本任务
  "显式跳过 GQA"要对应的具体约束，contract 里写清楚"不支持 `HV≠H`"，不是不声明。
- `block_dim ∈ {1,2}`。
- `initial_state` 只支持全零——非零初始 state 不在这次范围内，同样要报错而不是静默按零处理。
- `board`（SSH 真机）stage 已经 `passed`，但 `AGENTS.md` §4 那条警告原样适用：
  **SSH board 路径通过不等于本仓要用的本地 aclnn/CCE 编译路径通过**——这条本身就是本任务
  第一期要证明的事，不能假设已经解决。

**本任务在本仓 `kernels/projects/a5/gdn_chunk_fwd/` 建自己的单元**（`AGENTS.md` §3：
不改 ascriptor 仓，要改/验证就在本仓建派生单元），把 `a5.gdn_fwd` 的契约当参照定尺，
不是直接照搬代码。

## 步骤

1. **先冻结 ABI，再动手**（issue 里提的要求，也是 AGENTS.md §7 的门控纪律）：形状、dtype、
   舍入边界、支持域，在 `docs/research/gdn_chunk_fwd_gate_range.md` 里写清楚，
   不支持的组合（GQA、非零 initial_state、非标准 layout、scale、tail）在入口显式报错，
   错误信息说清楚哪条不满足、实际值是多少。
2. **双 oracle**：一个是 fla 的 `gated_delta_rule` naive（CPU fp32，语义权威）；另一个是
   本仓自建的独立 CPU fp32 参考（不是照抄 fla 的实现，是按语义重新写一遍再互相对照——
   两个只要有一个抄错，交叉验证就失去意义）。`o`/`final_state` 相对 L2 与 max_abs_diff
   都要报。
3. **真机优先，sim 只做定点诊断**（AGENTS.md §6 / GD2-01 更正过的同一条纪律）：
   完整选定的 workload 先在 A5 真机上跑，缩小形状的 sim/pipesim 只用于具体故障的定位，
   不是拿来铺开测全部形状。
4. **profile 之前不要相信任何性能推断**（AGENTS.md §6 铁律一）：先测基线，再谈优化。
   报同卡同步三明治（baseline-candidate-baseline，warmup/repeat，代表性长度建议
   T=1024/4096），带完整 SoC/CANN/toolchain 版本。
5. **形状网格覆盖重复头/chunk 与合法 `block_dim` 组合**（AGENTS.md §6 "形状维度要逐格扫"，
   C=1 多头那次教训——不要只扫单轴）。
6. **失败要留证据，不混淆真机与历史结论**（AGENTS.md §6："结论不跨 SoC 继承"同一条道理，
   这里是"不跨机器继承"）：这台机器测的数字不能代表另一台，也不能把 `a5.gdn_fwd` 契约里
   记的旧证据当成本任务自己的验证。

## 验收

- [ ] `docs/research/gdn_chunk_fwd_gate_range.md`：ABI 定尺域（含显式跳过 GQA 的边界）、
      双 oracle 交叉验证结论、量程判定（`exp`/除法/规约类算式逐处过一遍 AGENTS.md §6 方法论）。
- [ ] 入口门控：不支持的 GQA、非零 initial_state、非标准 layout/scale/tail 组合显式报错，
      不静默降级。
- [ ] 双 oracle 数字：`o`/`final_state` 相对 L2 + max_abs_diff，对 fla naive 与本仓独立参考
      都要报。
- [ ] 真机优先：完整选定 workload 先在 A5 上跑通；sim/pipesim 仅用于诊断，明确标注用途。
- [ ] 性能：profile 基线在先，同卡三明治数字带 SoC/CANN/opp 版本，T=1024/4096 两个代表长度。
- [ ] 形状网格覆盖重复头/chunk × `block_dim` 的组合边界点。
- [ ] 不声称 A2/A3 支持——本任务全程 A5，结论不外推。
- [ ] SoC + CANN 版本 + opp 算子包随每个真机数字一起报（PROTOCOL §3.7）。

## 验证命令

```bash
PYTHONPATH=<ascriptor>/library python kernels/projects/a5/gdn_chunk_fwd/run.py reference
PYTHONPATH=<ascriptor>/library python kernels/projects/a5/gdn_chunk_fwd/run.py check --launcher board --board a5
python -m pytest tests/test_gdn_chunk_fwd.py -q
python tools/gen_matrix.py --check
python tools/pm_board.py --check
```

## 已知陷阱

- `run.py` 的位置参数是 `{reference,check,profile}`，`sim`/`pipesim`/`board` 是 `check` 的
  `--launcher` 取值，不是位置参数（GD2-01 踩过，见其 spec 更正记录）。
- **这不是把 GDN 调回第一优先级**——不要因为这条任务存在就去动 KDA 的排期，也不要顺手
  尝试解 `gdn-no-gqa`（那是第四期的事）。
- `a5.gdn_fwd` 的 `board` stage 是 SSH 往返，不是本仓要的本地 aclnn/CCE 常驻桥——
  两者不是一回事，参照 `AGENTS.md` §4。
- 不要把 `a5.gdn_fwd` 契约里已有的验证记录当成本任务自己的证据——本任务是独立单元，
  独立验证。
