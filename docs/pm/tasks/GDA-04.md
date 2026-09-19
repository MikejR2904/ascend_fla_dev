# GDA-04 GDN decode（fused recurrent）：A5 真机验收（骨架，细化在 PK-05 完成后由 PM 补）

- 波次 / SoC：W0 / a5。**真机验收是必须项**。合入需要用户明确授权
- 优先级：`-` · 时限：16h · 依赖：`PK-05`（按 D-PM-22 的"backward → decode → 性能"顺序；**PK-05 完成后 PM 才会补全本规格并放行**）
- 写集：`kernels/projects/a5/gdn_fused_recurrent/**`、`ascend_fla/ops/gdn_fused_recurrent.py`、
  `tests/test_gdn_fused_recurrent.py`、`docs/research/gdn_fused_recurrent_gate_range.md`

## 这条任务是怎么来的

D-PM-22 排期里 decode 阶段的第一个切片（GDN）。来自 #89（外部需求，分诊后采纳；用户批准前是 gated）。

## 范围骨架（PM 补全时要回答的问题，先列在这里）

- **短步数递推更新**：`q,k,v:[B,S,·,128]`（S 很小）、`g,beta`、**带非零初始 state 并输出更新后的 state**（`[B,HV,128,128]` FP32）。
  这是一个**新的域**——GDA-01/02 的前向只接受零初始 state；本任务必须显式声明并测试它，不能靠前向的结论。
- **prefill→decode 一致性**（`AGENTS.md` §6：chunk 与 recurrent 互为最好的 oracle）：`GDA-02` 的 chunk 前向产出的 `final_state`
  必须能被本算子当作初始 state 接着递推，并与一次性长前向逐 token 对得上；这是硬判据。
- 分组语义与 GDA-02 一致（连续 `HV/H` 个 value head 共享一个 q/k head）；`block_dim` 与核切分的取值由实测定（`AGENTS.md` §6：
  一个 bd 一个进程、`bd` 之间输出逐位相同）。
- 精度：FP32 ≤ 1e-4 判定，BF16 只作质量检查；oracle 同 GDA-03（A = pinned naive，B = 独立参考）。
- **测量**：单步调用的固定成本单独测（桥的缓存键必须是 O(1)，`AGENTS.md` §6 铁律三）；不设速度门槛，如实标注基线。
- 机器与卡归 assignee 自己管理（D-PM-28）。

## 验收骨架

- [ ] 域声明与 ABI 冻结（非零初始 state、S 的取值范围、state 布局与 `GDA-02` 的 `final_state` 一致）。
- [ ] 真机：完整选定 workload 先跑；prefill→decode 一致性；`bd` 间逐位相同；FP32 相对 L2 ≤ 1e-4。
- [ ] 入口拒绝范围外组合；`git diff --stat` 对 `reserved_paths` 为空。
