# GDA-02 GDN chunk 前向：GQA/GVA 分组支持（PGDN 的前置）

- 波次 / SoC：W0 / a5 —— GDN-on-A5 例外任务的既定延续（D-PM-20 之后，用户 2026-09-18
  直接批准的排期：GDN 前向 → GQA → PGDN 前向 → backward → decode → 性能，见 D-PM-22）
- 优先级：P1 · 时限：20h · 依赖：GDA-01（已完成）
- 需要：A5 真机、ascriptor workspace（`AGENTS.md` §3 的 gitcode 来源）
- 写集：`kernels/projects/a5/gdn_chunk_fwd/**`、`ascend_fla/ops/gdn_chunk_fwd.py`、
  `tests/test_gdn_chunk_fwd.py`、`docs/research/gdn_chunk_fwd_gate_range.md`

## 这条任务是怎么来的

用户直接决定了 GDN/PGDN 的排期顺序：**GDN 前向（已完成，GDA-01）→ 给 GDN 补 GQA 分组
（本任务）→ PGDN 前向（PK-03，之后解锁）→ backward → decode → 性能**。这不是某个 agent
自己延伸出来的范围，是仓库所有者直接给的顺序（D-PM-22）。

`docs/research/pkda_semantics.md` §3 早就点明了这个依赖：PGDN 的公开 ABI 是
`g_atk,beta_atk: [B,T,H]` + `g,beta: [B,T,HV]`（`v: [B,T,HV,V]` 支持 `HV>H` 的 GVA 分组）——
**没有 GQA 分组，PGDN 连 ABI 都对不上，不是"先做哪个都行"的并列关系**。GDA-01 交付的是
显式非 GQA 版本（`HV must equal H`），本任务把这条约束换成真正的分组支持。

## 参照：KDA 已经做过一次同类扩展

本仓 `ascend_fla/ops/kda/chunk.py` 已经支持 `HV % H == 0` 的分组（`AGENTS.md` §2 说的
"KDA 六项能力比 GDN 更接近 fla 语义"里就有这一条）：`q/k` 用 `H` 个 key head，
`v`/`g`/`beta`/`initial_state` 用 `HV` 个 value head，`q/k` 在参与计算前按
`HV // H` 的比例复制（repeat-interleave）到 value-head 维度。**GDN 走同样的模式**，
fla 的 `naive_recurrent_gated_delta_rule`（本仓已有 pin，见 PK-01/PGDN naive.py 里
"GVA is applied if HV > H... each group of HV // H value heads shares one key head"）
用的正是这个语义——直接照抄这条权威定义，不要自己发明分组方式。

## 步骤

1. **冻结新 ABI**：`q,k: [B,T,H,128]`（不变），`v,g,beta: [B,T,HV,...]`（从 `[B,T,H,...]`
   放宽为 `HV % H == 0`），`initial_state: [B,HV,128,128]`。`HV == H` 时退化成 GDA-01
   现在的行为，必须逐位相同——这是最基本的回归判据。
2. **量程与精度**：GQA 分组本身不引入新的指数/除法算式（只是张量维度放宽 + 内部复制），
   `AGENTS.md` §6 那套门控跨度分析不需要重做；但**复制（repeat-interleave）本身要检查
   精度是否受影响**——`q/k` 被多个 value head 共享后，累加顺序和量级可能与非分组情形不同，
   仍要对 fla naive 与独立 CPU 参考双 oracle 验证。
3. **建单元/改单元**：在现有 `kernels/projects/a5/gdn_chunk_fwd/` 里扩展（不是另起一个
   单元）——分组只影响输入张量的维度约定与内部的头映射，五阶段的算式结构不变。
4. **验收网格覆盖 GQA 的边界组合**（`AGENTS.md` §6 "形状维度要逐格扫"）：
   `HV/H ∈ {1, 2, 4, 8}` × `C∈{1,2,3}` × 合法 `block_dim`，覆盖 `HV==H`（退化情形，
   必须与 GDA-01 逐位相同）与 `HV>H` 的多个分组比例。
5. 门控：`HV % H != 0` 或其它不支持组合在发射 kernel 之前报错，错误信息说清楚哪条不满足。
6. 真机验证遵循与 GDA-01 一致的规矩：完整选定 workload 先跑，sim/pipesim 只做定点诊断；
   profile 基线在先，同卡三轮三明治（T=1024/4096）；SoC/CANN/opp 随每个真机数字报。

## 验收

- [ ] 新 ABI 冻结：`q,k` 保持 `H`，`v/g/beta/initial_state` 放宽到 `HV`（`HV % H == 0`）。
- [ ] `HV == H` 与 GDA-01 现有实现逐位相同（硬回归判据）。
- [ ] 双 oracle（fla naive + 独立 CPU 参考）在 `HV/H ∈ {1,2,4,8}` 的网格上验证，
      预算沿用 GDA-01 已定的（FP32 1e-4、BF16 5e-3）。
- [ ] 门控拒绝 `HV % H != 0` 等不支持组合，错误信息明确。
- [ ] 真机数字：完整 workload 先跑，profile 基线在先，三轮三明治，SoC/CANN/opp 随数字报。
- [ ] `git diff --stat` 对 `reserved_paths` 为空。

## 已知陷阱

- 不要发明新的分组语义——直接照抄 fla `naive_recurrent_gated_delta_rule` 的 GVA 处理
  和本仓 KDA 已有的 `HV % H == 0` 约定，两边语义要对得上。
- `HV == H` 退化情形必须逐位不变，这是本任务最容易漏测的回归点。
- 本任务只做前向的 GQA 支持——backward、decode、性能优化都是排在后面的独立任务，
  不要在这条任务里顺带做。
- 完成后 `PK-03`（PGDN 前向）的 `gate: prereq-gdn-abi` 由 PM 根据本任务的完成情况解除，
  不由本任务的 assignee 自行判断"已经够用了"。
