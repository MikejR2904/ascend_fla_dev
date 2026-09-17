# ascend_fla

fla 系列线性注意力算子在昇腾 NPU 上的高效实现库。后端用
[ascriptor](https://github.com/ddddwee1/ascriptor)（指令级 Python 编译器），
以 torch 可调用算子对外暴露，目标是比现有方案更快。

这不是 fla 的 fork，也不是它的后端插件 —— 公共 API 由本仓自己定义，
`fla` 只在测试期充当语义权威（`naive.py` 作 CPU fp32 oracle）。详见 `AGENTS.md`。

| 想看什么 | 去哪 |
|---|---|
| 长期工作纪律 | [`AGENTS.md`](AGENTS.md) |
| 此刻进度与已纠正的判断 | [`docs/handoff.md`](docs/handoff.md) |
| 支持矩阵（json 权威，md 生成） | [`docs/matrix/`](docs/matrix/) |
| English | [README.en.md](README.en.md) |
| 想接活 / 提需求 | [`docs/pm/START.md`](docs/pm/START.md)、[申领入口 #28](https://github.com/ddddwee1/ascend_fla_dev/issues/28) |

> 仓库里有两条并行轨道：仓主在 A5 上做 GDN-2（见 `docs/handoff.md` §1），
> 外部 agent 走下面这张表里的任务（A2 → A3 → A5）。两条共用仓库、互不指挥，
> 边界由 `docs/pm/board.json` 的 `reserved_paths` 划定。

## 算子进展

<!-- fla-pm:kernels start -->
<!-- 由 tools/pm_board.py --readme 生成，请勿手改。改 docs/pm/board.json 后重新运行。 -->

| 算子族 | id | 归属 | kernel 数 | 任务进度 |
|---|---|---|---|---|
| KDA（Kimi Delta Attention） | `kda` | 可申领 | 5 | 1/14 |
| GDN（Gated DeltaNet） | `gated_delta_rule` | 可申领 | 3 | 0/3 |
| DeltaNet | `delta_rule` | 尚未排任务 | 2 | 2/2 单元有验证记录 |
| GDN-2（Gated DeltaNet 2） | `gdn2` | 仓主轨道 | 5 | 4/5 单元有验证记录 |
| 整网融合算子（模块 / 层级） | `fusion` | 可申领 | 4 | 0/4 |
| Mamba-1/2/3 | `mamba` | 未排期 (G4) | — | — |
| GLA（Gated Linear Attention） | `gla` | 未排期 (G4) | — | — |
| PKDA / PGDN（预条件） | `pkda` | 尚未排任务 | — | — |
| Log-Linear Attention | `log_linear` | 未排期 (G4) | — | — |
| DLA（动态线性注意力） | `dla` | 未排期 (G4) | — | — |
| StateX（宽状态） | `statex` | 未排期 (G4) | — | — |
| NAtS-L（动态路由混合） | `natsl` | 未排期 (G4) | — | — |
| NHA（Native Hybrid Attention） | `nha` | 未排期 (G4) | — | — |
| Local Linear Attention | `local_linear` | 未排期 (G4) | — | — |
| DeltaStack（栈式） | `deltastack` | 未排期 (G4) | — | — |
| TTT / MesaNet | `ttt` | 未排期 (G4) | — | — |
| NSA / 稀疏注意力 | `nsa` | 未排期 (G4) | — | — |

### 展开看细节（算子族 → kernel → 任务）

<details><summary><b>KDA（Kimi Delta Attention） —— 5 个 kernel，1/14 完成</b></summary>

_首个目标算子族，Kimi-Linear 用它_

| kernel | 归属 | 进度 | 下一步 |
|---|---|---|---|
| `kda_fwd_stable` | 可申领 | 1/8 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) A2-03 |
| `kda_bwd_stable` | 可申领 | 0/8 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) A2-03 |
| `kda_fused_recurrent` | 可申领 | 0/9 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) A2-03 |
| `kda_fwd` | 上游单元 | 5/9 passed | _上游 ascriptor 单元，本仓用 kda_fwd_stable 取代_ |
| `kda_bwd` | 上游单元 | 4/9 passed，1 项 gap | _上游 ascriptor 单元，本仓用 kda_bwd_stable 取代_ |

<details><summary>kda_fwd_stable —— 1/8 完成，起点 A2-03、A2-04、A5-04</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-04 | [#30](https://github.com/ddddwee1/ascend_fla_dev/issues/30) | `any` | bf16 | ✅ done | ★ 起点 |
| A2-03 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A5-04 | [#50](https://github.com/ddddwee1/ascend_fla_dev/issues/50) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A2-12 | [#39](https://github.com/ddddwee1/ascend_fla_dev/issues/39) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |
| A2-15 | [#42](https://github.com/ddddwee1/ascend_fla_dev/issues/42) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A2-16 | [#43](https://github.com/ddddwee1/ascend_fla_dev/issues/43) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A5-06 | [#52](https://github.com/ddddwee1/ascend_fla_dev/issues/52) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) |  |

</details>

<details><summary>kda_bwd_stable —— 0/8 完成，起点 A2-03、A2-K1、A5-04、A5-05</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-03 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) | ★ 起点 |
| A5-04 | [#50](https://github.com/ddddwee1/ascend_fla_dev/issues/50) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A5-05 | [#51](https://github.com/ddddwee1/ascend_fla_dev/issues/51) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A2-13 | [#40](https://github.com/ddddwee1/ascend_fla_dev/issues/40) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A2-15 | [#42](https://github.com/ddddwee1/ascend_fla_dev/issues/42) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A2-16 | [#43](https://github.com/ddddwee1/ascend_fla_dev/issues/43) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A5-06 | [#52](https://github.com/ddddwee1/ascend_fla_dev/issues/52) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) |  |

</details>

<details><summary>kda_fused_recurrent —— 0/9 完成，起点 A2-03、A2-K1、A5-01、A5-02、A5-03</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-03 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) | ★ 起点 |
| A5-01 | [#47](https://github.com/ddddwee1/ascend_fla_dev/issues/47) | `a5` | fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A5-02 | [#48](https://github.com/ddddwee1/ascend_fla_dev/issues/48) | `a5` | fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A5-03 | [#49](https://github.com/ddddwee1/ascend_fla_dev/issues/49) | `a5` | fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A2-14 | [#41](https://github.com/ddddwee1/ascend_fla_dev/issues/41) | `a2` | fp32 | 🔒 gated (machines:a2) |  |
| A2-15 | [#42](https://github.com/ddddwee1/ascend_fla_dev/issues/42) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A2-16 | [#43](https://github.com/ddddwee1/ascend_fla_dev/issues/43) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A5-06 | [#52](https://github.com/ddddwee1/ascend_fla_dev/issues/52) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) |  |

</details>

</details>

<details><summary><b>GDN（Gated DeltaNet） —— 3 个 kernel，0/3 完成</b></summary>

_Qwen3-Next 用它，六项 ABI 缺口待补_

| kernel | 归属 | 进度 | 下一步 |
|---|---|---|---|
| `gdn_fwd` | 可申领 | 0/3 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) A2-07 |
| `gdn_bwd` | 可申领 | 0/3 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) A2-07 |
| `gdn_fused_recurrent` | 可申领 | 0/1 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) A2-20 |

<details><summary>gdn_fwd —— 0/3 完成，起点 A2-07</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-07 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |

</details>

<details><summary>gdn_bwd —— 0/3 完成，起点 A2-07</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-07 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |

</details>

<details><summary>gdn_fused_recurrent —— 0/1 完成，起点 A2-20</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) | ★ 起点 |

</details>

</details>

<details><summary><b>DeltaNet —— 2 个 kernel</b></summary>

_上游单元已存在，本仓尚未排任务_

| kernel | 归属 | 进度 | 下一步 |
|---|---|---|---|
| `delta_rule_fwd` | 尚未排任务 | 7/9 passed | _上游单元已存在，本仓尚未排任务_ |
| `delta_rule_bwd` | 尚未排任务 | 7/9 passed | _上游单元已存在，本仓尚未排任务_ |

</details>

<details><summary><b>GDN-2（Gated DeltaNet 2） —— 5 个 kernel</b></summary>

_仓主并行轨道，见 docs/handoff.md §1_

| kernel | 归属 | 进度 | 下一步 |
|---|---|---|---|
| `gdn2_fused_recurrent` | 仓主轨道 | 6/6 passed | _仓主轨道：通用 CCE recurrent_ |
| `gdn2_fused_decode` | 仓主轨道 | 6/6 passed | _仓主轨道：模型专用融合 decode_ |
| `gdn2_short_conv_decode` | 仓主轨道 | 6/6 passed | _仓主轨道：打包短卷积 decode_ |
| `gdn2_norm2_w12_swiglu` | 仓主轨道 | 4/6 passed | _仓主轨道：融合 RMSNorm + SwiGLU_ |
| `gdn2_chunk_fwd_bwd` | 仓主轨道 | — | _仓主轨道：尚未实现，量程需单独设计_ |

</details>

<details><summary><b>整网融合算子（模块 / 层级） —— 4 个 kernel，0/4 完成</b></summary>

_算子本身之外，整网跑起来还要的那些：短卷积、门控 RMSNorm、q/k l2norm 与门控变换、合投影。现在全是 torch 原生算子（gaps: modules-are-torch-not-kernels / qk-l2norm-not-in-kernel / decode-layer-overhead）_

| kernel | 归属 | 进度 | 下一步 |
|---|---|---|---|
| `causal_conv1d` | 可申领 | 0/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |
| `fused_rms_norm_gated` | 可申领 | 0/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |
| `qk_l2norm_gate` | 可申领 | 0/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |
| `packed_projection` | 可申领 | 0/2 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) A2-40 |

<details><summary>causal_conv1d —— 0/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-41 | [#55](https://github.com/ddddwee1/ascend_fla_dev/issues/55) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |

</details>

<details><summary>fused_rms_norm_gated —— 0/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-41 | [#55](https://github.com/ddddwee1/ascend_fla_dev/issues/55) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |

</details>

<details><summary>qk_l2norm_gate —— 0/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-42 | [#56](https://github.com/ddddwee1/ascend_fla_dev/issues/56) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |

</details>

<details><summary>packed_projection —— 0/2 完成，起点 A2-40</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-40 | [#54](https://github.com/ddddwee1/ascend_fla_dev/issues/54) | `any` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-43 | [#57](https://github.com/ddddwee1/ascend_fla_dev/issues/57) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |

</details>

</details>

<details><summary><b>Mamba-1/2/3 —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>GLA（Gated Linear Attention） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>PKDA / PGDN（预条件） —— 本仓暂无 kernel</b></summary>

_用户 2026-09-16 点名细化排期。**fla 0.5.2 里没有 pkda 实现**，那三份 Gemini 文档也只给了机制描述（测试时回归 / 对角 Hessian 预条件），没有可对照的参考实现 —— 所以第一步是把权威来源找出来，见 PK-01_

</details>

<details><summary><b>Log-Linear Attention —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>DLA（动态线性注意力） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>StateX（宽状态） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>NAtS-L（动态路由混合） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>NHA（Native Hybrid Attention） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>Local Linear Attention —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>DeltaStack（栈式） —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>TTT / MesaNet —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>

<details><summary><b>NSA / 稀疏注意力 —— 本仓暂无 kernel</b></summary>

_未排期：属 gated epic G4（窄切片原则 —— 没有目标模型就不做），需用户放行_

</details>


### 数据类型

| dtype | 归属 | 涉及任务 | 说明 |
|---|---|---|---|
| `bf16` | 可申领 | 17 | 当前 ABI：q/k/v/o 与多数中间量 |
| `fp32` | 可申领 | 24 | 当前 ABI：state、门控累加、精度判定一律 fp32 |
| `fp16` | 未排期 (G3) | — | 契约明确拒绝（见 ops.json 的 no-tail-path 一条） |
| `int8` | 未排期 (G3) | 1 | 未排期；state 累积漂移需先有方案 |
| `mxfp8` | 未排期 (G3) | 1 | 未排期；950 原生解码待核实 |
| `hif8` | 未排期 (G3) | 1 | 未排期 |
| `mxfp4` | 未排期 (G3) | 1 | 未排期；需配 RHT 抑制离群值 |
| `hif4` | 未排期 (G3) | 1 | 未排期；三级微指数解码未核实 |

> 图例：⬜ 可申领 · 🔵 进行中 · 🔒 有前置条件未满足 · ✅ 已完成 · — 还没有任务。
> ★ 起点 = 该 kernel 链里传递依赖全在组外的任务，也就是要让这个 kernel 动起来先做哪一条。
> 任务顺序由依赖关系算出，不是手写的。**仓主轨道**不派给 agent（见 `docs/handoff.md` §1）；
> **未排期 (G4)** 的算子族本仓没有 kernel，按窄切片原则要有目标模型才开工。
<!-- fla-pm:kernels end -->
