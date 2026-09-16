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
| 想接活 / 提需求 | [`docs/pm/START.md`](docs/pm/START.md)、[申领入口 #28](https://github.com/ddddwee1/ascend_fla_dev/issues/28) |

> 仓库里有两条并行轨道：仓主在 A5 上做 GDN-2（见 `docs/handoff.md` §1），
> 外部 agent 走下面这张表里的任务（A2 → A3 → A5）。两条共用仓库、互不指挥，
> 边界由 `docs/pm/board.json` 的 `reserved_paths` 划定。

## 算子进展

<!-- fla-pm:kernels start -->
<!-- 由 tools/pm_board.py --readme 生成，请勿手改。改 docs/pm/board.json 后重新运行。 -->

| kernel | 目标模型 | 进度 | 下一步 |
|---|---|---|---|
| `gdn_bwd` | kimi-linear-48b-a3b、qwen3-next-80b-a3b | 0/3 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) A2-07 |
| `gdn_fused_recurrent` | qwen3-next-80b-a3b | 0/1 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) A2-20 |
| `gdn_fwd` | kimi-linear-48b-a3b、qwen3-next-80b-a3b | 0/3 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) A2-07 |
| `kda_bwd_stable` | kimi-linear-48b-a3b、qwen3-next-80b-a3b | 0/8 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) A2-03 |
| `kda_fused_recurrent` | kimi-linear-48b-a3b、qwen3-next-80b-a3b | 0/9 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) A2-03 |
| `kda_fwd_stable` | kimi-linear-48b-a3b、qwen3-next-80b-a3b | 0/8 | [#30](https://github.com/ddddwee1/ascend_fla_dev/issues/30) A2-04 |

<details><summary><b>gdn_bwd</b> —— 0/3 完成，起点 A2-07</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-07 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |

</details>

<details><summary><b>gdn_fused_recurrent</b> —— 0/1 完成，起点 A2-20</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) | ★ 起点 |

</details>

<details><summary><b>gdn_fwd</b> —— 0/3 完成，起点 A2-07</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-07 | [#35](https://github.com/ddddwee1/ascend_fla_dev/issues/35) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |
| A2-20 | [#45](https://github.com/ddddwee1/ascend_fla_dev/issues/45) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |

</details>

<details><summary><b>kda_bwd_stable</b> —— 0/8 完成，起点 A2-03、A2-K1、A5-04、A5-05</summary>

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

<details><summary><b>kda_fused_recurrent</b> —— 0/9 完成，起点 A2-03、A2-K1、A5-01、A5-02、A5-03</summary>

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

<details><summary><b>kda_fwd_stable</b> —— 0/8 完成，起点 A2-04、A2-03、A5-04</summary>

| 任务 | issue | SoC | dtype | 状态 | 说明 |
|---|---|---|---|---|---|
| A2-04 | [#30](https://github.com/ddddwee1/ascend_fla_dev/issues/30) | `any` | bf16 | ⬜ open | ★ 起点 |
| A2-03 | [#34](https://github.com/ddddwee1/ascend_fla_dev/issues/34) | `a2` | bf16、fp32 | ⬜ open | ★ 起点 |
| A5-04 | [#50](https://github.com/ddddwee1/ascend_fla_dev/issues/50) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) | ★ 起点 |
| A2-12 | [#39](https://github.com/ddddwee1/ascend_fla_dev/issues/39) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A2-K1 | [#44](https://github.com/ddddwee1/ascend_fla_dev/issues/44) | `a2` | bf16、fp32 | 🔒 gated (kernel-batch-approval) |  |
| A2-15 | [#42](https://github.com/ddddwee1/ascend_fla_dev/issues/42) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A2-16 | [#43](https://github.com/ddddwee1/ascend_fla_dev/issues/43) | `a2` | bf16、fp32 | 🔒 gated (machines:a2) |  |
| A5-06 | [#52](https://github.com/ddddwee1/ascend_fla_dev/issues/52) | `a5` | bf16、fp32 | 🔒 gated (wave:W-A3) |  |

</details>

> 图例：⬜ 可申领 · 🔵 进行中 · 🔒 有前置条件未满足 · ✅ 已完成。
> ★ 起点 = 该 kernel 链里依赖全在组外的任务，也就是要让这个 kernel 动起来先做哪一条。
> 任务顺序由依赖关系算出，不是手写的。
<!-- fla-pm:kernels end -->
