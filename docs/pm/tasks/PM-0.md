# PM-0 建立多 agent 任务管理系统，并记录 SoC 顺序决策

- 波次 / SoC：W0 / any（纯主机侧）· 负责人：PM · 时限：6h · 依赖：无

## 目标

1. 建立 PM 的事实源与工具：`docs/pm/board.json`、`docs/pm/PROTOCOL.md`、`docs/pm/tasks/*.md`、
   `tools/pm_board.py`（`--check` / `--render` / `--next`）、`tests/test_pm_board.py`。
2. 把 2026-09-14 的用户决策写进长期文档，否则 agent 会照旧版 `AGENTS.md` 拒绝做 A2：
   - SoC 顺序 **A2 (910B) → A3 (910C) → A5**；首波模型 Kimi-Linear（KDA）然后 Qwen3-Next（GDN）。
   - 工具链只用 ascriptor；A2 上先定性 split-K FP32 cube 缺陷。
3. 按 PM 计划切出 W0 的 8 个主机侧任务规格。

## 写集

`docs/pm/**`、`tools/pm_board.py`、`tests/test_pm_board.py`、`AGENTS.md`（§2 / §6 / §9 / 顶部指引）、
`docs/plan.md`（§5）、`docs/handoff.md`（新增顶部一节）。

## 验收

- `python tools/pm_board.py --check` 通过；`tests/test_pm_board.py` 全过，且每条冲突（写集交叠、卡重复租出、
  依赖成环、依赖未完成却在进行中、缺 spec、IP 地址）都有反例测试。
- `python tools/gen_matrix.py --check` 仍通过（本任务不改矩阵 json；矩阵按 SoC 分维是 A2-06 的事）。
- 主机侧全量 `pytest tests/ -q` 不退化（基线 42 passed / 5 skipped）。
- `AGENTS.md` 的改动经用户过目后再合入 main。
