# 启动指南：PM 部署、agent 上线、申领任务

> 协议细节见 `PROTOCOL.md`；本文只讲"怎么把人和会话跑起来"。

## 1. 结构

```
                    ┌──────────────────────────────────────────────┐
  用户 ──（终端 / claude.ai 远程控制）──> │ PM 会话  ascend-fla-dev-management-team      │
                    │ 运行位置：本仓主 checkout，main 分支          │
                    │ 读写：docs/pm/board.json、合入 main           │
                    └───────────────▲──────────────────────────────┘
                                    │ SendMessage（[FLA-PM] 协议）
          ┌─────────────────────────┼──────────────────────────┐
          │                         │                          │
  agent 会话（同一台机器）   agent 会话（另一台机器，     agent 会话 …
  worktree: .claude/worktrees/<ID>   需开 Remote Control）
  分支: task/<ID>                    SSH 到 NPU 机器按 machine_specs.md
```

- **消息通道**是 Claude Code 自带的跨会话消息：`ListAgents` 找人、`SendMessage` 发消息，**会话名就是地址**。
  同机会话直接可达；**跨机器要双方都开 Remote Control 且登录同一个 claude.ai 账号**。
- **云端会话（claude.ai/code 上的 cloud 会话）不能当 agent**：它能收消息但目前不能回消息，无法 ACK/STATUS/DONE。
- PM 是唯一写看板与 main 的会话。**看板是事实源**：PM 会话即使丢了上下文，重启后读 `board.json` 就能接上。

## 2. 部署位置

| 角色 | 在哪跑 | 为什么 |
|---|---|---|
| PM | 用户自己的主力机器上，本仓**主 checkout**（不是 worktree），`main` 分支，一个常开的终端窗口 | 合入 main、提交看板都在这里；开 Remote Control 让其他机器上的 agent 与用户手机都能找到它 |
| 主机侧 agent（W0 任务） | 任意有本仓 clone 的机器；同机最省事 | 只需 Python 环境（+ ascriptor / fla，看任务） |
| 真机 agent（W-A2 起） | 能 SSH 到对应 SoC 机器的开发机 | NPU 工作按 `AGENTS.md` §5 在远端做，agent 会话本身不必在 NPU 机器上 |

## 3. 一次性准备（每台跑会话的机器）

```bash
git clone <本仓> && cd ascend_fla_dev
tools/dev_env.sh              # 建 .venv（Python 3.11 + torch + pytest），并跑一遍主机侧测试
tools/dev_env.sh --with-fla   # 需要 fla oracle 的 agent（A2-05 / A2-07 / A2-08）用这个
claude --version              # 需要支持 --name / --remote-control 的版本
```

- 需要 ascriptor 的 agent：按 `AGENTS.md` §3 准备 `$ASCRIPTOR_WORKSPACE`，并按 `agent/compatibility.json` 选修订。
- 真机 agent：本机要有 git-ignored 的 `machine_specs.md`（**不提交、不在消息里贴机器信息**）。
- 期望的主机侧测试基线（macOS，无 NPU、无 ascriptor 树）：**61 passed / 6 skipped**（6 个跳过 = 5 个需要 torch_npu + 1 个需要 ascriptor kernels 树）。

## 4. 启动 PM

```bash
cd ascend_fla_dev && git checkout main
tools/pm_start.sh            # 校验看板/矩阵/测试 → 启动名为 ascend-fla-dev-management-team 的会话（带 Remote Control）
```

- 脚本会拒绝在非 main 分支或脏工作区上启动。
- `PM_REMOTE_CONTROL=0 tools/pm_start.sh`：只在本机用、不开 Remote Control。
- **同名会话只能有一个**。若已有一个 PM 会话在跑（例如当前这个），不要再起第二个 —— 名字会被加后缀，agent 就找错人。
  要重启：先退出旧会话，或 `claude --resume` 回到它。
- PM 会话闲着时收到消息会被唤醒处理；**但它不会自己定时醒来查心跳**。需要自动心跳时在 PM 会话里执行
  `/loop 30m 按 PROTOCOL.md §5 检查进行中任务的心跳并汇报`。

## 5. 启动 agent

```bash
cd ascend_fla_dev && git checkout main && git pull
tools/agent_start.sh agent-a --ascriptor                 # 纯主机侧、有 ascriptor
tools/agent_start.sh agent-b --fla                       # 纯主机侧、有 fla
tools/agent_start.sh agent-c --socs a2 --ascriptor --fla # 能用 a2 机器（W-A2 开波后）
```

脚本启动一个名为 `agent-a` 的交互会话，附加 `docs/pm/prompts/agent.md` 作为系统提示，并让它自动：
读文档 → `ListAgents` 确认 PM 在线 → 向 PM 发 `APPLY`。

**不用脚本也行**：在任意 Claude Code 会话里 `/rename agent-x`，然后发一句：

> 按 docs/pm/prompts/agent.md 当 agent：能力 socs=none ascriptor=yes fla=no，读完文档后向 PM 申领任务。

## 6. 申领一个任务的完整流程

```
agent                                             PM
  │ ListAgents → 看到 ascend-fla-dev-management-team
  │ SendMessage(to="ascend-fla-dev-management-team",
  │   "[FLA-PM] APPLY - from=agent-a\nsocs: none\nascriptor: yes\nfla: no\n…")
  │ ───────────────────────────────────────────────▶ │ tools/pm_board.py --next --ascriptor
  │                                                 │ 改 board.json（assigned/assignee/branch）→ --check → 提交 main
  │ ◀─────────────────────────────────────────────── │ "[FLA-PM] ASSIGN A2-01 …spec/branch/worktree/timebox"
  │ git worktree add .claude/worktrees/A2-01 -b task/A2-01 main
  │ EnterWorktree(path=".claude/worktrees/A2-01")
  │ "[FLA-PM] ACK A2-01 …"  ───────────────────────▶ │ 状态 → in_progress
  │ … 干活，每 60 分钟 / 每个里程碑 STATUS，发现问题即 RISK …
  │ "[FLA-PM] DONE A2-01 …" ───────────────────────▶ │ 在 task/A2-01 上重跑校验、核数字与原始日志
  │ ◀─────────────────────────────────────────────── │ REVIEW accept（或 rework 清单）
  │                                                 │ merge --no-ff → 看板 done → 提交
  │ ◀─────────────────────────────────────────────── │ CLOSE
  │ ExitWorktree(keep) → 可以再 APPLY
```

回复规则：收到的消息显示为 `<cross-session-message from="…">`，**回复时把 `from` 的值原样当 `to`**。

## 7. 用户怎么看进度、怎么拍板

- 随时：`.venv/bin/python tools/pm_board.py --render`，或在 PM 会话里说"status"。
- `gated` 的任务（kernel 批次、定位变更、资源）只有用户能放行 —— PM 会主动来问。
- A2/A3 机器就绪时：把主机写进 `machine_specs.md`，告诉 PM"a2 机器就绪，卡号 …"，PM 会放开 W-A2 的 gate 并补齐任务规格。

## 8. 故障排查

| 现象 | 原因与处理 |
|---|---|
| agent 的 `ListAgents` 里看不到 PM | 跨机器时双方都要开 Remote Control 且同一账号；PM 会话必须在运行（终端开着或远程在线） |
| 消息发出去了但对方没处理，对方界面显示"held" | 接收方的 `crossSessionInbound` 设置为 hold，或双方权限模式不一致（例如一方跳过权限提示、另一方没有）。**最稳妥是 PM 与 agent 用同一种权限模式启动**；是否把 `crossSessionInbound` 设为 `accept` 由用户在自己的 settings 里决定 |
| 看到两个 `ascend-fla-dev-management-team` | 起了两个 PM。退掉一个；agent 按 `ListAgents` 显示的 `名字 [ref]` 发给正确的那个 |
| PM 会话上下文丢了 / 重启了 | 看板是事实源：`tools/pm_start.sh` 重启，PM 开机流程会读 `board.json` 与各 agent 的在线状态继续 |
| agent 在 macOS 上 `import fla.ops...` 报 `No module named 'triton'` | fla 的 `ops/__init__` 会导入 triton。oracle 只需要 `naive.py`，按文件路径加载即可（见下） |
| `EnterWorktree` 拒绝路径 | 路径必须出现在 `git worktree list` 里；worktree 统一建在 `.claude/worktrees/<ID>` |

在没有 triton 的机器上加载 fla 的 KDA oracle：

```python
import importlib.util, pathlib, fla
p = pathlib.Path(fla.__file__).parent / "ops/kda/naive.py"   # 注意：import fla 本身不触发 triton
spec = importlib.util.spec_from_file_location("fla_kda_naive", p)
naive = importlib.util.module_from_spec(spec); spec.loader.exec_module(naive)
naive.naive_recurrent_kda, naive.naive_chunk_kda
```
