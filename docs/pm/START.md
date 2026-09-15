# 启动指南：PM 部署、agent 上线、申领任务

> 协议细节见 `PROTOCOL.md`；本文只讲"怎么跑起来"。agent 可以是**任何 GitHub 账号、任何模型**（或人）。

## 1. 结构

```
                     GitHub：ddddwee1/ascend_fla_dev（公开）
            ┌─────────────────────────────────────────────────────────┐
            │  申领入口 issue   任务 issue ×N（标签=状态）   PR ×N      │
            └───────▲─────────────────────▲──────────────────▲────────┘
     评论/标签/合入 │                     │ 评论 APPLY/ACK/   │ push 分支、开 PR
                    │                     │ STATUS/RISK/DONE  │
  ┌─────────────────┴──────────┐   ┌──────┴──────────────────┴────────────┐
  │ PM：用户机器上的一个         │   │ agent：任意账号 / 任意模型 / 任意机器  │
  │ Claude Code 会话，           │   │ fork → task/<ID> 分支 → PR            │
  │ gh 登录 **PM bot 账号**，    │   │ 自己管理 NPU 机器（AGENTS.md §5）      │
  │ 每 ~15 分钟 poll 一次        │   └──────────────────────────────────────┘
  │ 唯一写看板 / 合入 main       │
  └──────────────────────────────┘
        ▲ 用户：终端或远程控制看 PM；拍板 gated 项与"新账号首次合入"
```

- **看板是事实源**（`docs/pm/board.json`）。PM 会话丢了上下文也没关系，重启后读看板与 GitHub 就能接上。
- 仓库公开：issue、评论、PR 全网可见。**任何地方都不写机器信息。**

## 2. 部署 PM（一次性，用户操作 + PM 协助）

PM 跑在**用户自己的机器**上，本仓**主 checkout**（不是 worktree）的 `main` 分支，一个常开的终端窗口。

1. **建 PM bot 账号**：在 GitHub 注册一个专用账号（例如 `<你的名字>-fla-pm`），开两步验证。
2. **给 bot 写权限**：仓库 Settings → Collaborators → 邀请 bot 账号（Write），用 bot 账号接受邀请。
3. **本机装好工具**：
   ```bash
   tools/dev_env.sh                 # .venv（Python 3.11 + torch + pytest），并跑一遍主机侧测试
   gh --version                     # 没有就装 GitHub CLI
   ```
4. **用 bot 账号登录 gh**（交互式，在 Claude Code 里用 `!` 前缀自己运行）：
   ```bash
   ! gh auth login                  # 选 GitHub.com → HTTPS → 用 bot 账号浏览器登录
   ! gh auth setup-git              # 让 git push 走 bot 账号的凭据
   git config user.name  "<bot 账号>"
   git config user.email "<bot 账号>@users.noreply.github.com"
   ```
   已经登录过个人账号时，`gh auth login` 会新增账号，用 `gh auth switch` 切到 bot。
5. **把 bot 账号写进看板**：`docs/pm/board.json` 的 `pm_github_login`，提交推送。
6. **预览再发布**：
   ```bash
   .venv/bin/python tools/pm_github.py whoami          # 必须显示 gh 账号 == pm_github_login
   .venv/bin/python tools/pm_github.py sync            # dry-run：列出要建的标签、申领入口与任务 issue
   .venv/bin/python tools/pm_github.py sync --apply    # 用户确认后执行；issue 编号写回看板，再提交推送
   ```
7. **启动 PM 会话**：
   ```bash
   tools/pm_start.sh
   ```
   脚本会校验：在主 checkout 的 main、工作区干净、gh 账号就是 bot、看板/矩阵/测试都过，然后启动名为
   `ascend-fla-dev-management-team` 的 Claude Code 会话（附 `docs/pm/prompts/pm.md`），开机后按 15 分钟一轮轮询。
   `PM_REMOTE_CONTROL=0` 可以不开 Remote Control（开着时用户能从手机看 PM）。

> 同一时间只跑**一个** PM 会话：两个 PM 会重复派单、互相覆盖看板。

## 3. agent 上线（任何账号、任何模型）

```bash
# 1. fork 本仓到你的账号，然后
git clone https://github.com/<you>/ascend_fla_dev && cd ascend_fla_dev
git remote add upstream https://github.com/ddddwee1/ascend_fla_dev.git
# 2. 环境（主机侧测试基线：61+ passed，NPU / ascriptor 相关的会 skip）
tools/dev_env.sh [--with-fla]
# 3. 生成 APPLY 评论（按你的真实能力填）
tools/agent_setup.sh --login <you> --agent "<模型或工具名>" --task any --ascriptor "<修订号>" --fla
```

然后把输出的 APPLY 贴到**申领入口 issue**（`any`）或指定**任务 issue** 下（网页、`gh issue comment` 或 API 都行）。

**让模型当 agent**：把 `docs/pm/prompts/agent.md` 作为系统提示或第一条消息给它，并告诉它你的 GitHub 账号与能力。
这份提示不依赖任何特定工具：只要能读写本地仓库、跑命令、在 GitHub 上评论和开 PR 即可。

## 4. 申领一个任务的完整流程

```
agent（GitHub 账号 alice）                                  PM（bot 账号）
  │ 在申领入口 issue 评论：
  │ [FLA-PM] APPLY any from=alice …能力…
  │ ─────────────────────────────────────────────────▶ │ poll 拿到 → pm_board.py --next 取候选
  │                                                     │ 看板 assigned/assignee=alice/assignee_caps → 校验 → 推 main → sync
  │ ◀───────────────────────────────────────────────── │ 在 A2-01 issue 评论 [FLA-PM] ASSIGN A2-01 …
  │ 核对：ASSIGN 作者 == 看板 pm_github_login
  │ git fetch upstream && git switch -c task/A2-01 upstream/main
  │ 评论 [FLA-PM] ACK A2-01 …  ─────────────────────▶ │ 状态 → in_progress
  │ … 干活；STATUS / RISK 发在 A2-01 issue 下 …
  │ git push origin task/A2-01；向 upstream main 开 PR "[A2-01] …  Refs #<issue>"
  │ 评论 [FLA-PM] DONE A2-01 pr: #34 …  ────────────▶ │ ① 读完整 diff（不执行） ② 核写集
  │                                                     │ ③ 临时 clone 里跑校验 ④ 核验收数字
  │                                                     │ ⑤ 新账号首次合入 → 先问用户
  │ ◀───────────────────────────────────────────────── │ PR 下 REVIEW accept（或 rework 清单）
  │                                                     │ 合入 → 看板 done → 关 issue
  │ ◀───────────────────────────────────────────────── │ CLOSE → alice 可以再 APPLY
```

## 5. 用户怎么看进度、怎么拍板

- 进度：GitHub 上按标签筛 `fla-pm`；或本机 `.venv/bin/python tools/pm_board.py --render`；或在 PM 会话里说"status"。
- 需要用户决定的事，PM 会主动来问：`gated` 项放行（kernel 批次、定位变更、资源、开新波次）、**每个新账号的第一次合入**、
  首次把任务发布成 GitHub issue。
- A2/A3 机器就绪不再需要告诉 PM 机器细节 —— 机器归 agent；PM 会在有 agent 声明 `socs: a2` 后放开 W-A2 的 gate（需用户同意开波）。

## 6. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `pm_github.py` 拒绝写：`当前 gh 账号是 X，不是 PM 账号` | 故意的保护：用 `gh auth switch` 切回 bot 账号。不要把 `pm_github_login` 改成个人账号 |
| `pm_github_login 还是 null` | §2 第 5 步没做 |
| agent 说收到了 ASSIGN，但看板里不是他 | 可能有人冒充：只认 `pm_github_login` 发的 ASSIGN。PM 的 poll 会对非 PM 账号发的 PM 类消息打"疑似冒充"警告 |
| 评论发了 PM 没反应 | 首行不是 `[FLA-PM] <TYPE> …`（前面有闲话 / 格式不对），或 PM 会话没在跑。PM 每轮 poll 会回复格式错误的协议消息 |
| PR 被整体拒绝 | 改了写集外的文件、动了 `.github/`、加了依赖或网络/凭据相关代码（PROTOCOL §4.9） |
| 在 macOS 上 `import fla.ops…` 报 `No module named 'triton'` | fla 的 `ops/__init__` 会导入 triton。oracle 只需要 `naive.py`，按文件路径加载（见下） |

在没有 triton 的机器上加载 fla 的 KDA oracle：

```python
import importlib.util, pathlib, fla
p = pathlib.Path(fla.__file__).parent / "ops/kda/naive.py"   # import fla 本身不触发 triton
spec = importlib.util.spec_from_file_location("fla_kda_naive", p)
naive = importlib.util.module_from_spec(spec); spec.loader.exec_module(naive)
naive.naive_recurrent_kda, naive.naive_chunk_kda
```
