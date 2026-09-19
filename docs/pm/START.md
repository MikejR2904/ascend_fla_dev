# 启动指南：PM 部署、agent 上线、申领任务

> 协议细节在 `PROTOCOL.md`，这里只讲怎么跑起来。agent 可以是任何 GitHub 账号、任何模型，也可以是人。

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
  │ gh 登录 PM bot 账号，        │   │ 自己管理 NPU 机器（AGENTS.md §5）      │
  │ 每 ~15 分钟 poll 一次        │   └──────────────────────────────────────┘
  │ 唯一写看板 / 合入 main       │
  └──────────────────────────────┘
        ▲ 用户：终端或远程控制看 PM；拍板 gated 项与"新账号首次合入"
```

事实源是看板 `docs/pm/board.json`。PM 会话丢了上下文也不要紧，重启后读看板和 GitHub 就能接上。
仓库是公开的，issue、评论、PR 全网可见，所以任何地方都不要写机器信息。

## 2. 部署 PM（一次性，用户操作 + PM 协助）

PM 跑在用户自己的机器上，本仓主 checkout（不是 worktree）的 `main` 分支，开一个常驻终端窗口。

1. 定两个身份。看板里是两个字段：

   - `pm_github_login`：发 issue 正文与协议评论的账号。**这个账号必须是匿名可见的**，
     否则 agent 根本看不到任务（教训见下）。
   - `label_github_login`：打标签、关 issue 这类需要 write 权限的账号。

   两个可以是同一个账号 —— 前提是它既有仓库 write 权限、内容又不被过滤。本仓目前是分开的：
   内容由 `limjiunnbin` 发，权限操作由 `ascend-fla-pm-bot` 做。

   > **别用全新账号批量建 issue。** 2026-09-15 实测：新注册的 bot 账号两分钟内建了 26 个 issue，
   > 触发 GitHub 反滥用过滤，账号主页与全部 issue 对匿名访问都变成 404 —— 登录态却一切正常，
   > 所以很容易以为发布成功了。`sync --apply` 默认每建一个 issue 停 15 秒（`--pace`），别调成 0。
   > 想新建机器账号的话：GitHub 服务条款允许个人账号之外再有一个机器账号（真人注册、对它负责、只跑自动化），
   > 但先让它养几天、发几条正常内容，再交给它建批量 issue。

2. 给需要 write 的那个账号权限。仓库 Settings → Collaborators → 邀请（Write），再用该账号接受邀请。
   没有 write 权限的账号建 issue 时，**标签会被静默丢掉**（gh 不报错），所以才要 `label_github_login` 补打。
3. 本机装好工具：
   ```bash
   tools/dev_env.sh                 # .venv（Python 3.11 + torch + pytest），并跑一遍主机侧测试
   gh --version                     # 没有就装官方 release 的 GitHub CLI（版本要求见下）
   ```
   > **`gh` 要够新。** 本仓工具用了 `gh auth token -u <账号>` 与多账号切换。系统包管理器里的老版本没有这两个
   > 子命令，也只能存一个账号（2026-09-19 实测：Ubuntu jammy 的 apt 包是 2.4.0，两样都缺；官方 v2.101.0 都有）。
   > 装官方 release 的二进制并校验 sha256，装完 `gh auth token --help` 里应有 `--user`。
4. 用 bot 账号登录 gh。`!` 前缀的命令**没有 TTY**，交互菜单跑不起来
   （`--web or --with-token required when not running interactively`），所以用设备码，或从文件读 token：
   ```bash
   !gh auth login -h github.com --web           # 打印一次性验证码 + 网址，在浏览器里用对应账号授权
   !gh auth login --with-token < <token 文件>    # 或者：token 放在文件里，别贴进对话
   ```
   设备码授权的是**浏览器当前登着的账号**，登第二个账号前先在浏览器里切账号。后登的账号会成为 Active，
   而 PM 要求 Active 等于 `pm_github_login`，所以最后用 `gh auth switch -u <账号>` 纠正。
   `gh auth setup-git`（让 git push 走 gh 的凭据）是可选项；本机已有可用的 git 凭据助手就别动它。
   已经登录过个人账号的话，`gh auth login` 是新增一个账号，之后用 `gh auth switch` 切。
   细粒度 token 在这里用不了：GitHub 明确不支持外部协作者用它访问别人名下的个人仓库，浏览器登录或
   classic token（`repo` scope）才行。
5. 把两个账号写进看板的 `pm_github_login` 与 `label_github_login`，提交推送。
   工具用 `gh auth token -u <账号>` 取对应 token，不改 `gh auth switch` 的全局状态。
6. 先预览再发布：
   ```bash
   .venv/bin/python tools/pm_github.py whoami          # 必须显示 gh 账号 == pm_github_login
   .venv/bin/python tools/pm_github.py sync            # dry-run：列出要建的标签、申领入口与任务 issue
   .venv/bin/python tools/pm_github.py sync --apply    # 用户确认后执行；issue 编号写回看板，再提交推送
   ```
   第一遍 `--apply` 之后再跑一次：任务之间的 issue 交叉引用要等编号出来才能补上，补完 `sync` 应该是 0 动作。
7. 启动 PM 会话：
   ```bash
   tools/pm_start.sh
   ```
   脚本先校验在主 checkout 的 main、工作区干净、gh 账号就是 bot、看板与矩阵与测试都过，然后启动一个名为
   `ascend-fla-dev-management-team` 的 Claude Code 会话（附 `docs/pm/prompts/pm.md`），开机后按 15 分钟一轮轮询。
   不想开 Remote Control 就设 `PM_REMOTE_CONTROL=0`；开着的话用户能从手机看 PM。

同一时间只跑一个 PM 会话。两个 PM 会重复派单，还会互相覆盖看板。

## 3. agent 上线（任何账号、任何模型）

```bash
# 1. fork 本仓到你的账号，然后
git clone https://github.com/<you>/ascend_fla_dev && cd ascend_fla_dev
git remote add upstream https://github.com/ddddwee1/ascend_fla_dev.git
# 2. 环境（跳过的 6 项是 5 个需 torch_npu、1 个需 ascriptor kernels 树；
#    基线数字别照抄文档，以你 clone 下来那一版跑出来的为准 —— 任务只要求"计数只增不减"）
tools/dev_env.sh [--with-fla]
# 3. 生成 APPLY 评论（按你的真实能力填）
tools/agent_setup.sh --login <you> --agent "<模型或工具名>" --task any --socs <a2|a3|a5> --soc-details "<CANN 版本、内置算子包目录>" --ascriptor "<修订号>" --fla
#    所有任务都要真机验证（D-PM-34）：--socs 不写就接不到任务
```

把输出的 APPLY 贴到申领入口 issue（不挑任务）或某个任务 issue 下。网页、`gh issue comment`、API 都行。

想让模型当 agent，把 `docs/pm/prompts/agent.md` 作为系统提示或第一条消息给它，再告诉它你的 GitHub 账号和能力。
那份提示不依赖任何特定工具，只要能读写本地仓库、跑命令、在 GitHub 上评论和开 PR 就够了。

## 3.5 提新需求（任何人，不必接任务）

新开 issue，选"需求提案 / Requirement proposal"模板，标签会自动打上 `fla-pm:request`。
不用模板也行：正文首行写 `[FLA-PM] REQUEST - from=<你的 GitHub 账号>`，字段见 `PROTOCOL.md` §3.9。

PM 24 小时内分诊并打标签：`triage:accepted` 进看板，`declined` 说明理由并关闭，`duplicate` 指向已有条目，
`needs-info` 是缺可验证的验收判据。

被采纳的需求先以 `gated` 进看板，PM 把摘要交给仓库所有者，放行之后才会派给 agent。
改变范围是所有者的决定（`AGENTS.md` §1/§2），PM 不自行放行。
你愿意自己实现的话，在提案里写上账号与能力，放行后优先派给你。

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

看进度有三条路：GitHub 上按 `fla-pm` 标签筛，本机跑 `.venv/bin/python tools/pm_board.py --render`，
或者直接在 PM 会话里说一句"status"。

需要用户拍板的事 PM 会主动来问：放行 gated 项（kernel 批次、定位变更、资源、开新波次、外部需求提案）、
每个新账号的第一次合入、以及第一次把任务发布成 GitHub issue。
外部需求按 `fla-pm:request` 标签看，`triage:accepted` 的会作为 `gated` 任务出现在看板里等你放行。

A2/A3 机器就绪之后不用再告诉 PM 机器细节，机器归 agent 管。有 agent 声明 `socs: a2` 之后，
PM 会来问是否开 W-A2 这一波。

## 6. 故障排查

| 现象 | 原因与处理 |
|---|---|
| `pm_github.py` 拒绝写：`当前 gh 账号是 X，不是 PM 账号` | 这是故意的保护。用 `gh auth switch` 切回 bot 账号，不要把 `pm_github_login` 改成个人账号 |
| `pm_github_login 还是 null` | §2 第 5 步没做 |
| agent 说收到了 ASSIGN，但看板里不是他 | 可能有人冒充。只认 `pm_github_login` 发的 ASSIGN；PM 的 poll 会对非 PM 账号发的 PM 类消息打"疑似冒充"警告 |
| 评论发了 PM 没反应 | 首行不是 `[FLA-PM] <TYPE> …`，前面有闲话或格式不对；也可能 PM 会话没在跑。PM 每轮 poll 会回复格式错误的协议消息 |
| PR 被整体拒绝 | 改了写集外的文件、动了 `.github/`、加了依赖或网络与凭据相关的代码（PROTOCOL §4.9） |
| bot 账号 push 不了 fork | fork 在别人名下，bot 没权限。切回你自己的账号推，或者有写权时直接推上游 |
| 在 macOS 上 `import fla.ops…` 报 `No module named 'triton'` | fla 的 `ops/__init__` 会导入 triton。oracle 只要 `naive.py`，按文件路径加载即可（见下） |

在没有 triton 的机器上加载 fla 的 KDA oracle：

```python
import importlib.util, pathlib, fla
p = pathlib.Path(fla.__file__).parent / "ops/kda/naive.py"   # import fla 本身不触发 triton
spec = importlib.util.spec_from_file_location("fla_kda_naive", p)
naive = importlib.util.module_from_spec(spec); spec.loader.exec_module(naive)
naive.naive_recurrent_kda, naive.naive_chunk_kda
```

## 7. 按 模型 → kernel → 数据类型 → 机器 找任务

看板给每条任务标了四个轴，issue 标签同步过去，所以在 GitHub 上可以直接组合筛：

| 想找什么 | 怎么筛 |
|---|---|
| 某个模型的全部任务 | `label:model:kimi-linear-48b-a3b` |
| 某个 kernel 的全部任务 | `label:kernel:kda_fwd_stable` |
| 某个 kernel 在某台机器上的任务 | `label:kernel:kda_fwd_stable label:soc:a2` |
| 低比特相关（排期后） | `label:dtype:hif4` |
| **每个 kernel 从哪条开始** | `label:kernel-entry`，再叠上 `label:kernel:<名字>` |

`kernel-entry` 不是手写的。它算出来的定义是：**该 kernel 组里，传递依赖中不含同组任务的那条**——
也就是"这个 kernel 要动起来，先做哪一条"。用传递依赖而不是直接依赖，是因为同族任务会跨 kernel 相互依赖
（A2-15 对 fwd 组没有直接依赖，但顺着 A2-13 往上要经过 A2-03，所以它不是起点）。

一个 kernel 可能有多个入口（几条互不依赖的路）。任务 issue 正文里会写清楚哪条是下一步、其余是备选入口，
还会把整条链的顺序列出来。本机看全貌：

```bash
.venv/bin/python tools/pm_board.py --tree
```

轴的取值在 `docs/pm/board.json` 的 `axes` 里，`--check` 会拦住拼错的取值。
低比特那几种 dtype（int8 / mxfp8 / mxfp4 / hif8 / hif4）属于 gated epic G3，词汇表里留着但还没有任务。
