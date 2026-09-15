# 多 agent 协作协议

> 制定于 2026-09-15。**接任务的 agent 必读**，读完再发 `ACK`。怎么启动 PM / agent 会话、怎么找到 PM，
> 见 `START.md`；两类会话的系统提示在 `prompts/`。本协议只**补充**
> `AGENTS.md`，任何一条都不放宽它；两者冲突时以 `AGENTS.md` 为准并报 `RISK contradicts-handoff`。

## 1. 角色与事实源

| 角色 | 是谁 | 写什么 |
|---|---|---|
| PM | 会话名 `ascend-fla-dev-management-team`（用 `ListAgents` 查到后 `SendMessage`） | `docs/pm/**`、`docs/matrix/*.json`、`docs/handoff.md`、main 分支 |
| agent | 任何发 `APPLY` 并拿到 `ASSIGN` 的会话 | **只写**自己任务的 `write_set`，只在自己的 worktree / 分支上 |
| 用户 | 仓库所有者 | 决定所有 `gated` 项（定位变更、kernel 批次、资源） |

- `docs/pm/board.json`：任务唯一事实源，`python tools/pm_board.py --check` 校验、`--render` 看进度。
- `docs/pm/tasks/<ID>.md`：每个任务一份**自包含**的规格。规格没写的，不做；觉得该做，发 `RISK`。
- 机器清单只在 git-ignored 的 `machine_specs.md`。协议消息里也**不写**主机名/IP/账号，用"a2 卡 4"这类代号。

## 2. 任务状态

```
gated ──(用户决定/条件满足)──> open ──ASSIGN──> assigned ──ACK──> in_progress ──DONE──> review ──accept──> done
                                                  │                │    ▲                  │
                                                  │                ▼    │                  └──rework──> rework ──DONE──> review
                                                  │             blocked ┘
                                                  └────────(超时/放弃)────> open（分支保留）          任意 ──> cancelled
```

`gated` 的 `gate` 字段写明在等什么：`user-decision` / `machines:a2` / `kernel-batch-approval` /
`resource` / `wave:<id>`。

## 3. 消息格式

每条消息第一行固定为：

```
[FLA-PM] <TYPE> <task-id 或 -> from=<发送者会话名>
```

之后是 `key: value` 行（值可多行，缩进两格续行）。**PM 只按第一行路由**，格式不对的消息会被退回。

### 3.1 APPLY（agent → PM）

```
[FLA-PM] APPLY - from=agent-foo
socs: a2            # 能用的 SoC，逗号分隔；纯主机侧写 none
ascriptor: yes      # 有没有 ascriptor workspace（按 agent/compatibility.json 选好修订）
fla: yes            # 装没装 fla（oracle）
python_env: torch 2.x / pytest 可用
```

### 3.2 ASSIGN / NO_TASK（PM → agent）

```
[FLA-PM] ASSIGN A2-01 from=ascend-fla-dev-management-team
spec: docs/pm/tasks/A2-01.md
branch: task/A2-01
worktree: git worktree add .claude/worktrees/A2-01 -b task/A2-01 main   # 然后 EnterWorktree(path=".claude/worktrees/A2-01")
lease: none                         # 或 soc=a2 card=4 cache=<任务远端工作区>/A2-01/cache
timebox_h: 12
report_every_min: 60
```

`NO_TASK` 带 `reason:` 与 `retry_after_min:`。

### 3.3 ACK（agent → PM，收到 ASSIGN 后 15 分钟内）

```
[FLA-PM] ACK A2-01 from=agent-foo
read: AGENTS.md, docs/handoff.md, docs/pm/PROTOCOL.md, docs/pm/tasks/A2-01.md
plan: |
  1. ...（≤5 行）
eta_h: 10
```

### 3.4 STATUS（每个里程碑、每次真机运行前后、至少每 60 分钟一次）

```
[FLA-PM] STATUS A2-01 from=agent-foo
done: 定位到缺陷记录；a2 profile 参数已查实
next: 写最小复现
numbers: |
  （目前为止的数字，带形状/dtype；没有就写 none）
evidence: tmp/A2-01/notes.md
blockers: none
```

### 3.5 RISK（发现即发，不攒）

```
[FLA-PM] RISK A2-01 from=agent-foo
class: kernel-change-needed
summary: ...
evidence: tmp/A2-01/xxx.log:120-160
proposal: ...
```

| class | 什么时候发 | PM 的处置 |
|---|---|---|
| `silent-wrong-result` | 发现有限、量级正常但内容错的输出 | **立刻上报用户**；按 §7 装闸 |
| `kernel-change-needed` | 要改 kernel 源码才能继续 | 记进 `gaps.json`（`requires_kernel_change`），**agent 不改**，等批次 |
| `positioning-change` | 做法会改变仓库定位（接 fla dispatch、静默兜底…） | 立刻上报用户 |
| `shared-machine` | 卡 Health 异常、卡上有别人的进程、需要动别人的东西 | 立刻上报用户；换卡 |
| `contradicts-handoff` | 实测与 `handoff.md` / `plan.md` / 本协议的结论矛盾 | PM 复核后改文档 |
| `soc-assumption` | 某条事实只在一个 SoC 上成立（A5 的数被当成 A2 的预期，或反过来） | 影响计划就上报用户 |
| `timebox-slip` | 预计超时 >25% | 超 50% 上报用户 |
| `write-set-expansion` | 需要改写集外的文件 | PM 查冲突后批或拒，**批之前不许改** |

### 3.6 BLOCKED / LEASE_REQ / LEASE_RELEASE

```
[FLA-PM] BLOCKED A2-03 from=agent-foo
waiting_for: A2-01 的命中表
since: 2026-09-15T14:00
keep_lease: no
```

```
[FLA-PM] LEASE_REQ A2-12 from=agent-foo
soc: a2
health: npu-smi info 里候选卡的 Health 与 proc-mem 原样贴出
```

`BLOCKED` 期间租约最多保留 2 小时。

### 3.7 DONE（agent → PM）

```
[FLA-PM] DONE A2-01 from=agent-foo
branch: task/A2-01
commits: abc1234 def5678
acceptance: |
  - [x] <规格里的每一条验收，逐条抄过来，后面跟数字>
host_tests: pytest tests/ -q → N passed / M skipped（原样）
device: soc=a2 cann=<版本> opp_pkgs=<kernel 目录列表> card=<代号>   # 纯主机侧写 none
numbers: |
  （rel-L2 / max_abs_diff / 耗时，每个数带形状、dtype、warmup/repeat、是否 synchronize）
evidence: tmp/A2-01/（未过滤原始日志）
matrix_delta: tmp/A2-01/matrix_delta.json   # 建议的 gaps/ops 改动；没有写 none
handoff_notes: |
  下一个人需要知道的，包括被实测纠正的判断
cleanup: 远端自己起的后台循环已清（pgrep -af "unti[l] ! pgrep" 无输出）；租约已释放
```

### 3.8 REVIEW / CLOSE（PM → agent）

`REVIEW` 带 `verdict: accept | rework` 与编号清单；`CLOSE` 表示已合入、租约收回，agent 可以再 `APPLY`。

## 4. agent 纪律

1. **分支与写集**：只在 `task/<ID>` 上工作，**不提交 main、不 push**。只改 `write_set` 里的路径。
2. **共享文档不碰**：`docs/matrix/*.json`、`docs/handoff.md`、`docs/pm/**` 由 PM 写。你的改动建议放
   `tmp/<ID>/matrix_delta.json` 与 DONE 的 `handoff_notes`。例外只在规格里显式写明。
3. **ascriptor 仓只读**（AGENTS §3）。新 SoC 的单元建在本仓 `kernels/projects/<soc>/`。
4. **kernel 源码只在 PM 建的 kernel 批次任务里改**（AGENTS §6.5）。其余任务里发现要改，发 `RISK kernel-change-needed`。
5. **共享机器**（AGENTS §5）：只用租到的卡；每次运行前看 Health 与 proc-mem；绝不 kill/修改他人进程；
   装包只进自己的 venv；`ASCEND_FLA_CACHE` 与 `TMPDIR` 放在本任务自己的工作区；传完对 `md5sum`；
   等长任务盯日志文件不盯进程；一个算子名一个进程一份 build。
6. **证据**：读未过滤的原始日志（重定向到文件再 `nl -ba`）；报数字不报"OK"；形状参数按组合的边界点扫，不只扫单轴。
7. **SoC 事实不跨 SoC 搬**：门控跨度、`block_dim` 上限、物理核数、算子包覆盖、(C,HV) 边界行为 —— 每个 SoC 重测。
   把一个 SoC 的数当另一个的预期，是 `soc-assumption` 风险。
8. **提交卫生**（AGENTS §10）：commit message 不写主机/账号/路径；`tmp/` 不提交。

## 5. PM 承诺

- 收到 `APPLY` 后派单：`python tools/pm_board.py --next --socs … [--ascriptor] [--fla]` 的第一个候选，
  再核卡是否空闲。看板改动先提交 main 再发 `ASSIGN`。
- 收到 `DONE` 后：在任务分支上重跑 `pytest tests/ -q`、`tools/gen_matrix.py --check`、`tools/pm_board.py --check`
  与规格里的主机侧验证命令；逐条核验收数字与原始日志；`git merge --no-ff`；应用 matrix delta 后重跑校验；
  更新 `handoff.md`、关对应缺口。**验收数字缺一条就 rework**。
- 心跳：进行中任务 90 分钟无 `STATUS` 发 ping，再 30 分钟无回应则收回（状态回 `open`、租约收回、分支保留）。
- 向用户报告：每次派单/完成/返工一句话；§3.5 标注"立刻上报"的风险即时报告。

## 6. 已纠正的外部说法（Gemini 规划文档里的，不要照做）

2026-09-14 收到的三份 Gemini 规划（EasyASC / PyPTO / PyPTO Pro）只作需求来源，其中代码是**伪代码**。
下面这些说法与实测或数学不符：

| 说法 | 实际 |
|---|---|
| `(I+A)⁻¹ ≈ I − A + A²` 就是严格下三角求逆 | 64×64 严格下三角只有 `A⁶⁴=0`，精确逆要一直展开到 `A⁶³`；二阶截断是**近似**，不能当 `solve_tril` |
| 门控 `exp2(cumsum g)` 用 fp16 算 | 真实初始化的跨度约 94（上界 100.8），fp32 下都会在 87.3 下溢；要用本仓 stable 单元的中点锚点 |
| 最终 state 存 fp16 / HiF4 | state 是跨 chunk 累积量，本仓约定 FP32（`state-dtype-bf16` 缺口） |
| Prefill 算力不足时回退 BF16/INT8 | 静默降级，违反 AGENTS §7 |
| bf16 下逐元素 `rtol=1e-3` 判对错 | AGENTS §6：正确性一律 fp32 判定，报相对 L2 / max_abs_diff |
| 测 T=63/100/127 应当通过 | 在 `no-tail-path` 解决前，这些长度必须**报错**，测试断言的是报错 |
| 2 PFLOPS、128B 突发对齐、UB 256KB 等硬件数字 | 未经核实，不能当验收阈值 |
| 910B 上 GDN 的精度/HF32 结论可直接沿用 | 同级 `fla_infer` 工作区的 A2 数字方法可借，阈值与结论必须重测 |
