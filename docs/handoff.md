# 交接：下一个会话从哪里接

> 写于 2026-09-11，会话状态更新至 2026-09-19。**本文件只记"接手需要知道的"**；权威状态在
> `docs/matrix/*.json`（`gen_matrix.py` 校验），纪律在 `AGENTS.md`，分期在 `docs/plan.md`。
> 三者冲突时以 json 为准，并顺手修本文件。

## -1. 2026-09-19 PM 交接（本节最新，PM 换人先读这个）

**这是一次 PM 账号本身的交接，不是算子进度交接。** 下面按"现在手上有什么、有什么坑、
下一步该干什么"排列，配合 `docs/pm/board.json`（权威）和 `python tools/pm_board.py --render` 看。

### 正在飞的任务（现在没有；A5K-02、PK-02、PK-03 都已于 2026-09-19 合入）

> **2026-09-19T13:05Z 更新：用户给了常设合入授权（D-PM-33）——取代下面所有「合入需要用户明确授权」的说法。**
> 原话：「PR只要有通过真机验证，你审查后确认正确性，高质量就可以合入，不需要我授权。有争议时，再向我提问」。PM 的读法（用户可纠正）：
> 只适用于带通过的真机验证的 PR；审查流程不变；证据只有自述、缺原始日志/标识、真机项被省略的按「有争议」问用户；
> 没有真机验证的 PR（纯文档/主机侧）、新账号首次合入、写集扩大、预算/闸/域变化、申领人异议、A2 上的结论、仓库定位/gated 放行/kernel 批次批准，仍先问用户。
> 合入后在 main 上复跑全量并记进看板，再事后一句话告诉用户；自动模式的分类器若仍拦，不绕过。完整条文见 `docs/pm/prompts/pm.md`「合入授权」。
>
> **2026-09-19T12:55Z 更新：用户放行 GDA-03 与 PK-04（D-PM-32，原话「两个都放行」），已派单。**下面 12:45Z 那段里"全部 gated、等用户放行"对这两个已过时，其余仍然有效。
> - `GDA-03`（#91）→ session `gdn-series-20260919T115559Z-29cb31d7`，分支 `task/GDA-03`，24h；`PK-04`（#92）→ session `01a0b7ce-…`，分支 `task/PK-04`，16h。
>   同账号不同 session，并发规则放行；两边写集不相交。
> - **PK-04 的申领人放行前已在私有工作区做了补偿累加修复和一轮真机**（自述，没有 PR）：bd4 的 28 个全链案例、3 个域内对抗种子误差约 6e-6，其余 bd 与同卡性能在跑。
>   放行**不追认**这些进度：DONE 时逐项核对原始证据——原失败样本要保留、修复要有「先定位」的证据、修复后的全部验收要在同一份修复后的版本上重跑。补偿累加修法本身仍是候选。
> - **PM 对 D-PM-32 的读法**：用户没有单独回答「PK-04 里的 kernel 修复算不算需要用户批准的 kernel 批次」，PM 读作放行覆盖 PK-04 规格的全部（含「先定位再在已批写集内修」），
>   已在汇报里向用户说明，用户不同意可以撤回。
> - 仍 gated：`PK-05`、`GDA-04`、`PK-06`、`GDA-05`（前置任务完成后 PM 细化，放行是用户的事）。合入授权见上面 D-PM-33。
> - 下一轮起：两个 assignee 的 ACK 要在 24h 内到（GDA-03 与 PK-04 都是）；PK-04 的 PR 到了先做「来源」核对（它此前是 gated 的外部需求），再走审查流程。
>
> **2026-09-19T12:45Z 更新：新增六个 gated 任务，等用户放行；PK-04 有一条待用户知悉的 silent-wrong-result 线索。**
> - **来源**：#89（GDN/PGDN 后续）与 #68 下的 REQUEST，均已分诊采纳（`triage:accepted`）。**#90 是 #68 需求的重复，已关闭。**
>   来源是外部需求，所以全部 `gated`，放行只能是用户：
>   - `GDA-03`（#91，分组 GDN chunk 反向，完整规格）→ `PK-05`（#93，PGDN 反向，骨架）→ `GDA-04`（#94，GDN decode，骨架）→
>     `PK-06`（#95，PGDN decode，骨架）→ `GDA-05`（#96，GDN/PGDN 性能，骨架）；`PK-04`（#92，PKDA 前向 A5 真机验收，完整规格，独立于上面这条链）。
>   - 序列严格按 D-PM-22 的"backward → decode → 性能"，**不含** GDN-2、Qwen3-Next、A2、KDA，也不是被否决的 #80。申领人转述的用户指示
>     （"继续申请任务，直到所有 GDN 任务都完成""需要完成真机测试"）PM 无法从评论核实，已交用户确认。
>   - 两个申领人（session `gdn-series-20260919T115559Z-29cb31d7` 与 `01a0b7ce-…`）的 APPLY 都留在 #28/#92 上，放行后 PM 直接按它们 ASSIGN
>     （GDA-03 `timebox_h` 24、PK-04 16），先把 `origin.approved_by_user` 置 true、状态置 `open`。
>   - 后四个 GDN/PGDN 任务只是骨架：细化与放行在前置任务完成后由 PM 做，之前不动手。
> - **PK-04 的线索（申领人自述，PM 只复算了机理）**：原 PKDA 前向在域内对抗性门控（每 chunk 首项 g≈−155、其余 −1e-5，跨度 <155）的**真机**输出有限
>   但超 ≤1e-4 预算（o/S 相对 L2 1.30e-4/1.82e-4），首个偏差在 `prepare` 的门控前缀和。PM 用 numpy 复算：纯 FP32 顺序累加该 pattern 对高精度累加的
>   最大偏差恰为 0.000335693359375，与申领人报的一致，机理成立；kernel 实际这样累加、o/S 超预算，PM 没有复现。RISK 来自未派单的申领人，poll 标"不采信"，
>   所以没有当协议消息处理；PM 独立把这个模式写进了 PK-04 的验收与陷阱，**修法（补偿累加）没有批准**。矩阵里本来就没有 PKDA 的真机结论，没有条目需要更正；
>   `gaps.json` 等 PK-04 的 DONE 见到原始证据后再记（届时是 `requires_kernel_change`）。**待查（假设，没验证）**：KDA 链路的门控前缀和是否同构——
>   `AGENTS.md` §6 的"整条链同类算式列一遍"适用；KDA 的 155.97 精度扫描用的不是这种"一大后小"模式。
> - **PM 侧的工具坑**：`poll` 会跳过 PM 账号发的非协议评论，而 agent 与 PM 共用 `pm_github_login`，所以 **agent 的纯文字进展评论对 poll 不可见**
>   （这次的可行性预研、工具链失败、PK-04 的 RISK 之外的进展都在纯文字评论里）。每轮对有活跃申领人的 issue 匿名读一次评论
>   （`https://api.github.com/repos/<owner>/<repo>/issues/<N>/comments`），别只信 poll。
>
> **2026-09-19T11:45Z 更新：PK-02、PK-03 已合入，当前没有在飞任务。**（用户 D-PM-31 授权；合并后 `main` 全量 510 passed / 15 skipped。）
> - **PK-03**（PGDN 前向，PR #87，合并提交 `1f4da05`）：A5 真机验收完成——60 个原生 case、bd1↔bd2 逐位相同，全网格 FP32
>   相对 L2 ≤ 2.5e-6（预算 1e-4）。同卡测量的基线是'GDN 方程成本基线'（不是 GDA-02 实现的性能），PGDN 慢 3.4%–4.5%。
> - **PK-02**（PKDA 前向，PR #88，合并提交 `8b78c09`）：主机侧验收。FP32 五阶段、q/k 原样消费（调用方负责归一化）、前向跨度
>   门控保持 155；跨度 4096 的抵消误差 S=1.34e-4 超预算，保留在公开域之外。无真机声称。
> - **两处规格被实测纠正过**：PKDA 的 naive **不做** q/k 归一化，PGDN 的 naive **做** `F.normalize`，两个算子的归一化语义
>   不同、结论不共用（`pkda_semantics.md` §7）；PK-02 的 oracle 由'Triton chunk'改为'独立 CPU 参考'。
> - **（已被 13:05Z 的 D-PM-33 取代）合入都需要用户明确授权**：自动模式会拦未经人工确认的合入（A5K-02、PK-03 各被拦过一次）。
> - **角色边界**（用户 D-PM-28）：机器与卡不归 PM 管，PM 只管 tasks 的进度与提交代码的质量。
> - **下一步候选**：D-PM-22 排期是 GDN 前向 → GQA → PGDN 前向 → backward → decode → 性能，前三步已完成。PGDN/PKDA 的
>   backward、decode、性能都还没有任务号，要用户排。另有 `A2-02`、`A2-44` 已解门（D-PM-29），等申领。
> - **矩阵欠账**：`docs/matrix/ops.json` 的 `linear_attention_ops` 里没有 GDN/GDN-2/PGDN/PKDA 的 chunk 前向单元条目
>   （前面几批 agent 任务的 `matrix_delta` 都是 none）。`ops.json` 与 `gen_matrix.py` 也是仓主轨道高频改的文件，
>   等 A2-06（矩阵按 SoC 分维）协调好一并补，不零散加。
>
> 下面各条是合入前的记录。

> **2026-09-19T09:05Z 更新：A5K-02 已合入**（PR #83，合并提交 `2589942`，合入的是审过的头 `3e3670f`；用户 D-PM-27 授权）。
> 公开 KDA 调度的 stable 路径现在选修复后的 recurrent（奇数 C 多头不再静默写错 `o`），upstream 路径对奇数 C 且
> `B*HV>block_dim` 提前报错；派生 `inverse_mm` 只做了被批准的事（与 pin 上游逐行 diff：仅入口改名 + 两块 L0C 双槽改单槽）。
> 缺口表已更新：`c1-multihead-o-corrupt` 保持 P0，写明默认路径已修与仍未解决的边界；新增
> `kda-bwd-inverse-mm-mutex-over-budget`（P1，进 kernel 修复队列，指上游源码）。**下面 A5K-02 的条目是合入前的记录。**
> **`A2-02` 与 `A2-44` 已由用户解门**（D-PM-29，2026-09-19），转 `open`。两个任务的写集彼此也有重叠
> （`ops/kda/chunk.py`、`layers/kda.py`），先 APPLY 者先做，后者大概率再 gate 一轮；此前被 NO_TASK 的申领人
> 已在 #31、#86 下被通知重新 APPLY。
>
> **PM 的边界（用户 2026-09-19，D-PM-28）**：机器与卡不归 PM 管理，agent 自己管理；PM 只管 tasks 的进度与提交代码的质量。
> 此前 PM 给 PK-03 设的'占用证据必须由用户/管理员提供'关卡已撤销，`shared-machine` 类 RISK 只记录、不上报。

- **`A5K-02`**（issue #81，草稿 PR #83，`blocked`，assignee `limjiunnbin`，
  session `01a0ae7a-d1c0-7f02-b040-be977ec47393`）：D-PM-24 批的 `inverse_mm.py` 派生
  kernel（mutex 34→32 单槽化）已实现，公开前向选择/原始对照/全部 9 个 backward kernel
  在 `block_dim=4` 下 vendor-compile 通过，bd1-3 编译中。**唯一的阻塞是设备空闲窗口**——
  同机另有任务在跑，另一张卡不健康。写集含 `ascend_fla/ops/kda/chunk.py`、
  `chunk_bwd.py`、`kda_fwd_stable/**`、`kda_bwd_stable/**`（含新加的 `inverse_mm.py` 与
  `contract.json`）。最近一次 STATUS：2026-09-18T09:31。
  **2026-09-19T04:08Z 更新**：换成 successor session `01a0b7ce-3daf-7550-a7bf-31c67eb91ca3`
  （自述受用户指示接手，PM 已记录，session 自称、不核实）；PR #83 head 仍 `888862b`（草稿）。
  **阻塞原因变了**：不再是"等设备空闲窗口"，而是**设备 SSH 认证失败**，
  远端构建、设备健康/占用/锁状态都未知；它没有动设备。机器归 assignee 管，PM 不代为恢复访问。
- **`PK-03`**（issue #69）：2026-09-19T02:41Z 撤回，04:16Z 由新 session 重新申领，**用户确认（D-PM-25）后
  已重新 ASSIGN**（session `pgdn-forward-20260919T041214Z-7b59b3a6`，与 A5K-02 同账号不同 session，并发规则放行）。
  `PK-03.md` 已重写为"主机侧、窄范围"的可派状态：ABI 与现有 GDN 门控逐项一致；oracle A=fla naive.py +
  B=独立 CPU 参考；**真机验收是必须项**（用户 2026-09-19，D-PM-26），通过后可合入；机器与卡归 assignee 自己管理（D-PM-28），PM 不设占用证据关卡。原 session `gdn-pgdn-forward` 的申领作废。已 ACK，草稿 PR #87 已开、主机侧验收就绪，等真机。
  （PGDN chunk 前向 + ATK 预条件，D-PM-22 排期解锁。）

### 这次会话新合入的两个 PR（供快速对账，细节见 §0.x 各节和 gaps.json）

- **PR #84（`GDA-02`，commit `8303167`）**：GDN chunk 前向补 GQA/GVA 分组
  （`HV % H == 0`），`HV==H` 退化路径逐位不变（代码层面可证明，见 `kernels/pipeline.py`
  的 `expand_inputs`）。合入后解锁了 `PK-03`。
- **PR #85（`A2-40`，commit `1c1abe3`）**：整网融合算子（`causal_conv1d` /
  `fused_rms_norm_gated` / `qk_l2norm_gate` / `packed_projection`）的清单/ABI/验收设计，
  纯文档，没动 kernel。发现两处本仓层与 fla 的静默分叉：
  - **D1**：`ascend_fla/layers/kda.py` 用 `F.normalize(x, eps=1e-6)`（对范数取 max），
    fla 是 `x/sqrt(Σx²+eps)`（分母整体加 eps）。‖x‖=1e-4 时两者差出一个数量级
    （fla 给 0.0995，本仓给 1.0）。近零行才触发，Kimi-Linear 的正常输入大概率碰不到。
  - **D3**：`use_short_conv=False` 时本仓不做 silu，fla 做。Kimi-Linear 用短卷积碰不到，
    但构造参数接受这个开关。
  - 两条都记进了 `gaps.json`（`kda-layer-l2norm-eps-formula-diverges` P2、
    `kda-layer-missing-silu-when-no-short-conv` P3），修复任务是新建的 **`A2-44`**
    （纯主机侧，不动 kernel）。

### 写集重叠：一个新问题类别，这次连续撞上两次

`A5K-02` 在飞期间，`ascend_fla/ops/kda/chunk.py`（以及 `chunk_bwd.py`、
`fused_recurrent.py`）实际上被它"占用"着——它要在这些文件里切换公开调度选中的
recurrent kernel。这次会话里，**`A2-44`**（issue #86）和 **`A2-02`**（issue #31，
`platform.py` 的 SoC 显式化重构）先后 APPLY，写集都覆盖到这几个文件，都被我按
write-set-overlap 转 `gated` 并加了 `deps: [..., A5K-02]`，NO_TASK 回复已发（issue
#86、#31 各有一条说明）。

**给下一个 PM 的提醒**：`A5K-02` 合入之后，`A2-02` 和 `A2-44` 会同时解锁——**它们两个
自己的写集也有重叠**（都碰 `ascend_fla/layers/kda.py` 和 `ops/kda/**` 的部分文件），
到时候谁先 APPLY 谁先做，第二个大概率还要再 gate 一轮，除非先手动协调一下顺序或者
缩小其中一个的写集。这不是这次会话能提前处理的（两个任务都还没人在写），只是先说清楚
免得到时候被当成新问题重新分析一遍。

### 并发规则实战：这次真的被用上了，而且是级联的

> **2026-09-19 更新**：`PK-03` 在本节写完 11 分钟后被 WITHDRAW，04:16Z 又由新 session 重新申领并重新 ASSIGN
> （D-PM-25）。`limjiunnbin` 现在仍同时占 A5K-02（session `01a0b7ce-…`）与 PK-03（session `pgdn-forward-…`）
> 两个任务，按 session 不同放行；规则本身没变。

`docs/pm/PROTOCOL.md` §3.1 的 `session` 并发规则（同账号、不同 `session`，可以同时持有
两个任务）这次被 `limjiunnbin` 账号连续用了两轮：先是 GDA-01/A5K-01/A5K-02 这条链
（`session: 01a0ae7a-...`），现在又叠加了 `PK-03` 的 `session: gdn-pgdn-forward`。
**目前 `limjiunnbin` 账号同时占着 A5K-02（blocked）和 PK-03（assigned）两个任务**，
这是按规则允许的形式条件（两边都填了 session 且不同），**不代表 PM 核实过真的是两个人
在操作**——如果后续两边的 STATUS 内容有明显不一致或互相矛盾的地方，值得留意。

### A2-40 留下的问题：已决，不放宽（用户 2026-09-19，D-PM-30）

`A2-40` 文中提出：`A2-41` 涉及的四个融合算子对象都是纯向量运算、不经过 cube，
技术上不受 split-K FP32 cube 缺陷影响，但 `AGENTS.md` §2 写的是"A2-11 之前 A2 上任何
算子结论都不算数"（不分 cube/vector）。**要不要对纯向量单元放宽这条总闸，是规则层面的
决定。用户 2026-09-19 已决定不放宽**（D-PM-30）：`AGENTS.md` §2 不改，`A2-41` 的 `machines:a2` 闸与对
A2-11 的依赖不变。

### 心跳 / 轮询状态（交接时刻）

两个在飞任务的最近 STATUS/ASSIGN 时间都在 2026-09-18 上午，交接时刻约 2026-09-19
凌晨，年龄约 16~17 小时，远未到 48h PING 阈值。`tools/pm_github.py poll` 的游标已推进
到最新（2026-09-19T01:23:11Z），没有积压事件。下一个 PM 直接按
`docs/pm/prompts/pm.md` 的轮询流程接手即可，不需要补跑历史事件。

> **2026-09-19 更正**：上面"不需要补跑历史事件"**只在同一台机器上成立**。下一节是换到全新机器时踩到的两个坑。

### 换机器接手 PM 的两个坑（2026-09-19，换到一台全新机器时实测）；另见上面 12:45Z 更新里的 poll 盲区

- **poll 游标是机器本地的。** 它存在 `tmp/pm/github_state.json`（git-ignored），换机器就没有；
  没有时 `since` 默认 `1970-01-01`，第一次 `poll` 会把整个仓库历史当成新事件吐出来。
  做法：从旧机器拷过来；拷不到就按上一次交接记下的时间戳手工播种 `{"since": "<ISO 时间>"}`，
  **宁早勿晚**（偏早只会重放少量已处理的事件，偏晚会静默丢事件）。播种前先匿名读一次公开 API
  （`issues/comments?since=…`）看回放量。本次按 01:23:11Z 播种，只回放 2 个事件：
  PK-03 的 WITHDRAW（上一个 PM 已处理；因为 assignee 已被清空，工具报"不采信"，是**重放的假警报**，
  不是冒充）和 A5K-02 草稿 PR #83（无警告、无需动作）。
- **`gh` 要够新，而且 `!` 通道没有 TTY。** 工具用了 `gh auth token -u <账号>`，系统包管理器里的老版本
  没有它也没有 `gh auth switch`（Ubuntu jammy 的 apt 包是 2.4.0，两样都缺，也只能存一个账号）；
  官方 v2.101.0 两样都有。装官方 release 的二进制并校验 sha256。`!gh auth login` 会报
  `--web or --with-token required when not running interactively`，要用 `--web`（设备码）或
  `--with-token < <token 文件>`。做法与登录顺序见 `docs/pm/START.md` §2 第 3、4 步。

### 在装了 torch_npu 的共享机器上复跑 PR 测试（2026-09-19 实测，都是踩过的）

- **不清环境直接 `pytest tests/` 会碰设备。** 机器装了 torch_npu/CANN 且有 NPU 设备可见：`import torch` 会自动加载
  torch_npu，`tests/conftest.py` 的 session fixture 会去编译全部 kernel，NPU 测试也可能起跑，而设备是共享的。
  做法：设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，再在 `PYTHONPATH` 最前放一个只写 `raise ImportError(...)` 的
  `torch_npu/__init__.py` 桩，并**先验证 `torch.npu.is_available()` 为 False** 才跑。只 `env -i` 清环境不够：
  torch_npu 会半初始化，报 `TORCH_LIBRARY` 重复注册并段错误。
- **测试要 pin 版的 ascriptor。** 同级目录的 `ascriptor-kernels` 是 kernels 的 pin（用 `git rev-parse` 核对，工作区要干净）；
  同级的 `ascriptor`（library）修订常常不是 pin，用 `git archive <pin> | tar -x` 从 git 对象导出 pin 版到临时目录，
  别动别人的检出。用 `ASCRIPTOR_KDA_FWD` / `ASCRIPTOR_KDA_BWD` 指向 kernels 的 `projects/a5/kda_fwd` / `kda_bwd`，
  `PYTHONPATH` 指向导出的 library。gitcode 要凭据，匿名取不到。
- **有 10 个测试需要 pin 版 FLA 源码**（cache adapter 4、GDN 3、GDN-2 1、KDA gating 2），本机没有 fla 就会 skip；
  对账时要写清 `passed + 这 10 个 = 申领人报的 passed`。全量套件在这个环境里约 70 秒，卡住不是测试慢，是没排除 torch_npu。

## 0. 2026-09-15 更新：SoC 顺序变更 + 多 agent 协作（先读这节）

**先说清楚：现在有两条并行轨道，本节讲的是 agent 那条。**

| 轨道 | 谁在做 | 内容 | 看哪里 |
|---|---|---|---|
| 仓主轨道 | 仓主本人 | A5 上的 GDN-2（decode 链路、gdn2-1.3B 整网、四个 a5/gdn2_* 单元） | 下面 §1.1~§1.3 |
| agent 轨道 | 外部 agent，PM 派单 | A2 (910B) → A3 (910C) → A5，首波 KDA + Kimi-Linear | `docs/pm/board.json` 与 GitHub issue |

两条轨道共用仓库，但互不指挥：仓主那条不受看板的 SoC 顺序约束，看板也不因为仓主在 A5 上推进就改波次
（2026-09-16 用户确认，看板 D-PM-10）。边界靠 `board.json` 的 `reserved_paths` 划：那些路径 agent 不碰，
`pm_board.py --check` 会拦住写集与之相交的任务。

**agent 轨道的 SoC 顺序是 A2 (910B) → A3 (910C) → A5**（2026-09-14 定的，见 `AGENTS.md` §2 与
`docs/plan.md` 的「按 SoC 分波次」）。所以下面 §2 那三个候选属于 agent 轨道的 A5 波次，要等 W-A3 退出；
**它们和仓主正在 A5 上做的 GDN-2 是两回事，别混起来读。**

工作改成多 agent 推进：PM 维护 `docs/pm/board.json`，agent 按 `docs/pm/PROTOCOL.md` 申领任务并汇报，
看进度跑 `python tools/pm_board.py --render`。

现在能派的是 W0 的主机侧任务，A2 机器还没到位。最靠前的是 A2-01（split-K FP32 cube 缺陷定性，A2 的总闸）。
A2-06（矩阵按 SoC 分维）已改成 gated：它要重构的
`docs/matrix/*.json` 与 `tools/gen_matrix.py` 正是仓主轨道在高频改的文件，等协调好再开。

### A2-04 已完成（2026-09-17，PR #60 合入 8ac32e5）—— 两条要记住的

**① `c1-multihead-o-corrupt` 的根因找到了，而且暴露面比我们记的宽。**
不是漏了某次 DEvent/Mutex 配对，是 `Aqk` 的 L1 交接**两信用配固定槽**：
`aqk_l1_valid = DEvent(..., preset=True)` 给两个信用，槽却是 `aqk_slot = Var(c_idx % 2)`（按 chunk 号取）。
一个头最后一个 chunk 的槽是 `(C-1)%2`、下一个头第一个是 `0` —— **当且仅当 C 为奇数时相撞**。
与 ascriptor `library/docs/defects/M10-076-mutex-credits-and-handoff-slots.md` 同型，
但那条的修复在 `autosync` 里，而本 kernel 是手写同步、不过 autosync，**拦不住**。

**闸现在是漏的**：`_check_single_chunk_heads` 只拦 `C==1 and B*HV > block_dim`，
而奇数 C≥3 结构上同样暴露（kimi 的 `C = T/64`，T=192、320 都在里面）。
收紧成 `C % 2 == 1` 会拒掉真机上跑过的形状，按 §1 要显式决策 —— **待所有者放行，先别改代码**。
细节与证据在 `gaps.json` 的该条目，`docs/pm/deltas/A2-04.json` 是原始提案。

**② "C≥2 全对是因为 chunk 循环第二遍补上了同步"是错的，已从 `AGENTS.md` §6 更正。**
那句话是从"C=1 坏、C=2 好"的**症状规律**倒推出来的，写进文档两期没人怀疑，
既把根因指错方向（去找"缺的那次同步"），又把安全域说宽了。
**症状规律不只会误导修法，还会伪装成结论写进文档。**

证据等级：全部是 a5 管线模型值（pipesim + 调度回放），**不是真机数**。
但 `gaps.json` 里 2026-09-11 那张真机失效表被 **12/12 格逐头复现**（PM 在权威 workspace 上独立复算过），
且预测的 kimi `o` 相对 L2 1.060 / `final_state` 2.458e-03 与真机记录的 1.06 / 2.46e-03 吻合。

### GD2-01 已完成（2026-09-17，PR #64 合入 0397aa1）—— 例外任务，两条要记住的

**这条不属于上面 §0 的 A2→A3→A5 波次**，是用户对 D-PM-12（"GDN-2 不在看板造任务"）批准的一次
显式例外（D-PM-16，issue #61）：agent 轨道做 GDN-2 **chunk 前向**（forward-only），
写集限定新文件（`ascend_fla/ops/gdn2_chunk_fwd.py` 是 `ops/gdn2/` 的兄弟文件，不进那个包），
`reserved_paths` 一个字节没碰。

**① 门控跨度的新记录：真实 95B checkpoint 测出 1520.91455，超过 2026-09-14 记的 1461.214。**
`gaps.json` 的 `gdn2-chunk-gate-range` 已更新。选定的表示是逐 chunk FP32 对数前缀 + 直接算因果差分
（不是 KDA 那种端点/中点乘积），全部指数项恒非正，`exp(1461/2)` 这类值从未被求值——但**精度仍要单独测**，
不能因为"有限"就假设"够准"（这条是从 KDA 那次教训延续下来的）。

**② 真机上定位到一个 WY 依赖循环的编译器/硬件缺陷**（950PR_9589 V100, CANN 9.2.0）：VF 循环的内层
边界如果直接绑定外层归纳变量会漏掉最后一次贡献；改成固定 64 次迭代 + 显式 `j < i` 判断绕过。
**compiler-versus-silicon 的归因没有做**，只确认了这个具体写法在这台机器上不安全。

**GD2-02（真机数 + 真实整网验证 + 性能三明治）现在可派**（deps 已满足）。GD2-01 的原
assignee 后续又申请并做了 GD2-03（性能优化，用户批准 D-PM-17，正在进行）。

### PKDA / PGDN 权威来源确立（2026-09-17，PK-01 完成）—— 见 `docs/research/pkda_semantics.md`

用户提供本地 fla checkout（`/Users/limjiunnbin/work/flash-linear-attention`，0.6.0，
HEAD `e52dbc0e`——**与 GD2-01 用的 FLA pin 是同一个 commit**）。之前 PK-01.md 写的
"fla 0.5.2 没有 pkda/pgdn，本仓无 oracle"**只在 0.5.2 成立**，本仓 `.venv` 装的正是 0.5.2，
但上游 0.6.0（2026-06 合入，PR fla-org/flash-linear-attention#950）已经有完整实现。

**两条要记住的**：

① **PKDA 是 KDA 加一层 ATK 预条件，不是新算子族**——ABI 除三个 ATK 专属张量外，与本仓
`ascend_fla/ops/kda/chunk.py` 逐项对应，可直接复用 `kda_fwd_stable`/`kda_bwd_stable`。
PGDN 同理是 GDN 加同一层预条件，但需要的 GQA/GVA 分组正是 GDN 自己缺的 `gdn-no-gqa`——
**排在 GDN 自身 ABI 之后**，不是并列关系。

② **Gemini 文档"`k̃=k/(p+eps)` 会下溢"那条担心，看了真实算法后不成立。** ATK 的预条件乘子
是 `M = exp(-log(x)·s)`，`s = r/(1+|r|)` 恒落在 `(-1,1)`——这是这个函数形式本身的硬数学界，
不依赖状态 `A` 的大小，所以不存在"除法下溢/溢出"这回事。仍要测的是 `log(A+eps)` 在本仓真实
量级下的精度，以及 `k_precond` 替代 `k` 之后 KDA 已有的门控跨度判定要不要重新过一遍。

排上 **PK-02**（PKDA，open，主机侧，`kda_fwd_stable`/`bwd_stable` 派生单元）与
**PK-03**（PGDN，gated: prereq-gdn-abi）。两者都**没有已发布的预训练权重**——训练仓
（<https://github.com/ntumm120/preconditioned-deltanet>）只有从零训练的配置（340M/1B），
端到端验证到不了真实 checkpoint 的 logits/cache 一级，只能到双 oracle（fla naive + fla
自己已验证过的 Triton chunk）加真实规模形状。

### 2026-09-17/18：GD2-02~04、ascriptor 库缺陷、并发规则、GDA-01、A5K-01

**GD2-01 之后又做了两轮性能优化**：GD2-03（scan/output 逐 channel 重复算的标量指数改成
逐行向量化，D-PM-17）、GD2-04（WY 行对复用，D-PM-19）都已合入。两轮都遵守"先 profile
再改、精度判据不放松"，性能数字见各自的 PR（#67、#74）与 `contract.json`。**GD2-02**
（真机数 + 真实整网验证 + 性能三明治）还没人接，deps 已满足，随时可派。

**发现一个 ascriptor 库本身的缺陷**（`gaps.json` 的 `ascriptor-gm-transfer-two-slice-row-gap`，
P0）：`passes/device_lower.py` 的 `gm_transfer` 函数，双切片分支（`len(sliced)==2`）算行间距
时只看第二个切片维之后的形状，**漏了夹在两个切片维中间的标量索引维**——4 维 GM 张量
`[B,T,H,D]` 里 B/H 标量索引、T/D 切片这种模式会静默算错行间距，产出有限但错误的值。
这是库层面的问题，不是本仓能改的范围（`AGENTS.md` §3）；本仓侧的绕法是显式传
`gm_to_ub_pad(..., 正确的 gap)`，GDA-01 已经这么做并验证过。**扫过已合入的 kernel**，
找到 19 处同类模式，全部只搬一行、不受影响——但扫描范围有限（只覆盖直接 GM 参数 +
`<<=` 形式），不代表证明了整仓没有别处踩中。

**多 agent 并发规则改了**（`tools/pm_board.py` 的 `check()`）：原来"一个 agent 同时只接
一个任务"是按 GitHub 账号判定的，现在改成按 `(账号, assignee_caps.session)` 判定——
两个任务都填 `session` 且不同，同一个账号可以并发持有多个任务；只填一边、都不填、或
两边相同，仍按老规矩当同一个 agent。见 `docs/pm/PROTOCOL.md` §3.1。这条规则被
GDA-01 与 A5K-01 实际用上了——同一个 `limjiunnbin` 账号靠不同 `session` 同时接了两个任务。

**GDA-01 已合入**（非 GQA 的 GDN `gated_delta_rule` chunk 前向，A5，D-PM-20 窄范围例外，
PR #77）：真机 10/10 case 通过、三轮同步三明治 T1024 1.07x/T4096 1.08x。过程中原设备
出了真的硬件故障（RAS bus error `0x80f78009`），处置完全正确——不复位不 kill、只读诊断、
切到同机健康设备重跑、旧结果不混入最终声明。

**A5K-01 已合入**（KDA `Aqk` L1 交接缺陷修复，PR #78）：把 A2-04 诊断、pipesim 验证过的
槽轮转修法（`Var(((pair_idx - pair_begin) * C + c_idx) % 2)`）落到 `kda_fwd_stable` 的
派生 `recurrent.py` 里。真机 336/336 case（C=1..6 × 7 组 H/HV × 零/随机 state × bd=1..4，
含 Kimi 真实形状 H=HV=32）通过，PM 独立复算了不依赖真机的 12 格 pipesim 回归，逐字一致。
**没解决的边界，写进了 `gaps.json`**：修复只在这个独立单元里，`ascend_fla/ops/kda/chunk.py`
的公开调度**仍然选择原始（有缺陷的）recurrent**——那个文件不在 A5K-01 写集内，接线是
一个需要显式决定的独立步骤。**在接线完成之前，公开 API 的用户仍然会撞上这个缺陷。**

**当前可派（`tools/pm_board.py --next`）**：A2-01（P0，split-K 缺陷定性）、A2-02、A2-08、
**GD2-02**（真机验证 GD2-01~04 的累积成果）、**PK-02**（PKDA，KDA 派生单元）。
`GDA-01`/`A5K-01` 都是波次例外，做完之后想再申领同类工作要走"新 REQUEST → 用户批准 →
新任务号"，不能假设已批准的例外自动延伸到下一轮（GD2-03/04 就是这么一路批下来的先例）。

任务 issue 发过两轮。第一轮用新建的 bot 账号 `ascend-fla-pm-bot` 建了 26 个（#2~#27），两分钟内建完，
触发 GitHub 反滥用过滤：匿名访问这些 issue 和该账号主页全是 404，登录态却一切正常 —— 所以看着像发成功了。
那 26 个已全部关闭（`not_planned`）。第二轮改用 `limjiunnbin` 重发，申领入口是 **#28**，任务 issue 从 #29 起，
实测匿名可见。

由此定下两个身份（看板 D-PM-9）：`pm_github_login` 发 issue 正文与协议评论，必须匿名可见；
`label_github_login` 做打标签、关 issue 这类要 write 权限的操作。没有 write 权限的账号建 issue 时
**标签会被静默丢掉**，gh 不报错。

同一个坑还暴露了 `pm_github.py` 的三处问题，都已修并有反例测试：REST 列表被过滤后 sync 会重建重复 issue
（改走 GraphQL 并加了"记过编号却不见了就停下"的闸，84add96）；换账号后旧的已关闭 issue 会被当成"已存在"而复活；
建 issue 没有节流（现在默认 `--pace 15`）。

三份 Gemini 规划文档只当需求来源看。它们与实测矛盾的说法，列在 `docs/pm/PROTOCOL.md` §6。

## 1. 一句话现状

第一期（runtime 桥 + `kda_fwd` 接线 + 基线）与第二期（`kda_bwd` + autograd + KDA layer）
**已完成**；第三期进行中 —— **KDA 的 decode 已自写 kernel 并接进 layer，功能可用、性能不可用**。
Kimi/Qwen 模型注入与矩阵 CI 未开始；用户指定的 GDN-2 1.3B 已完成 torch_npu 基线、
架构拆分和 packed-inference 优化；除通用 **Ascriptor CCE `gdn2_fused_recurrent`** 外，又新增
模型专用 **`gdn2_fused_decode`**，把 raw gates、递推和 output norm 合成一 launch。Cast 与
整网时延显著下降；静态双槽又让合法的两段成对流水在真机生效。完整 stateful NPU Graph
与模型专用 short-conv/cache kernel 已把纯模型 decode 压到稳定约 **3.109ms/token**；随后
原生 fused RMSNorm + host-packed SwiGLU W1/W2 又压到 **2.654~2.664ms/token**（375.37~376.86
token/s），真实生成 decode loop 为 **340.62~354.33 calls/s**。低内存默认设备时间约86%已是
MatMul/GEMV。gamma-fold混合MLP虽通过kernel级真机闸（边界1.077x），却在真实95B递推中
把cache误差累积到1.523e-2，已拒绝；恢复三个BF16边界的保守版已过完整数值闸与同卡Graph
夹心A/B，显式opt-in达到 **2.587~2.590ms/token、386.13~386.52 token/s**，但增加约1.030GB
派生权重，故低内存默认仍不变。最新profile中权重流占90.18%、混合kernel的AIC MTE2平均
91.88%；后续大步收益必须触及权重字节。长
prefill/训练的 chunk fwd+bwd 尚未开始。

| 指标 | 值 |
|---|---|
| 真机全量测试 | **63 passed**（A5 / CANN 9.2.0，约 346s；本轮新 kernel 尚未并入这次全量） |
| 主机侧测试 | 上一轮全量 66 passed / 5 skipped；本轮 GDN-2/模块定向 **39 passed** |
| GDN-2 定向真机 | 通用 recurrent contract 5/5；fused decode、packed short-conv与mixed MLP的sim/pipesim/aclnn、真实整网及profiler passed；低内存默认375.37~376.86 token/s，mixed显式opt-in 386.13~386.52 token/s |
| 缺口 | P0 **1** · P1 17 · P2 9 · 已解决 11 · 共 38 |
| kernel 统一修复队列 | **24 项**（`gaps.json` 的 `summary.kernel_fix_queue`） |
| 工作区 | GDN-2 model-first 改动未提交；先看 `git status` |

### 1.1 GDN-2 1.3B model-first（2026-09-14 新增）

用户指定先跑 `LLM-OS-Models/gdn2-1.3B-fineweb-edu-100b` 的 95B checkpoint，再从真实
调用链定位算子。已完成：

- 只下载 `checkpoint-95B-model-ckpt.pth`，固定 HF revision `86327354…`；文件
  17,401,727,659 bytes，SHA-256 `4ac729c6…f6d`，目标 model mount 上又复核一次。
  HF 的 90B 与 95B 文件当前指向同一个 LFS 对象，但 archive 内部是
  `step_count=22667` / `iter_num=725344`，按 4,194,304 tokens/step 得 **95.072B**，
  所以 95B 文件本身身份正确，90B 名字才是发布侧异常。
- `ascend_fla/models/gdn2.py`：完整 LitGPT-compatible torch/torch_npu 基线，399 个
  state_dict 键与发布方 Apache runtime 逐键逐 shape 相同；`meta + mmap + assign`
  严格加载大 checkpoint，普通权重转 BF16、`A_log/dt_bias` 保持 FP32。
- 真机实权重：input `[1,2]` → logits `[1,2,32000]` BF16，全有限、范围
  `[-11.75,11.0]`；18 层 state `[1,16,128,128]` FP32；一次前向 vs 逐 token cache
  **max_abs_diff=0 / relL2=0**。NPU allocated 2,960,865,280 bytes。load 2.947s、首次
  forward 0.210s 只是单次冒烟，**不是性能基线**。原始记录在
  `tmp/gdn2-95b/gdn2-full-run.json`（git-ignored）。
- `ascend_fla/models/generation.py` + `benchmarks/generate_gdn2_text.py`：一次 prefill 后逐 token
  传递 recurrent/conv cache；采样只把最后一行 logits 搬回 CPU，避免把 torch_npu 随机算子
  混进“模型能跑”的结论。TinyLlama v1.1 tokenizer 固定到 revision `ff3c701f…`，验证了
  32K vocab、自动 BOS=1、EOS=2。真实 95B 权重的 greedy 输出为：
  `The capital of France is Paris. It is the capital of the country of France. It is the largest city in France. It is the most populous city in the European Union.`
  共生成 32 token，最终代码单次 0.629s / 50.9 token/s；这是跑通证据，**不是性能基线**。原始记录在
  `tmp/gdn2-95b/text-greedy.json` 与 `text-greedy.raw.log`（git-ignored）。
- 最关键的算子结论：**GDN-2 不是现有 GDN 的换名。** 目标需要 channel-wise
  `g[B,T,H,K]`、erase `b[B,T,H,K]`、独立 write `w[B,T,H,V]`；现有
  `a5.gdn_fwd/bwd` 只有 per-token 标量 `beta/g`，不能接。已记
  `gdn2-abi-not-gdn`，缺失算子记为 `gdn2_fused_recurrent` 与
  `gdn2_chunk_fwd_bwd`。
- 边界 profile（真实 95B 权重、warmup 后）：全模型 decode 不含 logits D2H 为 10.086ms/token；
  layer-0 block 568.5µs，其中 mixer 433.7µs。最大的四段是 recurrence+qk norm 140.8µs、
  三路 short conv 105.9µs、output norm-gate 58.9µs、g/b/w 激活 57.0µs。把同形状权重按行
  拼接后，qkvbw projection 在 T=1 快 2.54x、qkv short conv 快 4.27x；实测输出与分开执行
  逐位相同，但这只作诊断，正式闸仍是 relative-L2。
  证据：`tmp/gdn2-95b/profile-boundaries.json` 与原始日志（git-ignored）。
- 数值域 stress（真实权重、合法随机 token id、T=4096）：单 token `-g` 最大 60.926，
  64-token 累计跨度最大 **1461.214**，远超 KDA stable 的已验证域。recurrent 每步只算
  `exp(g_t)`，不受跨 token 配对量程影响；chunk 必须单独设计，不能复制 KDA。已记
  `gdn2-chunk-gate-range`，证据：`tmp/gdn2-95b/gate-domains-t4096.json`。

### 1.2 GDN-2 架构步骤 1/2（2026-09-14 已完成）

用户审定三项决策后完成“先重构、再打包”，**没有开始 gate/recurrent/chunk 自编译算子**：

- `reference/gdn2.py` 独立出 fp32-state recurrence oracle；`layers/gdn2.py` 接管
  `GatedDeltaNet2`、typed `GDN2LayerCache` 与显式 `core_backend`。这一阶段 backend 只有
  `torch`；后续 CCE 接线见 §1.3。未知名字始终报错，不静默 fallback。canonical 模型仍是
  399 个 LitGPT key。
- `models/gdn2_checkpoint.py` 只负责 checkpoint wrapper 前缀规范化；模型 loader 先以
  canonical 399-key strict load，再可选原地转成 `projection_layout="packed-inference"`。
  packed 布局仍有同样 1,450,096,416 个参数；加入 W1/W2 host packing 后内部 state_dict 是
  255 项（此前只打包 mixer 时是273项）；转换逐层消费
  原模块，不常驻两份约 0.85GB 投影。打包在 host 完成后才 H2D，packed reserved memory
  实测约 3.14GB；packed 求导会显式报错。
- q/k/v/b/w 五个同输入、无 bias 的投影合成一个 `F.linear`；q/k/v depthwise short-conv
  与三份 cache 沿 channel 合成一次调用，packed cache 是 `[B,6144,4]` BF16。f/output-gate
  的低秩首层只在 B=T=1 时合并；T>1 改用 packed weight 的两个连续 row slice 分开算。
- **这一条件分支来自一次真机纠错**：孤立 microbench 只验证 packed 第一层输出，最初把它的
  strided split 直接喂第二层，真实权重 T=6 的 output-gate 出现 max_abs_diff=7.8125e-3，
  最终 logits relL2=4.57e-3。改为 T>1 分开算后，真实 95B 的 BF16 T=6/T=64 与 FP32 T=6
  均通过数值闸（预算分别 1e-2 / 1e-5，实测 logits/cache relL2=0）。本次恰好逐位相同只是
  额外观测，**不是 CPU/NPU 或 packed/canonical 的契约**。另有 tiny FP32 模型直接对 CPU
  oracle，logits/recurrent/conv cache 最大 relative-L2=5.740e-4，以 1e-3 预算通过。
- warm decode 用两种加载顺序、warmup=100/iters=200 复测：canonical 10.184~11.625ms，
  packed 9.212~9.822ms，加速 **1.106x~1.184x**（按保守下界看 1.106x）。同一 32-token
  greedy 文本与 token ids 在最终 host-packed loader 上再次通过；单次 0.502s 只作 cold-ish
  端到端 smoke，不当稳态基线。复现：
  `benchmarks/verify_gdn2_packed.py`；证据在 `tmp/gdn2-95b/verify-packed-steady-*.json`、
  `verify-packed-host-pack-{bf16,fp32,t64-bf16}.json`、
  `packed-cache-chain.json`、`text-greedy-packed-host-pack-final.json` 与原始日志。

### 1.3 GDN-2 步骤 1~4：CCE recurrent 已接整网（2026-09-14 已完成）

- **ABI**：公开语义冻结为 token-major q/k/v/g/b/w `[B,T,H,128]` + FP32
  state `[B,H,128,128]`；g/b/w 已激活。kernel 内做 q/k FP32 L2 norm、`128^-0.5`
  scale、channel-wise decay/erase/write、输出与 final state。首版只收 B=1、T=1~16、
  H∈{1,16}、K=V=128、block_dim=8；CCE 只支持 inference，不满足就报错。
- **kernel + harness**：`kernels/projects/a5/gdn2_fused_recurrent/` 是本仓自有
  `ascriptor.kernel-unit/1`，CCE 静态检查 0 error / 0 warning / 220 surface ops，emit
  266 lowered ops；reference 5/5、sim 3/3 scoped、pipesim 2/2 scoped。Ascriptor 原生
  unit runner 用 `requested_backend=cce + launcher=aclnn` 跑完五个 contract case，o 最大
  relative-L2 `4.169e-6`、final_state 最大 `9.108e-8`（预算均 `1e-4`）。
- **接线**：`ascend_fla/ops/gdn2/fused_recurrent.py` 经已有进程内 ctypes→aclnn 桥直接吃
  NPU tensor；本轮没有再包一层 pybind，因为现有桥已经覆盖零拷贝 data_ptr、stream 与
  workspace。`models/gdn2.py` 在任何整网算子执行前 `prepare()`，`layers/gdn2.py::_run_core`
  显式分派 torch/cce，绝不 fallback。另修了 runtime cache：source signature 现在覆盖入口
  同文件的 `@vf/@func` helper；否则 helper 改动会错误复用旧 binary。
- **公开算子**：BF16 real-prompt case 的 o relative-L2=0、state=`6.167e-8`；一次 T=4
  与四次 T=1 串 state 的 o/state relative-L2=0。B1/T1/H16 完整 public call（含 cast +
  aclnn）`88.165us`，torch_npu 组合 `148.400us`，1.683x（warmup10/iters100/同步）。
- **整网**：真实 95B checkpoint、packed-inference 下，BF16 prompt/step logits
  relative-L2=`3.260e-3/3.186e-3`，最差 recurrent cache=`9.154e-3`（预算1e-2）；FP32
  为 `1.391e-5/2.065e-5`、cache=`1.785e-4`（预算1e-3）。argmax 与 cache offset 相同，
  bitwise 只作诊断。BF16 两种加载顺序都过；整网 decode CCE `8.182~9.435ms` 对 torch_npu
  `8.378~9.460ms`，目前基本持平，说明下一轮要 profile 周边而不是从单算子倍数外推。
  CCE core 下 canonical↔packed 也通过：prompt/step logits 与两类 cache relative-L2 均为0；
  同轮 decode `10.920ms` 对 `9.843ms`，packed 为1.109x。
- **吐字**：同一真实 checkpoint 由 CCE backend greedy 生成 32 token：
  `The capital of France is Paris. It is the largest city in France and the capital of the country. Paris is located in the north-eastern part of France. Paris is the`
  单次 0.550s / 58.18 token/s，只作 smoke。证据统一在 `tmp/gdn2-recurrent/`（git-ignored）。

### 1.4 GDN-2 CCE decode profiler（2026-09-15）

`benchmarks/profile_gdn2_backend.py` 临时给生产 forward 注入 `record_function` 范围，普通推理
路径没有 profiling 分支或额外开销。真实 95B / BF16 packed / B1T1H16 用 torch_npu
profiler Level1 + PipeUtilization，两个 backend 各自独立进程，1 warmup + 3 active：

- torch_npu `6043.4us/token / 1523 kernels` → CCE `5108.8us / 1163 kernels`，设备侧少
  `934.6us`（15.5%）和 360 次发射（23.6%）。几乎全部来自 core：`1189.3→282.2us`。
  直接 ctypes→aclnn 的 custom kernel 不挂 Torch 关系树，CCE core 的 282.2 = 范围内 casts
  186.2 + kernel CSV 的 custom 96.0。
- **CCE 剩余占比**：MLP 26.36%、short-conv+cache 16.85%、packed projection 16.20%、
  两次 block RMSNorm 12.04%、output norm-gate 8.65%、gate prep 7.38%、core 含 casts 5.52%。
  recurrent kernel 本体只有 5.336us/层、1.88% 整网设备时间，下一轮不能再先抠它的算术。
- MatMul 占 46.2%（145 calls），代表性 BF16 GEMV 的 MTE2 ratio≈94%、MAC≈11%，是权重
  读取主导。Cast 占11.1%，**CCE 与 torch 都是309次/token**：FP32 CCE ABI 108、output
  norm-gate 72、37个 model RMSNorm 111、gate prep 18，源码与 trace 可以逐项对齐。
  其中91次是 packed CCE inference 下严格冗余的：55个不变 norm weight 每 token 重转FP32，
  以及18层 recurrent output FP32→BF16 后立刻被 output norm 升回FP32的36次往返。剩余218次
  表达的是FP32规约/state精度语义，但可以搬进融合kernel内部，并不要求作为独立Cast算子存在；
  **没有一项是为了CPU/NPU逐位相同**，本仓本来就不以bitwise为契约。
- packed short-conv 每层为 cache 做两次 cat+两次 slice；T=1 时 window 与 new_cache 可共用，
  按当前各半计可先去掉约234.3us/token。`-exp(A_log).repeat_interleave` 在 inference 中是常量，
  却每 token 重算，exp+repeat+neg 共125.9us/54 launches，也应先预计算。
- **wall time 仍只能判持平**：warmup100/iters200 的反序 paired check，torch-first 为
  9.721 vs 9.523ms（1.021x），CCE-first 为9.318 vs9.562ms（0.975x）。两次跨过1，否决了
  单次 pre-profiler 1.246x 的加速说法。设备收益被上千次周边发射、casts 与 aclnn host
  固定成本吞掉。完整分析在 `tmp/gdn2-profiler/analysis.md`，新缺口为
  `gdn2-decode-fragmentation`。

### 1.5 GDN-2 Cast 清理与 BF16 fused decode（2026-09-15）

- packed inference 在 host 已按目标 dtype 量化权重；55 个 norm weight 以前每 token 再
  `.float()` 一次。现在把**同一份已量化值**只扩宽一次到 FP32，不改变旧数值边界；同时缓存
  channel-expanded `-exp(A_log)`。中间 profile 的 Cast **309→254**、565.925→466.875us/token。
- 新单元 `a5.gdn2_fused_decode` 固定真实 B1/T1/H16/K=V=128 BF16：直接吃
  q/k/v/f_raw/b_raw/w_raw/output_gate，在 CCE 内做 stable softplus、sigmoid、q/k norm、FP32
  recurrence/state、output RMSNorm+swish，最后只做一次 BF16 RNE store。旧 FP32 recurrent ABI
  不改，继续服务 prompt、FP32 与非精确匹配配置；不回退 Torch。包级 `prepare()` 在首次算子
  解析前把两者都编好。
- 固定 Ascriptor 0.1.0：静态双槽版本 check 0 error/0 warning/404 surface ops，CCE emit 415 lowered ops；
  reference/sim/pipesim 3/3，无 hazard/deadlock；aclnn real_decode 的 o/state relative-L2
  `8.574e-11/3.044e-7`。真实95B step logits `3.436e-3`、最差 recurrent cache `9.521e-3`，
  均在 BF16 `1e-2` 预算内；32-token greedy token ids 与旧 CCE 路径相同且文本可读。
- 最终 Cast **309→74/token**，Cast 时间565.925→147.355us；设备kernel总时长
  5108.848→4023.590us/token，launch 1163→640。warmup100/iters200 的反序 paired：
  torch-first 8.087→4.900ms（1.650x），CCE-first 8.726→6.188ms（1.410x），这次两边都
  稳定快于 torch。
- **静态双槽流水已通过**：动态下标 DBuff 的 `auto_sync` 把 slot n 与 n-1 保守判成同 root，
  插入 read1→compute0 与 write0→compute1 假依赖；所以旧 DB64 四 pipe 和为98.17%、真机反而
  4.804us。改成两个静态命名的64-row槽后，pipesim 从10463降到9613 cycles，同核多pipe
  重叠3.68%→11.44%；两次真机各54样本 mean=4.259/4.236us，夹心整state对照
  4.733us，合并mean/median=4.247/4.256us，快1.114x/1.108x。AIV
  Vector/Scalar/MTE2/MTE3 合并平均34.73/17.60/37.52/21.70%，四pipe和111.55%（对照86.84%），
  证明是真重叠。aclnn与真实95B整网数值均过原闸；两轮profile与整网任务的宿主监控中，
  物理0活跃0次、任务PID只在7。随后为补 canonical unit runner 证据做的一次**fresh CCE build**
  在实际 `test_aclnnop` 上7之前，捕捉到物理0上一个 `python`/0MB瞬时条目；时间与CANN `opc`
  编译进程吻合，但进程退出后无法补做PID namespace硬归属。实际 `test_aclnnop` 只在7，且0未
  留驻。严格“只碰7”期间后续应复用已构建产物，不再 fresh build；不能把编译器探针说成0次。
- **单pipe约80%仍未过，但原因已钉死**：第一遍必须对全部128行完成erase规约，才能形成delta；
  因而MTE2+第一遍Vector与第二遍Vector+MTE3被全局join分成两段，当前边界不可能形成三pipe
  长稳态。**不要减核或加冗余工作刷利用率。** 要继续满足该门槛只能扩大到
  RMSNorm+projection / output+o_proj 等含有用BF16 GEMV的混合边界；若优先整网时延，则做
  qkv short-conv+cache。证据在 `tmp/gdn2-cast-fusion/`。

### 1.6 GDN-2 stateful NPU Graph capture（2026-09-15）

- torch_npu 2.10 的 `NPUGraph` 可捕获完整18层BF16 packed CCE decode；本仓 ctypes→aclnn
  custom op 也能进图。单 custom kernel 从 eager 约69.5us降到 replay约14.0us；整网固定
  prompt-cache 从6735.3降到3834.9us/token（148.47→260.76 token/s，1.756x）。
- 真正可递推版本在图尾把18层新 recurrent/conv cache 拷回固定输入地址；4个不同token连续
  replay后，logits、两类cache均与eager**逐位相同**。排除早期探针后，v3~v5同轮总吞吐
  eager为5911.9~6323.7us/token，graph稳定在3908.2~3911.9us（255.63~255.87 token/s，
  1.512x~1.617x）；每token同步的串行graph为3915.6~3926.0us（254.71~255.39 token/s，
  对eager加速1.554x~1.736x），所以收益不是靠预提交200步伪造的。
- eager提交阶段约5.67~6.78ms/token，graph提交只有2.8~7.4us/token；graph的最终drain约
  3.90ms/token，说明host已不再配速，瓶颈确实移到设备侧。stateful固定地址代价是每token
  回写19,759,104 bytes，但只比fixed graph多约75us。清空eager allocator历史后复测：捕获
  一次约13.6ms，常驻allocated/reserved增量约39.6/130.0MB。
- 当前边界仍是**纯模型**：不含logits D2H、CPU argmax与下一个token H2D；`offset` 是Python
  元数据，由调用方递增，不进图。复现与证据：`benchmarks/bench_gdn2_graph_capture.py`、
  `tmp/gdn2-graph-capture/{fixed-v1,stateful-v3,stateful-v4,stateful-v5}.json`（git-ignored）。
- 本轮复用现有CCE产物，没有fresh build。接受的stateful-v3运行期间宿主采样50次：物理0
  活跃0次、物理7活跃16次；运行前后两卡均无本任务留驻。
- **生成接线与清洁配对验收已完成**：`models/gdn2_graph.py` 提供窄契约
  `GDN2NPUGraphDecodeRunner`，固定token/cache地址，图尾回写两类cache，并支持用同形状prompt
  cache做`reset()`而无需重捕获；`generate_tokens(..., decode_backend="npu-graph")` 与文本脚本已
  接入，默认仍是eager且不满足B1/T1/BF16/packed/CCE/NPU时明确报错。真实95B的64-token greedy
  做了两个独立进程、正反顺序配对；四次完整token ids逐项相同，文本连贯。graph decode loop
  （含logits D2H、CPU argmax、下一token H2D）稳定244.21/244.79 token/s，即4094.8/4085.1us；
  eager为173.02/148.84 token/s，即5779.6/6718.8us，两个顺序的graph加速1.411x/1.645x。
  64-token整请求（含冷prefill和每请求graph setup、不含模型load）为graph 99.11/104.23 token/s，
  eager 96.73/91.60，两个顺序仍都快1.025x/1.138x；graph setup为54.5~58.5ms，冷prefill
  274~327ms，所以短请求的总收益被一次性成本稀释。四个接受窗口的物理0活跃均为0，物理7
  各只有本任务一个PID。证据：`tmp/gdn2-graph-generation/{graph-greedy64-v2,
  eager-greedy64-v1,eager-greedy64-v2,graph-greedy64-v3}.json`（git-ignored）。另有一轮功能通过但
  与外部8-rank任务初始化竞争，已明确隔离为`rejected-contended-*`，性能数不得引用。

### 1.7 GDN-2 short-conv/cache（2026-09-15，已完成）

- **先收了无风险的重复工作**：packed T=1 decode 里卷积窗口本来就等于更新后的 cache，现复用
  同一连续 tensor，逐层去掉一次 Cat+Slice。设备 launch 640→604/token，short-conv+cache
  约855.95→642.39us/token，device kernel总时长4018.69→3800.17us/token。stateful graph
  从旧三轮3908.18~3911.91降到新两轮3693.01~3698.00us/token，即270.42~270.78 token/s，
  保守吞吐提升5.68%；logits与两类cache relative-L2均为0。真实64-token greedy的graph
  decode loop为256.47 calls/s，eager为147.36，token ids逐项相同。
- **CCE kernel 已实现并接线**：新单元 `a5.gdn2_short_conv_decode` 固定
  B1/T1/D6144/W4/BF16/block_dim=8，一个launch完成四tap FP32累加、SiLU与cache更新。16个
  向量参与者各拥有384通道；NDDMA把cache/weight转tap-major UB，cache用b16→b32两级
  interleave按channel-major写回，无核间同步。公开wrapper用零拷贝flatten view走现有
  ctypes→aclnn桥；packed CCE层只在精确匹配且已有cache时选它，prompt/T>1保留原路径。
- **当前证据**：静态0 error/0 warning/203 surface ops，CCE emit231；reference/sim/pipesim/
  canonical aclnn均3/3，y/cache exact，cache满足bitwise规则，pipesim无event imbalance/hazard/
  deadlock且2215 cycles。公开runtime双顺序与真实95B双backend顺序均过：step logits
  relative-L2=4.044e-3，最差recurrent/conv cache=9.313e-3/7.270e-3（预算1e-2），argmax与
  offset一致。定向主机测试37 passed。
- **设备收益已profile**：54个kernel样本mean/median=3.374/3.363us；每token把18 Conv2D+
  18 Cat+18 Slice+18 short-conv Swish换成18个custom call，launch604→550，device kernel总时长
  3800.169→3197.959us，跨采集暂估省602.210us（15.85%）；随后clean同卡反序Graph A/B
  实测省584.34us/token（15.82%），与profile估计吻合。但公开eager wrapper为40.99~42.00us，仍慢于
  torch_npu组合31.79~31.82us；小边界的aclnn/分配固定成本只有graph才能隐藏。
- **单pipe约80%在这个边界排不开**：bulk版MTE2/Vector/MTE3/Scalar为
  57.59/8.84/9.18/22.13%，四项和97.74%，基本串行。三路静态128槽与256+128槽都保持数值、
  event、hazard正确，但pipesim从2215退到3745/2819 cycles——仅三个vector tile，额外
  NDDMA/MTE3 setup大于可隐藏工作，故未上板并已回退。若要求单pipe接近饱和，下一架构边界必须
  吞入packed projection的BF16 GEMV；不能减核或造冗余工作刷比率。
- **clean Graph A/B 已完成**：物理5上按 CCE→Torch→CCE 反序、独立进程实测，CCE三轮
  3107.60~3110.82us/token（321.46~321.79 token/s，均值3108.96us/321.65 token/s），旧Torch
  short-conv同卡为3693.29us/270.76 token/s；因此少584.34us/token（-15.82%），吞吐+18.80%。
  逐token同步的CCE为3113.58~3120.34us/token（320.48~321.17 token/s），排除预提交假收益；
  三轮logits/recurrent/conv cache对eager relative-L2均为0。真实95B 64-token greedy的decode
  loop为299.93 calls/s（含logits D2H、CPU argmax、下一token H2D），整请求110.10 generated
  token/s（含冷prefill与每请求Graph setup、不含模型load），token ids与既有结果逐项相同。
  三个接受窗口合计297次宿主采样，物理0活跃0次、物理5只出现五个本任务PID；起初物理7被
  外部任务抢占的一轮仍隔离为`rejected-contended-*`，不得引用。证据：
  `tmp/gdn2-shortconv-kernel/current-performance/`（git-ignored）。

### 1.8 GDN-2 原生 RMSNorm + SwiGLU W1/W2 packing（2026-09-15）

- **不是二选一，先做低风险组合并用2×2消融归因**：packed NPU inference 的37个block/final
  RMSNorm改用已安装的单算子`npu_rms_norm`，weight保留checkpoint已量化的BF16值；18层W1/W2
  在host按行打包且不保留重复参数，T=1合成一个`[2304→12416]` GEMV，prompt/T>1仍从同一
  packed weight的两个连续row slice执行原尺寸GEMV。`o_norm`不变，仍以FP32 weight直入
  fused-decode CCE ABI。packed参数数仍为1,450,096,416，state_dict 273→255项。
- **2×2 clean Graph消融**：旧norm+分离W1/W2为3116.09us/token（320.91 token/s）；只换原生
  RMSNorm两轮均值2699.68us（370.41 token/s）；只pack W1/W2两轮均值3063.15us（326.46
  token/s）；两者一起两轮2653.51/2664.05us（376.86/375.37 token/s，均值2658.78us/
  376.11 token/s）。组合延迟-14.68%、吞吐+17.20%；按对称两因素归因，RMSNorm贡献89.74%，
  W1/W2贡献10.26%。逐token同步为2662.98/2663.80us附近，结论不是预提交假收益。
- **数值与吐字**：真实95B canonical↔packed两个加载顺序得到同一组数值：prompt/step logits
  relative-L2=1.252e-3/3.937e-3，最差prompt/step recurrent cache=1.391e-3/7.239e-3，step
  conv cache=4.282e-3，均过BF16 1e-2预算；argmax/offset一致。两次64-token greedy token ids
  与此前逐项相同，decode loop为340.62/354.33 calls/s；整请求100.50/116.45 generated token/s，
  差异来自冷prefill 402.6/320.4ms，不作为稳态回退。
- **新profile**：Cast 74→0/token，kernel rows 550→273/token，device total
  3197.959→2665.141us/token。剩余MatMul（含Addmm）2290.572us/85.95%，原生RMSNorm仅
  97.798us/3.67%，SiLU+Mul 79.280us/2.97%。packed W1/W2 GEMV本身733.967us/token，MTE2
  平均94.7%；若再写`RMSNorm2+W1/W2+SiLU×Mul`混合CCE kernel，可明确隐藏/删除的非GEMV
  上限约122.7us/token（4.6%）。它是BF16 ceiling实验，不再是单点最大收益；更大的下一步要
  减少权重字节。证据：`tmp/gdn2-rmsnorm-swiglu/`（git-ignored）。

### 1.9 GDN-2 混合 RMSNorm2/W1W2/SwiGLU（2026-09-15，v1拒绝/v2已完成）

- **先过权重流闸**：自写paired BF16 GEMV的N=64在pipesim中因194个tile均摊到28核，综合
  MTE2利用率86.42%→98.81%、模型周期672129→593354；但真机反而是44.345us/layer，慢于
  N=128的41.691us，证明翻倍的DMA/matmul固定开销超过尾部均衡收益。N=128最初写16个物理M行，
  比vendor慢6.47%；改为只发布唯一有效行后，两种顺序均值**39.272 vs 39.216us/layer**，只慢
  0.142%，且与vendor逐位相同，raw GEMV闸通过。pipesim看不见这2.4us/layer的FIX固定成本，
  速度排序必须以真机为准。
- **v1完整混合边界**：新unit `a5.gdn2_norm2_w12_swiglu` 固定B=T=1/D2304/I6208/block_dim28。
  host把不变BF16 gamma折进W1/W2；cube维持N=128饱和权重流，vector subblock并行算inverse RMS，
  通过两槽CvMutex取FP32 accumulator后完成缩放、SiLU与multiply，只把BF16 hidden写GM，W3不并。
  static 0 error/0 warning；reference 2/2、functional sim与pipesim random通过，无event imbalance/
  hazard/deadlock，临界MTE2约99.75%；正式op名经本仓ctypes→aclnn bridge真机输出对独立oracle
  relative-L2=3.444e-7、max_abs=1.526e-5。
- **v1 kernel级双顺序A/B**：18份不同权重下custom-first为40.537 vs43.609us/layer，vendor-first为
  40.416 vs43.593；均值**40.477 vs43.601us/layer，边界1.0772x**，折算18层省56.238us/token。
  synthetic相对当前组合路径为5.231e-3；真实checkpoint逐层gamma-fold探针的W3输出最坏
  relative-L2为4.984e-3（FP32 accumulator变体），当时看似都在BF16 1e-2预算内。
- **v1被整网否决**：prompt因仍走vendor而逐位相同；decode第1步logits/recurrent为
  3.254e-3/4.799e-3，但第2~4步recurrent累积为1.216e-2/1.367e-2/1.523e-2，final conv为
  1.191e-2，均越过1e-2；四步argmax虽相同也不能放松闸。该轮后段有外部进程进入同卡，所有
  timing一并拒绝，但确定性数值失败与竞争无关。这再次证明单层预算不能外推递推整网。
- **v2恢复舍入边界**：取消gamma fold；两个vector subblock各自算同一个RMS规约并通过V→C
  发布1152维`x*inv_rms*gamma` BF16；GEMV先落BF16 UB，SiLU再落BF16 UB，最后BF16 multiply。
  独立边界拆分证明不fold但保留FP32 epilogue仍有3.892e-3单层误差，而恢复两处BF16物化后与
  当前CPU公式逐位相同。正式unit的reference 2/2、random sim/pipesim均通过，relative-L2
  3.416e-7、max_abs1.526e-5、无event imbalance/hazard/deadlock，672723 cycles。真实95B的prompt
  与四个串行decode step在logits/recurrent/conv cache上relative-L2均为0，argmax与offset一致。
- **显式接入、不改默认**：`from_checkpoint(..., mlp_backend="cce")` 才启用；prefill仍走原
  `npu_rms_norm + packed W12`，v2 decode走保守kernel。为保持prompt控制和checkpoint ABI，当前
  每层另存一个non-persistent原始paired布局，18层增加1,029,832,704 bytes；去重是整网通过后的
  独立布局决策。`bench_gdn2_graph_capture.py`已有`--mlp-boundary-path cce`，另有
  `verify_gdn2_mixed_mlp.py`做4-token递推精度。
- **v2整网与profile均已通过**：同卡CCE→vendor→CCE的stateful Graph为2587.192/2656.712/
  2589.831us/token，CCE均值2588.511us、386.323 token/s；相对中间控制延迟-2.567%、吞吐
  +2.635%。逐token同步均值2597.205us，同样快2.371%。Level-1 profile为2576.490us/token、
  219 rows/token，相比vendor profile的2665.141us/273 rows少88.651us和54 rows；混合kernel
  54样本mean/median=42.339/42.168us，AIC MTE2平均/中位91.880%/92.150%，cube utilization
  平均87.914%，因此“至少一条有用pipe接近饱和”已满足。当前native GEMV+混合W12合计
  2323.348us/token，占90.175%。profile运行只在物理7出现目标容器PID；物理0始终是另一容器
  的VLLM，没有本任务PID。完整记录：`tmp/gdn2-rmsnorm-swiglu/performance-analysis.md` 与
  `tmp/gdn2-rmsnorm-swiglu/profile-conservative-v1/analysis.md`。
- **决策边界**：最快opt-in以18层额外1,029,832,704 bytes换2.6%吞吐，默认暂不切换。继续抠
  custom对裸W12 GEMV的1.564us/layer差距最多只值约28.1us/token；把W3并入也不减少约85.8MB/
  layer的W12+W3权重流。下一次大步收益需要weight-only低精度及独立任务质量预算；或者先设计
  prompt/decode共用布局，把当前1.03GB重复权重去掉。

## 2. 下一步的三个候选（带推荐）

> **若继续用户当前指定的 GDN-2 路线**：BF16 raw-gate recurrent+output-norm 与静态双槽
> 成对流水、Graph、short-conv与原生RMSNorm/W1W2 packing均已完成；最新纯模型stateful Graph
> 低内存默认为2.654~2.664ms/token（375.37~376.86 token/s），含D2H/argmax/H2D的真实decode为
> 340.62~354.33 calls/s。**host gap与大部分elementwise fragmentation已解决**：最新profile
> 的权重流占90.175%。保持BF16 dtype的
> `RMSNorm2+packed W1/W2+SiLU×Mul`的gamma-fold v1虽做到kernel级1.077x，却已被整网精度否决；
> 保守v2已通过真实整网、Graph夹心与profile，最快显式opt-in为2.587~2.590ms/token，代价约
> 1.03GB。其AIC MTE2平均91.88%，继续BF16融合只有约1% ceiling；若追求下一次大幅提升，应
> 单独决策weight-only低精度及任务质量预算。T=1 conv window/cache 复用
> 已经验收。训练/长 prefill 则做
> `gdn2_chunk_fwd_bwd`，但它被
> `gdn2-chunk-gate-range` 阻塞，必须先
> 解决真实 stress 中 1461 局部跨度的数值表示。不能复制 KDA stable，也不能拿现有 GDN
> kernel 改名冒充。下列三个仍是原 KDA 主线的排序。

### ① `decode-layer-overhead`（P1，推荐）

**单点收益最大。** 整层 decode 一步 458µs 里，KDA 算子只占 18%（83µs），
**层里其余占 67%（305µs）**，48 层外推 22ms/token。层里 T=1 时要走 7 个投影 + 3 个短卷积
+ 两次 fp32 l2norm 往返 + norm，每个几乎没有计算量却各要一次 launch。

> **把 kernel 再快一倍，整层只快 9%。** 这句话是这条的全部理由 —— 别先去调算子。

处置顺序写在缺口里：图捕获（形状固定，十几次 launch 压成一次）→ 合投影（q/k/v 并成一个
`[hidden, 3*key_dim]`，f/b/g 同理）→ 去掉 l2norm 的 dtype 往返。
复跑：`python benchmarks/bench_kda_decode_layer.py`。

### ② `fwd-caches-not-emitted`（P2，但顺带收益大）

九个反向检查点里 `g_cumsum` / `h` / `v_new` 现在在 host 侧用 torch 补。搬进 kernel 能
**同时**：去掉训练路径对内置算子包的依赖、删掉 `_scan_states(on_cpu=)` 与
`chunk_kda_bwd(layout_device=)` 两处 CPU 绕行、省掉每步 C 次 bmm。属于 kernel 队列。

### ③ `block-dim-ceiling`（P1）

物理 28 cube / 56 vec，chunk 的契约只声明到 4，而实测到 4 仍是**线性**扩展 ——
说明头寸还在。但 kernel 源码归 ascriptor 仓（`AGENTS.md §3` 只读），要跨仓做。
参考点：decode 那个自写 kernel 声明到 28 并实测跑通，所以 28 本身是安全的。

## 3. 动手前必读的五条（都是这两个会话踩出来的）

1. **kernel 的问题攒批统一修，不零散改**（`AGENTS.md §6.5`）。发现一条就记进 `gaps.json`
   打 `requires_kernel_change` + `kernel_change_note`，本仓侧先按 §7 装闸报错，
   等攒够一批再进 ascriptor 侧。`gen_matrix.py --check` 会校验队列不漏项。
2. **一个算子名，一个进程，一份 build。** 扫 `block_dim` 或比 `impl` 必须分进程，
   `runtime/binding.py` 的 `_claim_op_name()` 会拦。
3. **一个进程要用的 kernel 必须在第一次执行之前全部编完。** prefill+decode 的进程要
   `prepare(decode=True)`；测试靠 `tests/conftest.py` 的 session fixture（在测试函数里调
   **来不及**，pytest 同进程跑全部测试）。
4. **profile 之前不要相信任何性能推断**（§6 铁律一）。这两个会话里我在 `block_dim` 和
   decode 瓶颈上各错一次，都是先有推断后看数据。
5. **判结论要读未过滤的原始日志。** CANN 会打不带换行的 `path string is NULL`，
   粘在下一行前面，`grep -v` 会把整行删掉 —— 我因此差点对着少两行的输出下结论。
   做法：重定向到文件再 `nl -ba` 看，不要在管道里过滤。

## 4. 共享机器的规矩

机器清单、SSH 方式、CANN 路径、conda 环境在 **git-ignored 的 `machine_specs.md`**。
绝不把主机名/IP/端口/账号/路径写进任何会被提交的文件。

- 当前调度允许从4~7里选择空卡，kernel调试另获准使用物理0；先前固定物理7的窗口已经结束。
  每次都必须以**当下空闲**为准，授权不等于预约：本轮物理4在静态验证期间就被外部作业占用，
  随后的物理0也在测试启动后被外部作业抢占，因此都立即停止继续使用。用卡前
  `npu-smi info` 看 Health，
  再 `npu-smi info -t proc-mem -i N` 确认无进程。卡会中途从 OK 变 `Critical`，
  换卡要把理由和当时的占用数字写进环境脚本的注释。
- **登录 shell 可能把 `ASCEND_RT_VISIBLE_DEVICES` 重置为全部设备。** 不能只在
  `docker exec -e` 里传卡号再启动 `bash -l`；要在所有 profile/CANN 环境加载之后重新
  export，并在 import torch_npu 前断言“只看见一张卡”。2026-09-14 的受控探针与随后六轮
  真机任务均由宿主每秒同时采样物理 0/5：六轮合计 151 个时间样本，物理 0 活跃样本为 0，
  选定卡活跃样本为 38，任务只落在选定卡。2026-09-15 profiler 的 kernel CSV 全部显示
  Device_id=7，宿主监控里本任务 PID 也只在7；0上的 VLLM 属于另一容器。完整环境细节与
  原始监控日志分别放 git-ignored 的 `machine_specs.md`、`tmp/gdn2-{recurrent,profiler}/`。
  例外：2026-09-15 的 canonical aclnn **fresh build** 在 actual launcher 前捕捉到物理0上
  `python`/0MB 一次，时间对齐 CANN `opc` 编译；`test_aclnnop` 仍只在7。它没有计算/显存驻留
  证据，但也不能记成“物理0完全未触碰”，后续严格单卡窗口禁止再做 fresh CCE build。
- 绝不 kill 或修改他人进程；共享宿主机上不要复位别人也在用的卡。
- 装包只进自己的 venv（`python -m venv --system-site-packages`）。
- `scp` 完一定对 `md5sum`（中断会留下不完整文件且不报错）。
- **开机必查** `ls $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/` 有没有
  `ascend950`：没有则 torch_npu 的计算算子全不可用，**层级验证做不了**，
  只能做纯前向的算子级验证（可用面表见 `AGENTS.md §5`）。
- 收工前扫一遍自己留下的后台循环：`pgrep -af "unti[l] ! pgrep"`（模式要自排除，
  否则 `pgrep -f` 会匹配到自己 —— 实测留下过 9 个永不退出的空转循环）。

## 5. 怎么复跑各项结论

| 结论 | 命令 |
|---|---|
| 主机侧一致性（常量、声明、契约对齐） | `python -m pytest tests/ -q` |
| 真机全量 | `python -m pytest tests/ -q`（在有 `ascend950` 的机器上） |
| GDN-2 文本生成 | eager：`python benchmarks/generate_gdn2_text.py --checkpoint <checkpoint.pth> --tokenizer <tokenizer-dir> --device npu:0 --temperature 0 --top-k 0`；graph另加 `--core-backend cce --projection-layout packed-inference --decode-backend npu-graph` |
| GDN-2 CPU vs torch_npu | `python benchmarks/verify_gdn2_cpu_npu.py --device npu:0`（FP32 relative-L2，不判 bitwise） |
| GDN-2 canonical vs packed | `python benchmarks/verify_gdn2_packed.py --checkpoint <checkpoint.pth> --device npu:0 --dtype float32 --order canonical-first`；BF16 与反序加载另跑，bitwise/argmax 只作诊断 |
| GDN-2 CCE recurrent 算子 | `python benchmarks/verify_gdn2_recurrent.py --device npu:0 --block-dim 8`；调用前必须已把进程可见设备限制为一张允许卡 |
| GDN-2 torch_npu vs CCE 整网 | `python benchmarks/verify_gdn2_backends.py --checkpoint <checkpoint.pth> --device npu:0 --dtype bfloat16 --order torch-first`；FP32 与反序加载另跑 |
| GDN-2 decode profiler | `python benchmarks/profile_gdn2_backend.py --checkpoint <checkpoint.pth> --trace-dir <empty-dir> --output <profile.json> --device npu:0 --core-backend cce`；torch backend 必须另开进程 |
| GDN-2 NPU Graph | `python benchmarks/bench_gdn2_graph_capture.py --checkpoint <checkpoint.pth> --device npu:0 --mode stateful --short-conv-path cce --output <result.json>`；相邻独立进程用 `--short-conv-path torch` 作旧路径A/B；`--mode fixed-cache` 只量host上限，不代表可递推生成 |
| GDN-2 Ascriptor unit CCE | `python kernels/projects/a5/gdn2_fused_recurrent/run.py check --case all --launcher aclnn --device a5 --backend cce --block-dim 8` |
| 真实形状精度验收 | `python benchmarks/verify_real_shapes.py --check drift\|bitwise\|gqa\|bwd` |
| decode 验收 | `python benchmarks/verify_decode.py`（acc / chain / bd / split） |
| decode 整层拆解 | `python benchmarks/bench_kda_decode_layer.py` |
| 门控跨度扫描 | `python benchmarks/probe_bwd_span.py` |
| kernel 静态校验（**本机可跑**） | `PYTHONPATH=<ascriptor>/library ascriptor check <file>::<fn>` |
| 矩阵一致性 | `python tools/gen_matrix.py --check` |

## 6. 本会话纠正过的判断（别重新推导错一遍）

这一节的价值在于**阻止重复犯错**，所以都留着，连同错误的那一版。

| 我原本以为 | 实测 |
|---|---|
| decode 是带宽瓶颈（state 64KB×2/头 ≈ 2.5µs） | **错**。是每次调用约 48~58µs 的固定成本；设备侧边际只有 2.7~4.8 µs/token |
| 优化 decode 要调 kernel | **错**。整层 458µs 里算子占 18%，层侧占 67% |
| `o = qᵀ·state_dec + β(q·k)·delta` 能省一趟 | 恒等式对，但 `RegList.cadd()` 的结果**只落在 lane 0**（不广播），当乘数用只有 1/128 个 lane 对。改成把读出挂到第二趟，连 `cadd` 都不用，IR op 184→156 |
| 反向的精度已经到 ABI 地板（bf16 `g_cumsum`） | **错**。算子离 fp32 参考反而更近（跨度 46：对 fp32 2.889e-02，对 bf16-g 4.169e-02）。我那个构造把 bf16 cumsum 差分回增量，是灾难性相消 |
| 默认初始化的门控跨度 ≈ 94（常数） | **是随机变量**，上界 100.8。同 seed 换 RNG 消耗顺序就从 64.55 变 94.0。所以反向闸从 100 改成 **105**，测试用 `_calibrate_span` 确定性标定 |
| C≥2 的失败是"跨步 D2H 静默给错数据" | 是**硬报错**（`Op Slice does not has any binary` / `errno:561000`）。修法不变（先整块 D2H 再切），但机制写错会误导下一个人 |
| `ShortConvolution` 只有整段前向 | **错**。它本来就支持 `cache` + `output_final_state` 单步解码，还处理 `T < kernel_size` 的补零 |
| `torch-npu-baseline-missing` 还没做 | 早做完了，缺口忘了关 —— 矩阵一边说"已完成"一边说"尚无实现"。**验收打 ✅ 时要回头关对应缺口** |
| 孤立验证 packed low projection 的误差很小，就能推出整链在预算内 | **错**。T>1 的 split 是跨行 view，继续喂第二层 Linear 改了 Ascend 路径；必须逐段再测下一消费者，并最终对 full logits/cache 设 relative-L2 硬闸。bitwise 只作定位线索 |
| `docker exec -e ASCEND_RT_VISIBLE_DEVICES=N ... bash -lc ...` 会一直保留单卡可见性 | **错**。登录初始化可把它重置成全部设备，此后代码里的 `npu:0` 就是物理 0。卡号必须在 login/profile/CANN 初始化后再 export，并由宿主 `npu-smi` 在任务运行期间验证实际落卡 |
| GDN-2 recurrent 公开算子快 1.683x，整网也会接近这个倍数 | **错**。trace 里 recurrent kernel 本体只占 CCE 设备时间1.88%，core连 casts 也只有5.52%；paired whole-model 两次为1.021x与0.975x，只能判持平。先减少周边融合边界和 host 固定成本 |

## 7. 两个"闸"的现状（改之前先读）

| 闸 | 值 | 依据 |
|---|---|---|
| 门控跨度（`MAX_GATE_SPAN`） | `stable` 前向 **155** / 反向 **105**；`upstream` 均 80 | 两维是因为两条链约束不同：前向受**有限性**约束（到 155.97 精度不退化），反向受**精度**约束且先于有限性到来（有限到 169.8，但 `dq` 在 130 超预算）。105 的**下界**由 fla 初始化的跨度上界 100.8 定 —— 低于它会把默认初始化的层用自己的门控拒掉 |
| C=1 多头（`_check_single_chunk_heads`） | `C==1 and B*HV > block_dim` 报错 | 上游 `kda_sub45_fused_kernel` 在这里写出**静默错误**的 `o`（有限、量级正常、内容错，只有每个 cube 核的最后一个头对）。实测表逐格钉在 `tests/test_kda_gating.py` —— 这类缺陷闸一松就再没有东西会报警 |

**后果**：64 token 粒度的 prefill 用不了（T=64 即 C=1）。prefill 要么一次 ≥128 token，
要么按头分批。decode 不受影响（不走那个 kernel）。

## 8. decode 的用法（刚接好，还没别处记）

```python
from ascend_fla.ops.kda import prepare
prepare(decode=True)          # 进程启动时一次；晚了会报"已经执行过 aclnn 算子"

cache = {}                                     # 空字典 = 从零开始并开始记录
o, _ = layer(x_prefill, cache=cache, mode="chunk")          # T 必须是 64 的倍数
for step in steps:                                          # T ≤ 16
    o, _ = layer(step, cache=cache, mode="fused_recurrent") # cache 原地更新
```

- `cache` 两个键：`recurrent_state`（`[B,HV,128,128]` fp32，**K 在前**）与
  `conv_state`（`(q,k,v)` 三个 `[B,D,conv_size]`）。
- **两条路径不自动互换**：服务不了就报错并指出另一条。数学等价但数值不同
  （decode 全 fp32，对参考 1e-07；chunk 有 bf16 中间量，3e-03）。
- **decode 不可求导**（只有 chunk 有反向 kernel）。层里直接报错，不会静默不建图。
- 接 HF/fla 模型还要一层 cache 适配器：fla 的 KDA layer 用 `state_v_first=True`
  （V 在前），见 `state-layout-k-first`。

## 9. 本会话的提交（倒序）

```
8d1f183 记上全量真机结果：63 passed
ad3dc60 矩阵里"decode 完全缺失"的条目同步为已完成一半
cccd7f8 decode 接进 layer：两条路径 + cache 交接，并把遗留问题与优化点记全
e3ec138 探 decode：自写 kda_fused_recurrent kernel，真机通了，但瓶颈不在算子
3cd44dc 建立 kernel 问题的统一修复队列：记账 + 装闸，不零散改
7be9a65 真实形状精度验收四项全部完成：链长不累积、核切分逐位相同、GQA 与反向都在预算内
7cf0dd1 真实形状精度验收抓到一个 P0 静默错误：C=1 多头时 o 内容错，已装闸
941d1b3 关掉过期的 torch_npu 基线缺口，并把 toy-case-shapes 收窄到精度侧
aa73c96 远端等待循环会自匹配 pgrep 模式：记下这个坑与替代写法
1047f58 整层反向在真机跑通，并修正三处被实测纠正的说法
```

## 10. 没做的事（按是否阻塞第三期）

**阻塞第三期**：`decode-layer-overhead`、`decode-call-overhead`、`fwd-caches-not-emitted`、
`state-layout-k-first`（接 HF 模型要它）、矩阵 CI。

**不阻塞**：`stable-unit-no-harness`（三个本仓单元都没接 ascriptor harness —— `check` 与
runtime 桥都过了，缺的是 `unit.py` + `run.py`）、`kernel-nd2nz-suboptimal`（lint 标出的访存
低效点，带板上实测倍数与具体改法）、GDN/DeltaNet 扩族（第四期，六项 ABI 缺口）。

**刻意不做**：`gate-span-still-bounded` 的进一步放宽（反向的约束是精度不是有限性，
分块 log-sum-exp 可能帮不上忙，先做 `kda-fwd-bwd-dtype-mismatch` 看曲线会不会整体下移
—— 那是推测，要测）。
