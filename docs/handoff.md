# 交接：下一个会话从哪里接

> 写于 2026-09-11 会话末。**本文件只记"接手需要知道的"**；权威状态在
> `docs/matrix/*.json`（`gen_matrix.py` 校验），纪律在 `AGENTS.md`，分期在 `docs/plan.md`。
> 三者冲突时以 json 为准，并顺手修本文件。

## 0. 2026-09-15 更新：SoC 顺序变更 + 多 agent 协作（先读这节）

**SoC 顺序改为 A2 (910B) → A3 (910C) → A5**（2026-09-14 定的，见 `AGENTS.md` §2 与 `docs/plan.md` 的「按 SoC 分波次」）。
所以下面 §2 那三个候选都是 A5 波次的事，要等 W-A3 退出之后再说。

工作改成多 agent 推进：PM 维护 `docs/pm/board.json`，agent 按 `docs/pm/PROTOCOL.md` 申领任务并汇报，
看进度跑 `python tools/pm_board.py --render`。

现在能派的是 W0 的主机侧任务，A2 机器还没到位。最靠前的两项是 A2-01（split-K FP32 cube 缺陷定性，A2 的总闸）
和 A2-04（C=1 多头那个 P0 的确切根因）。

三份 Gemini 规划文档只当需求来源看。它们与实测矛盾的说法，列在 `docs/pm/PROTOCOL.md` §6。

## 1. 一句话现状

第一期（runtime 桥 + `kda_fwd` 接线 + 基线）与第二期（`kda_bwd` + autograd + KDA layer）
**已完成**；第三期进行中 —— **KDA 的 decode 已自写 kernel 并接进 layer，功能可用、性能不可用**。
模型注入与矩阵 CI 未开始。

| 指标 | 值 |
|---|---|
| 真机全量测试 | **63 passed**（A5 / CANN 9.2.0，约 346s） |
| 主机侧测试（macOS 无 NPU） | 42 passed / 5 skipped |
| 缺口 | P0 **1** · P1 14 · P2 9 · 已解决 11 · 共 35 |
| kernel 统一修复队列 | **21 项**（`gaps.json` 的 `summary.kernel_fix_queue`） |
| 工作区 | 干净，全部已提交 |

## 2. 下一步的三个候选（带推荐）

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

- **只用 4~7 卡**，0~3 让给别人；用卡前 `npu-smi info` 看 Health，
  再 `npu-smi info -t proc-mem -i N` 确认无进程。卡会中途从 OK 变 `Critical`，
  换卡要把理由和当时的占用数字写进环境脚本的注释。
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
