# `a5.kda_fused_recurrent` — KDA 的 decode 路径

**本仓自写的第一个 kernel**（前两个本仓单元 `kda_fwd_stable` / `kda_bwd_stable` 是上游的
带标注改写）。理由：ascriptor 侧六个 a5 单元全是 chunk 路径，整族没有 recurrent 单元
（`gaps.json` 的 `fused-recurrent-missing`）。语义基准是 `fla.ops.kda.fused_recurrent_kda`，
本仓 oracle 是 `reference/kda.py` 的 `kda_recurrent_ref`。

## 什么时候用它

| 场景 | 路径 | 理由 |
|---|---|---|
| T=1（纯 decode） | **本单元** | chunk 要把 T 补到 64，白做 64 倍 |
| T=2~16（投机解码） | **本单元** | 同上，且 state 常驻，代价与 T 线性 |
| T≥64 且是 64 的倍数 | chunk | 那边把 64 个 token 批成矩阵乘，单位 token 便宜得多 |

## 算法与两趟扫

```
state ← state · exp(g)          # 逐 K 维衰减，g 是 [K]，沿 V 广播
delta  = v − kᵀ·state
state ← state + β·k ⊗ delta
o      = qᵀ·state               # 用的是更新后的 state
```

* **第一趟**：衰减 state，同时攒 `kᵀ·state_dec`（给 delta）。
* **第二趟**：做 rank-1 更新，**顺手攒 `qᵀ·state_new`** —— o 要的就是它。

两趟都在 UB 内，GM 只碰一次 state（读一次、写一次），所以 T 越大越摊薄。

**K 方向的规约不需要跨 lane 规约**：state 是 `[K,V]`，沿 K 即沿**行**，
所以是"取一个标量广播乘一行、累加"（`.single()` + RegList 累加）。

### 一个被实测证伪的"优化"

最初我用代数恒等式把 o 挪到第一趟：

```
o = qᵀ(state_dec + β·k⊗delta) = qᵀ·state_dec + β(q·k)·delta
```

恒等式本身是对的，但真机实测 **`o` 的相对 L2 是 8.5e-02，而同一次的 `final_state`
精确到 3.2e-08** —— 错只在 o。`final_state` 与 o 唯一不共享的就是那个修正项，于是定位到
`RegList.cadd()`：**它的结果只落在 lane 0，不是广播到所有 lane。**
（ascriptor 自家 kernel 里 `cadd()` 之后一律 `.single_value()` 取值；要当乘数用还得经 UB
再 `.single()` 读回 —— `gdn_bwd/kernels/scan_state.py` 的 `broadcast_scalar_vf` 就是干这个的。）
所以 `delta * corr` 只有 1/128 个 lane 是对的。

把读出挂到第二趟之后，连 `cadd` 都不需要了 —— **那个优化既错又多余**，IR op 数还从 184 降到 156。
留着这段是因为两件事可复用：① `cadd()` 不广播；② **误差的"位置"能指出 bug 在哪** ——
两个输出共享大部分计算，只有一个错，差集就是嫌疑范围。

## 并行：一个头一个核，不要跨核同步

按 `B*HV` 切给向量核，**每个头整份 state（64KB）常驻一个核的 UB**，核间不需要任何同步。
这是刻意的：上游 chunk 的融合尾部把 K 切给 sub-block、于是要手写同步，而手写那份假设了
C≥2 —— 那就是 `c1-multihead-o-corrupt` 那个 P0。本 kernel 的 kernel 级循环只有一层（头），
`auto_sync()` 能覆盖。

## 实测（2026-09-11，Ascend950PR / CANN 9.2.0）

**精度**：八个形状下 `o` 的相对 L2 **9.575e-08~1.789e-07**、`final_state`
**3.199e-08~1.440e-07**（对 fp32 递推参考）。比 chunk 路径的 3.2e-03 紧四个数量级 ——
因为全程 fp32，没有 bf16 中间量。

**state 串接**：逐 token 调 T 次并串接 state，与一次调 T 个 token **逐位相同**
（bd=1 与 bd=4 都验过）。decode 的正确性就建立在这条上。

**门控跨度没有上限**：逐 token 只用 `exp(g_i)`（量级 ~1.5），不存在 chunk 路径那种
`exp(累计跨度)` 的量程问题。这是 recurrent 形式的结构性优势 —— 对比
`gate-span-still-bounded`（chunk 路径前向 155 / 反向 105）。

**block_dim**：1/2/4/8/16/28 全部跑通且结果相同。28 → 56 个向量核，是物理上限，没有死锁。

**延迟（这是本轮最重要的发现）**：kimi decode 形状 B1/HV32/T1

| 项 | 耗时 |
|---|---|
| 整次调用 | 53~67 µs |
| host 布局转换 | 15.4~15.7 µs（26~29%） |
| **设备侧边际**（T=1→16） | **2.7~4.8 µs/token** |
| **每次调用的固定成本** | **约 48~58 µs** |

边际报区间是因为两次运行的 T=16 整次分别是 101.0 与 125.4 µs —— 单点会把噪声当结论。
边际与按「4 头/核 × 2 趟 × 128 行 × ~8 指令」估的设备时间同量级，所以
**固定成本才是瓶颈**：不是带宽（按 64KB×2/头估的下限约 2.5µs），也不是 kernel。
`block_dim` 在 1~28 之间总时长 54~67 µs **没有趋势**，正是这个的征兆。

我原本预测 decode 会是带宽瓶颈。**那个预测错了**，而且如果不拆开量就会把 bd 无效
误判成"扩展性不行"——正是 AGENTS.md §6 铁律一说的那种错。

下一步因此不是调 kernel，而是 `decode-call-overhead`：decode 期间形状固定，
桥可以把 acl tensor 描述符与 workspace 查询缓存起来复用。48 层模型按 55µs/层算是
2.6ms/token，不可接受。

复现：`benchmarks/verify_decode.py`（acc / chain / bd / split 四项）。

## 还没做

* **没接进 layer**：`layers/kda.py` 的 `mode="fused_recurrent"` 仍然报错。decode 还需要
  短卷积的逐 token 状态推进与 cache 寻址，那是另一件事。
* **没做 harness 集成**（`unit.py` + `run.py`），与两个 stable 单元同（`stable-unit-no-harness`）。
* **T>16 要分批**：入口报错而不是自动分批 —— 自动分批会把一次调用的语义悄悄变成多次，
  state 串接的正确性得另外验，那不是算子层该默默做的决定（AGENTS.md §7）。
* **GDN / DeltaNet 的 decode 没做**，随各自扩族再补。
