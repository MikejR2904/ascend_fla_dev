<!-- 由 tools/gen_matrix.py 生成，请勿手改。改 docs/matrix/*.json 后重新运行。 -->

# 支持矩阵

记录于 2026-09-10。本文件由 `docs/matrix/*.json` 生成。状态词汇沿用 ascriptor：`passed` / `untested` / `gap` / `failed`。

## 目标模型形状

ascriptor A5 定尺 ABI：`[B, H, C, L, D]`，L=64，D=128，q/k/v `bfloat16`，beta/g `float32`。

| 模型 | 算子族 | 优先级 | 目标期 | H | HV | head_k | head_v | dtype | 定尺匹配 | 阻塞缺口 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B-Instruct | gated_delta_rule | secondary | 第 4 期 | 16 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ❌ | `gdn-no-gqa` |
| Kimi-Linear-48B-A3B-Instruct | kda | primary | 第 1 期 | 32 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ✅ | — |
| fla GatedDeltaNetConfig 默认值 | gated_delta_rule | reference-only | — | 6 | 6 | 256 | 512 | — | ❌ ❌ ✅ | `fixed-kv-128`, `asymmetric-kv-dim` |
| fla KDAConfig 默认值 | kda | reference-only | — | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |
| fla DeltaNetConfig 默认值 | delta_rule | reference-only | — | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |

定尺匹配三格依次为 head_k / head_v / head 分组。

### 算子测试应覆盖的形状

- **smoke** — B=1, H=1, C=1 · ascriptor 现有 case 的规模，仅用于接线冒烟
- **qwen3_next_layer** — B=1, H=16, HV=32, C=16, T=1024 · 单层真实形状，第一期精度验收目标
- **kimi_linear_layer** — B=1, H=32, HV=32, C=16, T=1024 · KDA 单层真实形状
- **long_context** — B=1, H=16, HV=32, C=64, T=4096 · 覆盖 chunk 边界与 state 传递，不是为了测误差累积

> 算子测试矩阵应覆盖的形状。T 必须是 64 的倍数（L=64 无 tail 路径），C = T / 64。

### 待核实的内部规格

- **qwen3.5-9b**（gated_delta_rule）：32 heads / head_dim 128 / chunk 64, bf16, 24 layers — 规格需从权威 config 核实后再补入 models[]；当前仅作参考，不要当作已验证形状。

## 算子支持状态

ascriptor pin：`0.1.0.dev1` · library `77619116f9b3` · 支持硬件 a5 · deferred a2, a3

| 算子 | 族 | 方向 | reference | sim | pipesim | emit | **compile** | board(cce) | 本仓接线 |
|---|---|---|---|---|---|---|---|---|---|
| `a5.gdn_fwd` | gated_delta_rule | forward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.gdn_bwd` | gated_delta_rule | backward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| **`a5.kda_fwd`** ★ | kda | forward | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | ✅ 完成 |
| **`a5.kda_bwd`** ★ | kda | backward | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.delta_rule_fwd` | delta_rule | forward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.delta_rule_bwd` | delta_rule | backward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |

★ 标记第一期的首个目标。

> **compile** 一列指 `ascriptor compile` CLI（纯源码发射+编译、不执行），**不是** aclnn launcher —— unit runner 把 board / aclnn / pypto 都记为 `board` stage。本仓的 aclnn 本地编译与零拷贝调用已独立实测通过，见下方 `our_runtime_bridge`。

### 缺失的算子

- **kda_fused_recurrent**（kda）— decode 路径（逐 token 递推 + state 传递）。完全缺失，需要新写，不是接线。没有它就只有 prefill/训练，没有推理解码，也失去 chunk↔recurrent 互验这个最好的 oracle。第三期目标。
- **gdn_fused_recurrent**（gated_delta_rule）— decode 路径。同上。随 GDN 扩族（第四期）再补。

### 可复用原语

| 原语 | 对应 fla | 本仓状态 |
|---|---|---|
| `chunk_row_scan` | fla 的 chunk cumsum（ops/utils/cumsum.py 的 chunk_local_cumsum） | ⬜ 未开始 |
| `matrix_normalization.row_l2` | fla.modules.l2norm | ⬜ 未开始 |
| `gated_approximations` | fla.modules.activations / fused swiglu | ⬜ 未开始 |

### 性能基线

> 两条基线。基线一：torch_npu 原生算子拼出的同语义 KDA（ascend_fla/reference/kda.py 的 kda_chunk_vectorized），它同时是第二 oracle。基线二：本仓经 runtime 桥调用的 ascriptor 自编译算子（ascend_fla/ops/kda/chunk.py）。两者**同机同卡同输入**测量，否则加速比无意义。报数必须带形状/dtype/warmup/iters/是否同步 —— 见 AGENTS.md §6。

机器：8-card Ascend950PR docker host, CANN 9.2.0 (V100R001C25B046), torch 2.10.0+cpu / torch_npu 2.10.0.post2, python 3.11.16, NPU 0 · 记录于 2026-09-11

条件：dtype=bfloat16 K=V=128 chunk=64 warmup=3 iters=10 synchronized=yes forward_only=yes

> 与 ascriptor_self_compiled 同一批运行测得（每个 block_dim 的子进程各自重测一次基线），所以两张表可以直接比。先前在 NPU 7 / iters=5 下测的 5.866 / 6.497 / 7.040 / 14.469ms 已被这组取代 —— 条件不同的数不能混用。

| 形状 | B/H/HV/T | 四次测量 (ms) | 中位数 | o relL2 vs CPU |
|---|---|---|---|---|
| smoke | B1/H1/HV1/T64 | 6.633 / 6.473 / 4.581 / 4.684 | 5.579 | 1.898e-05 |
| kimi_linear_layer | B1/H32/HV32/T1024 | 7.920 / 5.445 / 5.439 / 5.811 | 5.628 | 2.721e-05 |
| qwen3_next_layer | B1/H16/HV32/T1024 | 8.051 / 5.524 / 5.115 / 5.697 | 5.611 | 2.765e-05 |
| long_context | B1/H16/HV32/T4096 | 17.668 / 12.155 / 11.816 / 12.223 | 12.189 | 2.622e-05 |

**波动**：四次测量里**第一个子进程始终最慢**（6.633 / 7.920 / 8.051 / 17.668），应是设备初始化与缓存冷启的成本漏进了它的计时。中位数比均值更能代表稳态。


> 每个 block_dim 一个子进程 —— 同名算子多 build 在一进程内会互相覆盖，见 gaps.json 的 op-name-collision-in-process。四个 block_dim 的 relL2 完全相同。

| 形状 | B/H/HV/T | bd=1 | bd=2 | bd=3 | bd=4 | bd1→4 | o relL2 |
|---|---|---|---|---|---|---|---|
| smoke | B1/H1/HV1/T64 | 0.334 | 0.238 | 0.222 | 0.238 | 1.40x | 3.288e-03 |
| kimi_linear_layer | B1/H32/HV32/T1024 | 4.955 | 2.516 | 1.724 | 1.311 | 3.78x | 3.962e-03 |
| qwen3_next_layer | B1/H16/HV32/T1024 | 4.923 | 2.491 | 1.713 | 1.302 | 3.78x | 3.964e-03 |
| long_context | B1/H16/HV32/T4096 | 19.622 | 9.971 | 6.794 | 5.138 | 3.82x | 4.168e-03 |

**block_dim=4 下 vs torch_npu 基线**：smoke 19.7x · kimi_linear_layer 4.43x · qwen3_next_layer 4.38x · long_context 2.38x

> benchmarks/profile_bridge_overhead.py --sync-each 的设备耗时归因（ms/次）。同步会破坏流水，所以总和大于流水模式下的总耗时。

| 段 | bd=1 (ms) | bd=4 (ms) |
|---|---|---|
| `KdaSub2ScoreKernel` | 1.583 | 0.407 |
| `TrilInverse64V2StrictBf16Kernel` | 1.354 | 0.361 |
| `KdaSub45FusedKernel` | 1.311 | 0.329 |
| `KdaSub3WyKernel` | 0.486 | 0.129 |
| `KdaSub1GateKernel` | 0.198 | 0.058 |
| `layout_to_bhcld_x5` | 0.318 | 0.276 |
| `layout_from_bhcld` | 0.095 | 0.064 |
| `total` | 5.838 | 1.901 |

**怎么读**：layout 重排只占 7%（bd1）—— 我先前猜它是瓶颈，错了。大头是三个 kernel，而它们正是 ascriptor lint 里 15 处 ub_to_l1.nd2nz 的所在（见 gaps.json 的 kernel-nd2nz-suboptimal）。bd=4 下 layout 占比升到 18%，因为 kernel 侧随核数缩短而重排不随。


**观察**：torch_npu 基线：数据量从 smoke 到 kimi_linear_layer 差 512 倍，耗时只差 ~15% —— 它完全被 kernel launch 开销支配（向量化后仍有 63 次求逆迭代 + NT 次 chunk 迭代的 python 循环），**不是硬件算力上限**。自编译侧相反：耗时随工作量近线性（T 从 1024 到 4096，bd4 下 1.311→5.138ms，正好 3.9 倍），是真正的算力账。这也解释了 smoke 上 19.7x 的加速 —— 那里 torch_npu 在付固定开销而我们不付。

**跨 CANN 版本一致性**：kda_fwd 经 runtime 桥在 CANN 9.1.0 与 9.2.0 两台机器上的 relL2 逐位相同（smoke 3.288e-03 / multi_chunk 3.383e-03 / gva 3.359e-03），说明这个偏差来自算子自身的数值路径（见 gaps.json 的 kda-fwd-bwd-dtype-mismatch），与 CANN 版本无关。

**测量注意**：torch_npu 基线的跑间波动约 20%（kimi_linear_layer 在四个子进程里测到 7.920 / 5.445 / 5.439 / 5.811ms），所以加速比带同等量级的不确定度。每行的比值用的是该子进程自己测的基线，不是跨进程平均。

**下一步**：① 抬高 block_dim 上限（gaps.json 的 block-dim-ceiling，已升 P1）—— 扩展性到 4 仍线性，物理上有 28 cube。② kernel 侧的 nd2nz 返工（kernel-nd2nz-suboptimal）。③ 反向的同类测量，第二期随 kda_bwd 一起做。

### 全链路三层

> 全链路的上三层。窄切片原则：按算子倒推，用到哪个做哪个。首个目标是 KDA 链路，其依赖面比 GDN 少一个 module。

**modules**

- `causal_conv1d` — ⬜ 未开始 · none — 需新写
- `fused_rms_norm_gated` — ⬜ 未开始 · partial — matrix_normalization 可借
- `rms_norm` — ⬜ 未开始 · partial — matrix_normalization 可借
- `l2norm` — ⬜ 未开始 · matrix_normalization.row_l2

**layers**

- `kda` — ⬜ 未开始
- `gated_deltanet` — ⬜ 未开始

**models**

- `kimi-linear` — ⬜ 未开始 · 注入 — 用上游模型定义，替换 linear attention layer
- `qwen3-next` — ⬜ 未开始 · 注入 — 用 HF transformers 的模型定义，替换 linear attention layer

## 缺口

P0 0 项 · P1 11 项 · P2 7 项 · 共 23 项

**首个里程碑**：第一期五项已全部有结论，并补齐了同机性能对比：aclnn 编译、runtime 桥、kda_fwd 接线、KDA 本地基线均实测通过；自编译算子在 block_dim=4 下比 torch_npu 组合快 4.43x（kimi_linear_layer）/ 2.38x（long_context T=4096）/ 19.7x（smoke）。过程中修掉两个自己的 bug（bridge-per-call-overhead、op-name-collision-in-process），它们先后让 block_dim 的效果被完全掩盖。当前最大的性能项是 block-dim-ceiling（已升 P1）：扩展性一路线性到契约上限 4，而硬件有 28 cube。第二期的前置障碍仍是 kda-fwd-bwd-dtype-mismatch。

**建议的首个目标**：KDA（Kimi-Linear / fla-kda-default 形状）。其 ABI 已是 token-major BTHK、GQA 原生支持、initial_state 与 final_state 均为 FP32、backward 产出 dh0 —— 上述多数 ABI 缺口对它都不适用。唯一需要前置补齐的是本地验证证据（kda-no-local-evidence）。

### 为什么首个目标是 KDA

> 选择 KDA 作为首个目标的依据：下列能力 KDA 有、GDN 没有。

| 仅 KDA 具备 | 仅 GDN 具备 |
|---|---|
| GQA 分组 (HV % H == 0) | 本地 validation.json 证据齐全 |
| token-major 公开布局 |  |
| 非零 initial_state 输入 |  |
| backward 产出 dh0 |  |
| final_state 为 FP32 |  |
| block_dim 上限 4 而非 2 |  |

### 按算子族速查

| 算子族 | P0 | P1 | P2 |
|---|---|---|---|
| KDA | — | `kda-fwd-bwd-dtype-mismatch`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`no-tail-path`<br>`block-dim-ceiling` | `npu-builtin-ops-missing`<br>`toy-case-shapes`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`modules-layer-missing`<br>`torch-npu-baseline-missing`<br>`kernel-nd2nz-suboptimal` |
| GDN | — | `gdn-no-gqa`<br>`layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`state-dtype-bf16`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path`<br>`block-dim-ceiling` | `npu-builtin-ops-missing`<br>`toy-case-shapes`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`modules-layer-missing`<br>`torch-npu-baseline-missing` |
| DeltaNet | — | `layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path` | `npu-builtin-ops-missing`<br>`toy-case-shapes`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`torch-npu-baseline-missing` |

### P1

#### `kda-fwd-bwd-dtype-mismatch` — kda 的 fwd 与 bwd 对同名张量声明了不同 dtype

- **类别** abi · **适用于** KDA · **阻塞** `phase 2`
- **依据** kda_fwd inputs：beta float32、initial_state float32、g_raw float32，final_state 输出 float32。kda_bwd inputs：beta bfloat16、initial_state bfloat16、g bfloat16、dht bfloat16。
- **影响** autograd.Function 组装时，forward 保存的 FP32 张量必须降到 BF16 才能喂给 backward，这一步有精度损失且不在任何一侧的契约预算内。
- **建议** 第二期组装 autograd 前先量化这次降精度的影响：同输入下 FP32 保存 vs BF16 保存的梯度差异。若不可接受，需要改 bwd 的 ABI 接受 FP32。不要默默插一个 .to(bfloat16) 了事。

#### `gdn-no-gqa` — gdn_fwd/bwd 无独立 value-head 维度，不支持 GQA 分组

- **类别** abi · **适用于** GDN · **阻塞** `qwen3-next-80b-a3b`, `phase 4`
- **依据** gdn_fwd contract.json domain.shape："B,H,C are positive runtime dimensions" —— 没有 HV。对比 kda_fwd domain 有 "H_HV": "positive; HV % H == 0"，kda_bwd 亦然。
- **影响** 阻塞 Qwen3-Next（16 key heads / 32 value heads）。
- **建议** 第四期做 GDN 扩族时一并解决，实现可借鉴 kda 已有的 HV%H==0 分区方式。严重度因第一期改走 KDA 而由 P0 降至 P1。

#### `layout-not-token-major` — gdn 与 delta_rule 的公开布局是 [B,H,C,L,D]，与 fla 的 token-major 不一致

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** gdn_fwd contract domain.layout："Contiguous CPU tensors in [B,H,C,L,D]"。对比 kda_fwd："Contiguous token-major public tensors; explicit local permutation to BHCLK/BHVCLK kernel tensors"，kda_bwd："Public tensors are contiguous BTHK/BTHV"。
- **影响** 接入 GDN 时调用方需要 permute，带来额外访存开销。KDA 不受影响 —— 它的公开接口已是 token-major，内部自行 permute。
- **建议** GDN 接线时让 kernel 内部做 permute（照 kda 的做法），而不是把转换推给调用方。

#### `nonzero-initial-state` — gdn 与 delta_rule 只支持零初始 state

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** gdn_fwd / delta_rule_fwd contract.json domain.initial_state: "zero only"。kda_fwd 则以 float32 [B,HV,128,128] 作为正式输入，case 中有 random 与 zero 两种。
- **影响** GDN/DeltaNet 无法做 state 传递 —— 长序列分段训练、prefill→decode 交接、chunked prefill 都做不了。KDA 不受影响。
- **建议** GDN 扩族时照 kda_fwd 的方式加非零初始 state 入口。在此之前，门控必须对 GDN 传入非零 initial_state 的调用报错。

#### `d-initial-state-absent` — gdn 与 delta_rule 的 backward 不产出初始 state 的梯度

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** gdn_bwd / delta_rule_bwd contract.json："no d_initial_state in the preserved backward production ABI"。对比 kda_bwd 有 dh0 输出（[B,HV,128,128]）与 dht 输入。
- **影响** GDN/DeltaNet 无法支持可训练初始 state、序列并行（CP）与分段反向。KDA 不受影响。
- **建议** 记录为 GDN/DeltaNet 的已知限制。门控应在 initial_state.requires_grad 时报错。

#### `state-dtype-bf16` — gdn 的 final_state 为 BF16，fla 惯例为 FP32

- **类别** precision · **适用于** GDN · **阻塞** —
- **依据** gdn_fwd contract outputs.final_state: bfloat16 [B,H,128,128]。对比 kda_fwd 的 final_state 为 float32。
- **影响** state 是跨 chunk 累积量，BF16 存储的误差会进入下一段递推。影响幅度未在 A5 上测过。KDA 不受影响。
- **建议** GDN 接线时测：同形状下 BF16 state 与 FP32 state 的输出差异，长 C（如 C=64）下是否放大。不要沿用 A2 上的结论。

#### `fused-recurrent-missing` — decode 路径算子完全缺失（全算子族）

- **类别** coverage · **适用于** KDA / GDN / DeltaNet · **阻塞** `phase 3`
- **依据** ascriptor kernels catalog 中无 fused_recurrent 类单元；现有六个单元均为 chunk 路径。
- **影响** 只有训练与 prefill，没有推理解码。也失去了 chunk↔recurrent 互验这个最好的 oracle。
- **建议** 第三期为 KDA 新写（非接线），以 fla.ops.kda.fused_recurrent_kda 为语义基准。GDN/DeltaNet 的 decode 路径随各自扩族再补。

#### `no-varlen` — 无变长序列（cu_seqlens）支持

- **类别** coverage · **适用于** 全部 · **阻塞** —
- **依据** kda_fwd contract domain.scope 明确写 "Fixed length; no cu_seqlens, cp_context, safe_gate, gate fusion or state_v_first"。其余单元 ABI 亦均为规整形状。
- **影响** 训练常用的 sequence packing 无法使用，等长 padding 会浪费算力。顺带：gate fusion 不支持，意味着 g 必须在 kernel 外算好再传入。
- **建议** 列为开放问题（见 docs/plan.md §7）。短期门控拒绝。gate 在外计算对 KDA layer 是自然的（f_proj 本就是 Linear），不构成阻塞。

#### `scale-param-no-slot` — fla 的 scale 参数在 ascriptor ABI 中无入口

- **类别** abi · **适用于** GDN / DeltaNet · **阻塞** `phase 4`
- **依据** 各单元 inputs 中均无 scale 标量。gdn/delta_rule 的 case parameters 里的 "scale": 0.05 是输入生成幅度（delta_rule_fwd domain.input_values: "generated q/k/v scale 0.05"；kda_fwd domain.input_generation: "q/k/v stddev 0.04"），不是算子参数。 【2026-09-11 修正】KDA 不受影响：kda_sub2_score_kernel 与 kda_sub45_fused_kernel 都有 `scale: f32` 标量参数（单元固定传 128**-0.5），kernel 层面有入口，不需要 host 预乘。本仓 chunk_kda_fwd 已把 scale 作为可选参数直通 kernel。
- **影响** fla 语义下 q 要乘 scale（默认 head_dim**-0.5 ≈ 0.0884）。无入口则只能 host 侧预乘，多一次 elementwise 全量遍历，与性能目标冲突。
- **建议** 优先在 kernel 内吸收 scale（已有 q 的读取点可顺带乘）。第一期若先用 host 预乘打通，必须在性能报告中标注这部分开销。

#### `no-tail-path` — L=64 固定且无 tail 路径，T 必须是 64 的整数倍

- **类别** abi · **适用于** 全部 · **阻塞** —
- **依据** kda_fwd domain.tails："T=C*64; partial chunks, K/V tails, fp16, and L=32 are rejected by this authored kernel unit." 其余单元同为 "No L/D tails"。
- **影响** 任意序列长度的推理与训练都需要 host 侧 padding，或拒绝。另：fp16 与 L=32 同样被拒绝。
- **建议** 门控显式报错并给出最近的合法 T。padding 方案要在性能报告中算进开销。

#### `block-dim-ceiling` — kda 的 block_dim 上限 4 是当前最大的性能天花板：扩展性到 4 仍是线性的

- **类别** performance · **适用于** KDA / GDN · **阻塞** `phase 4`
- **依据** kda_fwd / kda_bwd domain.block_dim: [1,2,3,4]；gdn domain.block_dim: [1,2]；delta_rule_fwd domain.block_dim: {min:1, max:32}。 | 2026-09-11 实测（Ascend950PR / CANN 9.2.0，每个 block_dim 一个进程，bf16，warmup 3 / iters 10，同步，仅前向）：kimi_linear_layer 4.955 / 2.516 / 1.724 / 1.311ms（bd=1/2/3/4，bd1→4 提速 3.78x）；long_context 19.622 / 9.971 / 6.794 / 5.138ms（3.82x）。单 kernel 级（sync-each 归因）：三个重 kernel 从 1.583/1.354/1.311ms 降到 0.407/0.361/0.329ms。四个 block_dim 的 relL2 完全相同，分区不改变数值结果。
- **影响** **扩展性一路线性到声明上限，说明这是声明限制而不是实现限制。** Ascend950PR 物理上有 28 cube / 56 vec，我们只用到 4 —— 按线性外推还有 ~7 倍空间。这让它成为比 kernel-nd2nz-suboptimal 更靠前的优化项：后者是常数因子，前者是可用核数。
- **建议** 已有答案：是声明限制。下一步是在 ascriptor 侧把 kda_fwd/kda_bwd 的 domain.block_dim 上限抬高并补 cases 覆盖（contract 的 core_ownership 说 gate 按 B*HV*C 切、scores/WY/inverse 按 cube 组切、融合尾部按 B*HV 头对切 —— kimi 形状下 B*HV=32、B*HV*C=512，工作量足够喂满 28 核）。kernel 源码归 ascriptor 仓所有（AGENTS.md §3：只读），改动要走那一侧。本仓的 SUPPORTED_BLOCK_DIM 与 contract 的声明由 tests/test_kda_gating.py 锁在一起，抬高后会同时提醒。

### P2

#### `npu-builtin-ops-missing` — 内置算子包的覆盖随机器而异：部分 Ascend950PR 机器上 torch_npu 的计算算子不可用

- **类别** environment · **适用于** 全部 · **阻塞** —
- **依据** 238（Ascend950PR_957b / CANN 9.1.0）上 $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ 只有 ascend910_93 与 ascend910b 两个 SoC 目录。torch.randn(device='npu') 报 aclnnInplaceNormal_1_StatelessNormalAiCore 找不到 JSON 配置；torch.zeros、bf16->fp32 Cast 同样失败。torch 本身是 2.10.0+cpu。 【2026-09-11 补充】另一台 8 卡 Ascend950PR 机器（CANN 9.2.0，innerversion V100R001C25B046）的 opp 下有 **ascend950** 算子包，SoC 报 Ascend950PR_9579，实测 randn / zeros / fp32+bf16 matmul / bf16↔fp32 cast / permute+contiguous / einsum / cumsum 全部可用。所以这不是 SoC 级缺陷，而是**算子包安装差异**：CANN 9.1.0 的 opp 只装了 910 系列。
- **影响** 选机器决定能做什么：装了 ascend950 算子包的机器上 torch_npu 基线与 layer 级验证都可做；没装的机器上只能跑自编译 kernel（empty/H2D/D2H/data_ptr/stream 可用，计算算子全不可用）。runtime 桥在两种机器上都工作 —— 这正是它的价值。
- **建议** 三条路：① 性能基线改用 ascriptor 自己的 profile 子命令 + 自编译 kernel 之间的对比；② 在有完整算子包的机器上做 torch_npu 基线（a2/910B3 有 ascend910b）；③ 确认是否存在 950PR 的算子包可安装。选哪条取决于基线要回答的问题 —— 要对比 ascriptor vs torch_npu 就必须有内置算子，换机器是最直接的。

#### `toy-case-shapes` — 现有算子 case 全是玩具形状，未在真实模型形状上验证

- **类别** validation · **适用于** 全部 · **阻塞** `phase 1 验收`
- **依据** kda_fwd 四个 case 为 B=1, H=1, HV=1~2, C=1~2；gdn_fwd 为 B=1, H=1~3, C=1~3。
- **影响** 正确性证据的形状覆盖与真实负载相距很远（Kimi-Linear 单层为 H=HV=32，T=1024 时 C=16）。多核分区、UB 压力、核归属路径在玩具形状下可能根本没被触发。
- **建议** 第一期的精度验收必须用 models.json 的 test_case_shapes.kimi_linear_layer，而不是沿用上游 case。

#### `fixed-kv-128` — K=V=128 固定，不支持其他 head_dim

- **类别** coverage · **适用于** 全部 · **阻塞** `fla-gated-deltanet-default`
- **依据** 各单元 domain 固定 D=128 / K=128 V=128。
- **影响** fla GatedDeltaNetConfig 默认值（head_k=256 / head_v=512）不被支持。
- **建议** 不去支持它 —— 那是 fla 的参考默认值，不是落地模型规格。两个真实目标模型都是 128。门控明确拒绝并在错误信息中说明。

#### `asymmetric-kv-dim` — K 与 V 共用同一个 D 维，不支持 head_k != head_v

- **类别** abi · **适用于** 全部 · **阻塞** `fla-gated-deltanet-default`
- **依据** 各单元 inputs 中 q/k/v 的末维同为 128，domain 只声明单一 D=128（kda 分别声明 K=128 与 V=128，但两者都固定）。
- **影响** expand_v != 1.0 的配置不被支持（fla GatedDeltaNetConfig 默认 expand_v=2.0 → head_v 是 head_k 的两倍）。
- **建议** 与 fixed-kv-128 同样处置：两个真实目标模型都是 head_k == head_v == 128，不去支持非对称维度。门控拒绝并说明。

#### `modules-layer-missing` — modules 层的必需组件缺失

- **类别** coverage · **适用于** KDA / GDN · **阻塞** `phase 2`
- **依据** fla.layers.kda 只依赖 FusedRMSNormGated 与 ShortConvolution 两个 module（o_norm 用 activation="sigmoid"）；fla.layers.gated_deltanet 多需要一个 RMSNorm。ascriptor 侧有 matrix_normalization 与 gated_approximations 可部分借用，没有 causal_conv1d。
- **影响** 没有这些，layer 层无法组装，端到端梯度检验做不了。KDA 的依赖面比 GDN 少一个 module。
- **建议** 第二期按 KDA 的需要做窄切片：先 FusedRMSNormGated(sigmoid) 与 causal_conv1d。causal_conv1d 需新写；norm 可从 matrix_normalization 扩展。第一期可先用 torch 实现占位，但要在矩阵里标明是 torch 而非自有 kernel。

#### `torch-npu-baseline-missing` — 缺少 torch_npu 组合实现作基线与第二 oracle

- **类别** validation · **适用于** 全部 · **阻塞** `phase 1 验收`, `phase 2 性能报告`
- **依据** 本仓尚无任何实现代码。
- **影响** 没有它就只有一个 oracle（fla naive），也没有性能对照 —— 无法判断"高效率"是否达成。
- **建议** 第一期与 runtime 桥并行做：reference/ 下用 torch_npu 原生算子拼出同语义 KDA，同时充当基线与第二 oracle。

#### `kernel-nd2nz-suboptimal` — kda_fwd 的 5 个 kernel 里有 15 处 ub_to_l1.nd2nz，ascriptor lint 实测它比 compact-NZ 慢 10 倍

- **类别** performance · **适用于** KDA · **阻塞** —
- **依据** 编译 kda_fwd 时 ascriptor lint 在 intra.py(2)、triangular_inverse.py(8)、wy.py(2)、recurrent.py(3) 共 15 处报同一条 warning：ub_to_l1.nd2nz 会展开成多次 MTE3 burst（每个 NZ fractal 列一次），不是单条指令，板上实测比 compact-NZ move 慢 10 倍（D-084）。lint 还给了第二条：staging tile 的自然 block stride 是 tile 行数（天然 16 对齐，正好是 bank 阶梯最差的一档，~8.7 cycle/store vs ~1.1），补一行 padding 让它变奇数；D-226 里两个返工的 kernel 上这步收益 -12.3us / -10.8us（总量 -17.7us / -28.2us），比换 move 本身更大。
- **影响** 这 15 处在 kernel 源码里，不在本仓。按 D-226 的比例，单 kernel 量级的收益在数十微秒 —— 但要先确认设备侧耗时占比（见 bridge-per-call-overhead），占比低的话改了也看不出来。
- **建议** 第四期。kernel 源码归 ascriptor 仓所有（AGENTS.md §3：只读），所以这里只登记线索，改动要走 ascriptor 侧。动手前先用 benchmarks/profile_bridge_overhead.py 确认设备侧占比足够大，否则是在优化一个不在关键路径上的东西。
