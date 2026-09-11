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
| **`a5.kda_fwd_stable`** ★ | kda | forward | ✅ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ 完成 |
| **`a5.kda_bwd`** ★ | kda | backward | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | ✅ 完成 |
| **`a5.kda_bwd_stable`** ★ | kda | backward | ✅ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ 完成 |
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


#### 训练步（fwd+bwd）

> 完整一步训练（fwd + bwd）对照 torch_npu 组合版 + autograd。benchmarks/bench_kda_train_step.py，warmup 2 / iters 5，同步，layout_device=npu，每个 block_dim 一个子进程。fwd 一列**不含**门控跨度检查（与第一期的数可比），检查的代价单列。

记录于 2026-09-11

| 形状 | B/H/HV/T | bd | fwd | +检查点 | fwd+bwd | torch_npu | 加速 |
|---|---|---|---|---|---|---|---|
| smoke | B1/H1/HV1/T64 | 1 | 0.194 | 0.409 | 1.504 | 15.085 | 10.03x |
| smoke | B1/H1/HV1/T64 | 4 | 0.240 | 0.558 | 1.732 | 17.251 | 9.96x |
| kimi_linear_layer | B1/H32/HV32/T1024 | 1 | 4.898 | 5.952 | 15.864 | 21.731 | 1.37x |
| kimi_linear_layer | B1/H32/HV32/T1024 | 4 | 1.314 | 2.379 | 5.069 | 24.463 | 4.83x |
| qwen3_next_layer | B1/H16/HV32/T1024 | 1 | 4.914 | 5.955 | 15.893 | 25.549 | 1.61x |
| qwen3_next_layer | B1/H16/HV32/T1024 | 4 | 1.335 | 2.446 | 5.154 | 26.089 | 5.06x |
| long_context | B1/H16/HV32/T4096 | 1 | 19.567 | 23.809 | 63.792 | 106.711 | 1.67x |
| long_context | B1/H16/HV32/T4096 | 4 | 5.134 | 9.485 | 20.275 | 114.935 | 5.67x |

kimi_linear_layer / bd=4 的拆分（ms）：fwd_kernels 1.314 · caches_host_side 1.065 · bwd_kernels 2.690 · gate_range_check 0.212 · total 5.069

**怎么读**：① 训练步的加速比（bd=4 下 4.83x~5.67x）**高于**仅前向的（4.4x）——torch_npu 侧的反向要穿过它那张 python 循环图，被 launch 开销支配得更厉害。② host 侧补检查点 1.065ms，占训练步 21%（bd=1 时只占 7%）。它是 torch 算子，**不随核数缩短**，所以 block_dim 越高占比越大 —— 抬高 block-dim-ceiling 之后这一项才真正凸显（见 fwd-caches-not-emitted）。③ 九个反向 kernel 2.690ms 占 53%，是训练步里最大的一块。④ 门控跨度检查 0.212ms —— 占训练步 4%、占仅前向 16%，可用 check_gate_range=False 关掉，但关掉后越界就是 NaN 而不是报错。


**观察**：torch_npu 基线：数据量从 smoke 到 kimi_linear_layer 差 512 倍，耗时只差 ~15% —— 它完全被 kernel launch 开销支配（向量化后仍有 63 次求逆迭代 + NT 次 chunk 迭代的 python 循环），**不是硬件算力上限**。自编译侧相反：耗时随工作量近线性（T 从 1024 到 4096，bd4 下 1.311→5.138ms，正好 3.9 倍），是真正的算力账。这也解释了 smoke 上 19.7x 的加速 —— 那里 torch_npu 在付固定开销而我们不付。

**跨 CANN 版本一致性**：kda_fwd 经 runtime 桥在 CANN 9.1.0 与 9.2.0 两台机器上的 relL2 逐位相同（smoke 3.288e-03 / multi_chunk 3.383e-03 / gva 3.359e-03），说明这个偏差来自算子自身的数值路径（见 gaps.json 的 kda-fwd-bwd-dtype-mismatch），与 CANN 版本无关。

**测量注意**：torch_npu 基线的跑间波动约 20%（kimi_linear_layer 在四个子进程里测到 7.920 / 5.445 / 5.439 / 5.811ms），所以加速比带同等量级的不确定度。每行的比值用的是该子进程自己测的基线，不是跨进程平均。

**下一步**：① 抬高 block_dim 上限（gaps.json 的 block-dim-ceiling，已升 P1）—— 扩展性到 4 仍线性，物理上有 28 cube。② kernel 侧的 nd2nz 返工（kernel-nd2nz-suboptimal）。③ 反向的同类测量，第二期随 kda_bwd 一起做。

### 全链路三层

> 全链路的上三层。窄切片原则：按算子倒推，用到哪个做哪个。首个目标是 KDA 链路，其依赖面比 GDN 少一个 module。

> ``done-torch`` 表示功能完成但实现是 torch 原生算子拼的，不是本仓自编译的 kernel —— 见 gaps.json 的 modules-are-torch-not-kernels。``done`` 只给自编译算子。

**modules**

- `causal_conv1d` — 🔶 完成（torch 实现） · none — 需新写
  - 证据：2026-09-11 第二期：ascend_fla/modules/convolution.py ShortConvolution；tests/test_modules.py 纯 CPU 11 项通过（含因果性、归一化与门控的先后顺序、cache 分段与整段一致性等判别性检查）。
- `fused_rms_norm_gated` — 🔶 完成（torch 实现） · partial — matrix_normalization 可借
  - 证据：2026-09-11 第二期：ascend_fla/modules/fused_norm_gated.py FusedRMSNormGated；tests/test_modules.py 纯 CPU 11 项通过（含因果性、归一化与门控的先后顺序、cache 分段与整段一致性等判别性检查）。
- `rms_norm` — ⬜ 未开始 · partial — matrix_normalization 可借
- `l2norm` — 🔶 完成（torch 实现） · matrix_normalization.row_l2
  - 证据：2026-09-11：layers/kda.py 里用 F.normalize 在 fp32 下做，未单列模块。

**layers**

- `kda` — 🔶 完成（torch 实现）
  - 证据：2026-09-11 第二期：ascend_fla/layers/kda.py KimiDeltaAttention，参数名与 fla 逐项对齐（KDA 算子自编译，周边 modules 是 torch —— modules-are-torch-not-kernels）。层级梯度实测：三个形状下输出相对 L2 4.9e-03，全部 17 个参数的梯度在 3.6e-03~2.2e-02，预算 0.1（A_log/dt_bias/f_proj 用 0.25，因为它们的梯度直接由 dg 来）。参考是同一份权重的 CPU 层，只把 KDA 算子换成 fp32 逐 token 递推版。承担了 fla 放在 kernel 里的三件事（q/k 的 l2norm、门控变换、beta sigmoid）。默认初始化（跨度 ~94）另有两项：前向对递推 oracle 相对 L2 4.697e-03（已测）；整层反向的有限性与精度由 test_default_init_backward_matches_cpu_reference 盯，**真机未跑**（反向的 kda_bwd_stable 还没在真机比对过，见 bwd-gate-range-overflow）。梯度对齐那三个形状是在 exp(A_log)=1 下测的 —— 为的是把「接线对不对」和「深衰减下 bf16 本来就糙」分开，不是因为默认初始化跑不了。decode 路径未接（fused-recurrent-missing）。
- `gated_deltanet` — ⬜ 未开始

**models**

- `kimi-linear` — ⬜ 未开始 · 注入 — 用上游模型定义，替换 linear attention layer
- `qwen3-next` — ⬜ 未开始 · 注入 — 用 HF transformers 的模型定义，替换 linear attention layer

## 缺口

P0 0 项 · P1 12 项 · P2 11 项 · 已解决 9 项 · 共 32 项

**第一期里程碑**：第一期五项已全部有结论，并补齐了同机性能对比：aclnn 编译、runtime 桥、kda_fwd 接线、KDA 本地基线均实测通过；自编译算子在 block_dim=4 下比 torch_npu 组合快 4.43x（kimi_linear_layer）/ 2.38x（long_context T=4096）/ 19.7x（smoke）。过程中修掉两个自己的 bug（bridge-per-call-overhead、op-name-collision-in-process），它们先后让 block_dim 的效果被完全掩盖。当前最大的性能项是 block-dim-ceiling（已升 P1）：扩展性一路线性到契约上限 4，而硬件有 28 cube。第二期的前置障碍 kda-fwd-bwd-dtype-mismatch 已量化（降 P2）。

**第二期里程碑**：第二期已完成：kda_bwd 九 kernel 链 + 九个前向检查点 + autograd + KDA layer（含 2 modules），训练步在 bd=4 下比 torch_npu 组合版快 4.83x~5.67x。
过程中发现并根治了第二期最关键的一个缺口：**按 fla 的默认初始化（chunk 内门控跨度 ~94），上游 kernel 在前向与反向各有一处量程失效，方向相反** —— 前向在 87.3 下溢、反向在 88.72 上溢。两处都在本仓建了派生单元（kda_fwd_stable / kda_bwd_stable），把成对衰减分解的锚点从区间端点改到中点，数学同义。
门控闸因此做成二维 `{impl: {forward, backward}}`：前向 155（受有限性约束，到 155.97 精度完全不退化）、反向 100（受**精度**约束，它先于有限性到来 —— 梯度到 169.8 都有限但 dq 在 130 处超预算）。**「不吐 NaN」不等于「能用」，两者要分别测、闸按更严的定。**
剩一项端到端确认（默认初始化下整层反向的精度）卡在机器可用性上 —— 算子层面的同一件事已由 test_kda_bwd_deep_npu.py 在跨度 94 测过，六项梯度全在契约预算内。
当前最大的性能项仍是 block-dim-ceiling（P1）。

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
| KDA | — | `fused-recurrent-missing`<br>`no-varlen`<br>`no-tail-path`<br>`block-dim-ceiling`<br>`qk-l2norm-not-in-kernel`<br>`state-layout-k-first` | `kda-fwd-bwd-dtype-mismatch`<br>`npu-builtin-ops-missing`<br>`toy-case-shapes`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`torch-npu-baseline-missing`<br>`kernel-nd2nz-suboptimal`<br>`fwd-caches-not-emitted`<br>`modules-are-torch-not-kernels`<br>`stable-unit-no-harness`<br>`gate-span-still-bounded` |
| GDN | — | `gdn-no-gqa`<br>`layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`state-dtype-bf16`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path`<br>`block-dim-ceiling` | `npu-builtin-ops-missing`<br>`toy-case-shapes`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`torch-npu-baseline-missing` |
| DeltaNet | — | `layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path` | `npu-builtin-ops-missing`<br>`toy-case-shapes`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`torch-npu-baseline-missing` |

### P1

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

#### `qk-l2norm-not-in-kernel` — fla 的 KDA 在 kernel 内做 q/k 的 L2 归一化、门控变换与 beta sigmoid，ascriptor 的不做

- **类别** abi · **适用于** KDA · **阻塞** —
- **依据** fla 的 KimiDeltaAttention 调 chunk_kda 时传 use_qk_l2norm_in_kernel=True、use_gate_in_kernel=True、use_beta_sigmoid_in_kernel=True，即三步都在它的 kernel 里：g 的变换是 -exp(A_log) * softplus(g + dt_bias.view(HV,K))，beta 过 sigmoid，q/k 沿头维 L2 归一化。ascriptor 的 kda_fwd/kda_bwd 的 inlet 只收已经变换好的 q/k/g/beta，contract 里没有 A_log / dt_bias 这两个入口。
- **影响** **门控挡不住这一条** —— 没做归一化的 q/k 在数值上完全合法，算子会照算并给出一个静静地错的结果。这是本仓目前唯一"错了不报错"的语义缺口，因此列 P1。另外这三步的梯度也落在调用方这边，autograd 链要从层级算起。
- **建议** 当前处置：ascend_fla/layers/kda.py 显式做这三步并在 fp32 下做，ops/kda 的 docstring 与 __init__ 写明"调用方需已做"。直接调算子的人要自己负责。若要彻底消除风险，得在 ascriptor 侧给 kernel 加 A_log/dt_bias 入口与 l2norm —— 那是第四期的事，收益还包括省掉几趟 elementwise 的访存。

#### `state-layout-k-first` — 我们的 state 是 [B,HV,K,V]，fla 的 KDA layer 用 state_v_first=True

- **类别** abi · **适用于** KDA · **阻塞** `phase 3`
- **依据** ascriptor kda_fwd 的 initial_state / final_state 均为 (B,HV,128,128) 且 K 在前（kernel 内 next_state = state * exp2(g_last)[:,None] + kg^T @ v_new，按 K 维缩放行）。fla 的 KimiDeltaAttention 调 chunk_kda 与 fused_recurrent_kda 时都传 state_v_first=True。
- **影响** 第三期把本仓的 layer 注入 Kimi-Linear 时，与上游 cache 交接要转置，否则 decode 第一步就会用错的状态起算。K=V=128 让形状相同，**转置错了不会报形状错** —— 又是一个静默失败面。
- **建议** 第三期在注入层里做转置并加一个显式断言（比如用非对称测试值验证方向）。不要在算子里改布局 —— 算子的布局由 kernel 决定，改它等于改 kernel。

### P2

#### `kda-fwd-bwd-dtype-mismatch` — kda 的 fwd 与 bwd 对同名张量声明了不同 dtype

- **类别** abi · **适用于** KDA · **阻塞** —
- **依据** kda_fwd inputs：beta float32、initial_state float32、g_raw float32，final_state 输出 float32。kda_bwd inputs：beta bfloat16、initial_state bfloat16、g bfloat16、dht bfloat16。 | 2026-09-11 量化（benchmarks/quantify_bwd_dtype_mismatch.py，CPU fp32 参考 + autograd，四个形状 smoke/multi_chunk/gva/kimi_shaped 结论一致）。把三个效应拆开，基准为全程 fp32：A 保存值走 bf16 往返（bwd kernel 实际看到的输入）：dq/dk/dv/dg 相对 L2 1.4e-03~1.6e-03，dbeta 1.6e-04，dinitial_state 3.3e-04。B 梯度输出舍到 bf16（bwd ABI 的输出精度）：一律 ~1.65e-03。C 两者叠加：2.17e-03~2.37e-03，且 C ≈ √(A²+B²)（1.633²+1.686² → 2.347 vs 实测 2.370）—— 两个效应相互独立，没有病态放大。
- **影响** **降精度不免费，但与契约已声明的输出精度同量级。** B（输出舍到 bf16）是 bwd ABI 规定的，无论如何都要付；A（保存值降精度）把总误差从 1.65e-03 的地板抬到 2.3e-03，即 ×1.4。没有引入新性质的误差，因此第二期可以按 .to(bfloat16) 组装 autograd，把这个数记在案。注意这是**算子级**相对 L2，按 AGENTS.md §6 的纪律不能外推成任务级精度结论 —— 要声称对训练的影响，得跑任务级实验并拆出"替换实现"与"改精度"两个对照组。
- **建议** 已量化，不再阻塞第二期。组装 autograd 时照 bwd ABI 降到 bf16，但**显式**做、在 docstring 里写明代价，不要当成无害的类型适配。若后续任务级实验显示不可接受，再回头推动 bwd ABI 接受 FP32（那是 ascriptor 侧的改动）。

#### `npu-builtin-ops-missing` — 内置算子包的覆盖随机器而异：部分 Ascend950PR 机器上 torch_npu 的计算算子不可用

- **类别** environment · **适用于** 全部 · **阻塞** —
- **依据** CANN 9.1.0 的那台 Ascend950PR（见 machine_specs.md）上 $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ 只有 ascend910_93 与 ascend910b 两个 SoC 目录。torch.randn(device='npu') 报 aclnnInplaceNormal_1_StatelessNormalAiCore 找不到 JSON 配置；torch.zeros、bf16->fp32 Cast 同样失败。torch 本身是 2.10.0+cpu。 【2026-09-11 补充】另一台 8 卡 Ascend950PR 机器（CANN 9.2.0，innerversion V100R001C25B046）的 opp 下有 **ascend950** 算子包，SoC 报 Ascend950PR_9579，实测 randn / zeros / fp32+bf16 matmul / bf16↔fp32 cast / permute+contiguous / einsum / cumsum 全部可用。所以这不是 SoC 级缺陷，而是**算子包安装差异**：CANN 9.1.0 的 opp 只装了 910 系列。
【2026-09-11 再补充】缺算子包时**跨步视图的 D2H 也不可用** —— 它要走 NPU 侧的 `Slice`。最小复现（形状 [1,128,2,8]，对应 C=2 的 g_cumsum）：
  `dev[:, 63::64].cpu()` → RuntimeError：`Op Slice does not has any binary` / `launch failed for Slice, errno:561000`
  `dev.cpu()[:, 63::64]` → 一致
  同一块张量按 C=1 切（`[:, 127::128]`，只取一行）→ 两种写法**都一致**，因为切片等效连续。
这条 C=1/C≥2 的分界正好解释了一次真实失败：`chunk_kda_bwd` 里 `g_last = g_cumsum[:, 63::64]` 绕 CPU 时，single_chunk 与 grouped_idle_cores（都是 C=1）通过，multi_chunk / grouped_heads / gentle_decay（都是 C≥2）全挂。**是硬报错不是静默出错** —— 我最初写成「静默给出错误数据」，最小复现证伪了。修法：先整块 D2H 再在 CPU 上切。
- **影响** 选机器决定能做什么：装了 ascend950 算子包的机器上 torch_npu 基线与 layer 级验证都可做；没装的机器上只能跑自编译 kernel（empty/H2D/D2H/data_ptr/stream 可用，计算算子全不可用）。runtime 桥在两种机器上都工作 —— 这正是它的价值。 【2026-09-11 修正】「runtime 桥在两种机器上都工作」只对**前向**成立。训练路径要在 host 侧补三个反向检查点（`fwd-caches-not-emitted`），那一段是 torch 算子 —— 在缺算子包的机器上原本直接失败。已加 CPU 绕行；层级验证仍然只能在有 ascend950 算子包的机器上做（层里的投影/卷积/softplus/RMSNorm 全是 torch_npu 算子）。 另外：算子入口现在**要求输入连续**并在不满足时报错（`chunk.py` 与 `chunk_bwd.py` 的 `_check`）。在这种机器上「悄悄 .contiguous() 一下」根本做不到 —— device 上要 d2d copy，跨步 D2H 要 Slice，两条都缺，所以只能报错。
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

#### `torch-npu-baseline-missing` — 缺少 torch_npu 组合实现作基线与第二 oracle

- **类别** validation · **适用于** 全部 · **阻塞** `phase 1 验收`, `phase 2 性能报告`
- **依据** 本仓尚无任何实现代码。
- **影响** 没有它就只有一个 oracle（fla naive），也没有性能对照 —— 无法判断"高效率"是否达成。
- **建议** 第一期与 runtime 桥并行做：reference/ 下用 torch_npu 原生算子拼出同语义 KDA，同时充当基线与第二 oracle。

#### `kernel-nd2nz-suboptimal` — kda 的前后向 kernel 里有一批 ascriptor lint 标出的访存低效点（nd2nz 展开、偶数 block stride）

- **类别** performance · **适用于** KDA · **阻塞** —
- **依据** 编译 kda_fwd 时 ascriptor lint 在 intra.py(2)、triangular_inverse.py(8)、wy.py(2)、recurrent.py(3) 共 15 处报同一条 warning：ub_to_l1.nd2nz 会展开成多次 MTE3 burst（每个 NZ fractal 列一次），不是单条指令，板上实测比 compact-NZ move 慢 10 倍（D-084）。lint 还给了第二条：staging tile 的自然 block stride 是 tile 行数（天然 16 对齐，正好是 bank 阶梯最差的一档，~8.7 cycle/store vs ~1.1），补一行 padding 让它变奇数；D-226 里两个返工的 kernel 上这步收益 -12.3us / -10.8us（总量 -17.7us / -28.2us），比换 move 本身更大。 | 反向侧同族线索：scan_fused.py:53 的 snapshot_and_cast_state_vf 报"strided block store with an even block stride (64)" —— 连续 datablock 撞同一组 UB bank、store 口串行化，板上实测比奇数 stride 慢 ~2x、在 16 的倍数上慢 ~8x（同一条 D-084）；处置同样是把目标行补 1 让 stride 变奇数（65）。
- **影响** 这 15 处在 kernel 源码里，不在本仓。按 D-226 的比例，单 kernel 量级的收益在数十微秒 —— 但要先确认设备侧耗时占比（见 bridge-per-call-overhead），占比低的话改了也看不出来。 反向侧的占比尚未测 —— 当前反向的瓶颈在 host 侧补检查点（fwd-caches-not-emitted），要先换掉那一段，kernel 级优化才看得出效果。
- **建议** 第四期。kernel 源码归 ascriptor 仓所有（AGENTS.md §3：只读），所以这里只登记线索，改动要走 ascriptor 侧。动手前先用 benchmarks/profile_bridge_overhead.py 确认设备侧占比足够大，否则是在优化一个不在关键路径上的东西。

#### `fwd-caches-not-emitted` — kda_bwd 要九个前向检查点，前向 kernel 只直接给出六个

- **类别** abi · **适用于** KDA · **阻塞** `phase 2 性能`, `phase 3`
- **依据** kda_bwd 的 unit.py validate_inputs 要求 saved 恰好含九项：g_cumsum/w/u/qg/kg/v_new 为 (B,T,HV,128)、Aqk/Akk 为 (B,T,HV,64)、h 为 (B,C,HV,128,128)，全 bf16、token-major。前向 kernel 直接产出的只有 w/u/qg/kg（kda_sub3_wy_kernel）、Aqk（kda_sub2_score_kernel）、Akk（tril_inverse64_v2_strict_bf16_kernel）六项。gate kernel 只写 eg = 2**g_cumsum，不写 cumsum 本身；kda_sub45_fused_kernel 内部算了逐 chunk 状态 h 与 v_new = u - w @ h，但只写出 o 与 final_state。单元自己是用 CPU 参考 build_saved_forward() 造 saved 的，不是用 kernel。 | **代价已实测**（benchmarks/bench_kda_train_step.py，Ascend950PR / CANN 9.2.0，bf16，warmup 2 / iters 5，同步）：kimi_linear_layer 形状下补检查点 1.053ms（bd=1）/ 1.065ms（bd=4），long_context 4.242 / 4.351ms —— **随 block_dim 基本不变**，因为它是 torch 算子而不是我们的 kernel。于是它的占比随 block_dim 上升：fwd+bwd 的 7%（bd=1）→ 21%（bd=4）。
- **影响** autograd 的前向必须补齐这三项。g_cumsum 可由 log2(eg) 得到（一次 elementwise）；h 与 v_new 只能重跑 chunk 递推，当前在 host 侧用 torch 做（C 次迭代 × 2 次 bmm）。
**修正先前的判断**：我曾写它是"反向链的性能瓶颈"，实测不是 —— kimi_linear_layer / bd=4 下它占 21%，而九个反向 kernel 占 53%（2.690ms / 5.069ms）。它是一笔确定的、值得收的账，但不是主因。**真正要紧的是它不随核数缩短**：block_dim 上限若被抬高（block-dim-ceiling），kernel 侧会继续变快而这一段不会，占比会继续涨。
**2026-09-11 新发现的第二个后果：它让训练路径依赖内置算子包，而纯前向路径不依赖。**`_scan_states` 与检查点的降 bf16 用的是 Cast / bmm / stack，在只装了 910 算子包的机器上全部不可用 —— 实测表现为 `copy_d2d_baseformat_opapi … error code is 561103` + `Cast ADD_TO_LAUNCHER_LIST_AICORE failed`。这推翻了「我们自己编译的 kernel 在两种机器上都不受影响」这句话的适用范围：它对**前向**成立，对**训练**不成立，因为训练要补的三项检查点不在 kernel 里。已加 `on_cpu` 绕行（`_scan_states(on_cpu=)`、`chunk_kda_bwd` 的 `layout_device`），把检查点生产和那一次 strided `contiguous()` 整段搬到 CPU —— 这是**可用性**开关不是性能开关。把三项挪进 kernel 之后这些绕行可以删掉。
**2026-09-11 顺带修掉的一条**：`chunk_kda` 此前**无条件**走 autograd.Function，于是`no_grad` 下的推理也照样产那九个检查点（纯浪费，占训练步的 21%），而且被**反向**那条更严的门控闸（stable 下 100）挡着 —— 推理本来只受前向的 155 约束。现在不需要梯度时直接走 `chunk_kda_fwd`；`o` / `final_state` 逐位相同（共用同一次 kernel 调用），钉在 tests/test_kda_gating.py::test_chunk_kda_skips_caches_when_no_grad_is_needed。
- **建议** 按 AGENTS.md §3 在本仓 kernels/ 下建自己的单元：做一个 kda_sub45_fused_kernel 的变体，额外写出 h 与 v_new（两个 GM 输出 + store，内部量已有），再做一个 gate 变体直接写 g_cumsum。改 ascriptor 仓是不允许的。优先级排在 block-dim-ceiling 之后 —— 先抬核数上限，那一项的收益更大，而且抬完之后这一项的占比才真正凸显。
做完之后顺带删掉 `_scan_states(on_cpu=…)` 与 `chunk_kda_bwd(layout_device=…)` 两处绕行。

#### `modules-are-torch-not-kernels` — modules 层是 torch 原生算子实现，不是本仓自编译的算子

- **类别** performance · **适用于** KDA · **阻塞** —
- **依据** ascend_fla/modules/convolution.py 用 F.conv1d + F.silu；fused_norm_gated.py 用 rsqrt/mean/sigmoid 两趟完成（名字里的 Fused 只为与 fla 对齐，并未融合）。layers/kda.py 里的投影、l2norm、softplus 门控、sigmoid 也都是 torch 算子。
- **影响** ① 这些步骤在内置算子包不全的机器上不可用（需要 conv1d/silu/matmul），而自编译的 kda 算子本身不受影响 —— 所以层级验证比算子级验证对机器挑剔。② 层级耗时里有一部分不归本仓的"高效率算子"管，报层级性能数时必须拆开说，否则会把 torch 的开销算进算子账上。
- **建议** 第四期按测得的占比决定做哪些。归一化与门控是 elementwise，融合收益直接；短卷积是 depthwise，值得单独做一个 kernel。动手前先 profile 层级耗时拆分，别重复 bridge-per-call-overhead 那次"先推断后测量"的错。

#### `stable-unit-no-harness` — 本仓的 kda_fwd_stable 单元还不能用 ascriptor harness 独立跑

- **类别** verification · **适用于** KDA · **阻塞** —
- **依据** kernels/projects/a5/kda_fwd_stable/ 目前有 contract.json、README.md 与三个 kernel 文件，但缺 unit 协议要求的 unit.py（make_inputs/reference/execute）与 run.py —— 它们要对接 ascriptor 的 _unit_runner。现在的验证全部经本仓 runtime 桥 + pytest 做。
- **影响** ① 拿不到 ascriptor harness 的 sim / pipesim / cannsim 几个 stage 的证据，也就用不上它的逐 stage checkpoint 比对（那对定位 kernel 内部错误很有用）。② 这个单元不能被 ascriptor 侧的人独立复现，不利于把修法推回上游。contract.json 的 support 里已如实标注证据来源，没有假装有 harness 证据。
- **建议** 补 unit.py 与 run.py。reference 可以直接用 ascend_fla/reference/kda.py 的逐 token 递推版（它没有跨度上限，正是宽域下唯一可用的 oracle）。做完后把 contract.json 的 support 按 harness 实际结果更新。

#### `gate-span-still-bounded` — 稳定化把门控跨度上限从 80 抬到前向 155 / 反向 100，但没有去掉上限

- **类别** numerics · **适用于** KDA · **阻塞** —
- **依据** `kda_fwd_stable` / `kda_bwd_stable` 走的是**对称分解**：把 `exp(a_i − a_j)` 拆成两个以中点为锚的因子，各压到 ±span/2，有限性的理论上限正好翻倍 —— 前向 `2 × -ln(FLT_MIN_NORMAL) ≈ 174.7`，反向 `2 × ln(BF16_MAX) ≈ 177.4`。
**但两条链的闸不是同一回事，这是实测出来的**：
* 前向的约束是**有限性**。实测跨度到 155.97 时 `o` 的相对 L2 仍稳定在 2.85e-03~3.19e-03，完全不随跨度退化 —— 所以闸就设在实测最深点 155。
* 反向的约束是**精度，而且它先于有限性到来**。梯度到 169.76 都还是有限值，但对 fp32 递推参考的相对 L2 随跨度单调上升：dq 在 46/94/105/110/130/169 处是 2.89e-02 / 4.55e-02 / 4.86e-02 / 4.99e-02 / 6.17e-02 / 6.88e-02，**130 处越过契约预算 0.05**；dg 在 169 处崩到 6.49e-01（预算 0.25）。所以反向的闸设在 100（105 也过，110 只剩 0.2% 余量）。
于是 `MAX_GATE_SPAN` 是二维的：`{impl: {forward, backward}}`，纯推理用前向那条、训练用反向那条。fla 默认初始化的层跨度约 94，两条都满足。
**跨度不随 T 增长** —— 它是 chunk 内（64 token）的量，cumsum 每 chunk 重置。推高它的是 `exp(A_log)` 与 `dt` 的乘积。
- **影响** ① fla 默认初始化给出跨度 ~94，反向的闸 100 只剩 6% 余量。`A_log` 与 `dt_bias` 都是可训练参数，训练中 `dt` 变大就会撞上限 —— 届时是**报错**（设计如此），但会中断训练。要继续得显式 `check_gate_range=False` 并接受超预算的梯度，或者压 `dt`。
② 门控检查默认开，每次前向对 g 做一次 cumsum + 两次规约，实测 0.212ms / 步（bd=4 / kimi_linear_layer，占训练步 4%）。
③ **有限性上限（反向 169.76）比声明上限宽**。需要更深跨度又能接受精度退化的调用方，可以自己关掉检查 —— 曲线在 `kda_bwd_stable/contract.json` 的 `accuracy_vs_span` 里，照着选，不要瞎试。
- **建议** 现在不做。真撞上限时按代价排序：
① **先查 dq 为什么是约束项**。它在跨度 46 时就已用掉 58% 预算，说明深衰减下 `dq` 的主项（`d_qg · exp(g) · scale`，见 inverse_epilogue）对 bf16 的 `g` 最敏感。若把 `g_cumsum` 检查点从 bf16 升到 fp32（kda_bwd 的 ABI 问题，见 kda-fwd-bwd-dtype-mismatch），这条曲线可能整体下移 —— **这是推测，要测**。
② 把 64×64 的 tile 再按行列分块，每对子块用各自的中点（等价于分块 log-sum-exp），有限性上限随分块数线性增长。但若约束是精度而不是有限性，这一项帮不上忙。
③ fla 的 `lower_bound` / `safe_gate`：给门控设下界。那**会改变数学**，属于模型侧决策，不能当数值修补悄悄加上（本仓目前显式拒绝这两个开关）。
