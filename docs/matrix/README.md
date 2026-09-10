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

> **compile** 一列是本仓 runtime 桥依赖的本地 CANN 编译路径 —— 全部 `untested`，这是第一期的首个里程碑（见 `gaps.json` 的 `aclnn-compile-untested`）。真机 `board` 全部 passed，但走的是 SSH 远端编译，不是同一条路。

### 缺失的算子

- **kda_fused_recurrent**（kda）— decode 路径（逐 token 递推 + state 传递）。完全缺失，需要新写，不是接线。没有它就只有 prefill/训练，没有推理解码，也失去 chunk↔recurrent 互验这个最好的 oracle。第三期目标。
- **gdn_fused_recurrent**（gated_delta_rule）— decode 路径。同上。随 GDN 扩族（第四期）再补。

### 可复用原语

| 原语 | 对应 fla | 本仓状态 |
|---|---|---|
| `chunk_row_scan` | fla 的 chunk cumsum（ops/utils/cumsum.py 的 chunk_local_cumsum） | ⬜ 未开始 |
| `matrix_normalization.row_l2` | fla.modules.l2norm | ⬜ 未开始 |
| `gated_approximations` | fla.modules.activations / fused swiglu | ⬜ 未开始 |

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

P0 1 项 · P1 10 项 · P2 6 项 · 共 20 项

**首个里程碑**：已达成：aclnn 本地编译 + runtime 桥均在 Ascend950PR 上实测通过（2026-09-11）。当前首要障碍是 npu-builtin-ops-missing —— 它阻塞性能基线与第二期 layer 验证。

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
| KDA | `npu-builtin-ops-missing` | `kda-fwd-bwd-dtype-mismatch`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`no-tail-path` | `toy-case-shapes`<br>`block-dim-ceiling`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`modules-layer-missing`<br>`torch-npu-baseline-missing` |
| GDN | `npu-builtin-ops-missing` | `gdn-no-gqa`<br>`layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`state-dtype-bf16`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path` | `toy-case-shapes`<br>`block-dim-ceiling`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`modules-layer-missing`<br>`torch-npu-baseline-missing` |
| DeltaNet | `npu-builtin-ops-missing` | `layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path` | `toy-case-shapes`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`torch-npu-baseline-missing` |

### P0

#### `npu-builtin-ops-missing` — CANN 的内置算子包不覆盖 Ascend950PR，torch_npu 的计算算子全不可用

- **类别** environment · **适用于** 全部 · **阻塞** `phase 1 ⑤`, `phase 2 layer 级验证`
- **依据** 238（Ascend950PR_957b / CANN 9.1.0）上 $ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/kernel/ 只有 ascend910_93 与 ascend910b 两个 SoC 目录。torch.randn(device='npu') 报 aclnnInplaceNormal_1_StatelessNormalAiCore 找不到 JSON 配置；torch.zeros、bf16->fp32 Cast 同样失败。torch 本身是 2.10.0+cpu。
- **影响** ⑤ torch_npu 组合基线无法在这台机器上跑（需要 matmul/einsum 等内置算子）。第二期的 layer 级验证同样受阻 —— nn.Linear / norm / conv 都依赖内置算子。实测可用的只有：torch.empty、H2D/D2H 拷贝、data_ptr、current_stream —— 这恰好够 runtime 桥用，自编译的 kernel 不受影响。
- **建议** 三条路：① 性能基线改用 ascriptor 自己的 profile 子命令 + 自编译 kernel 之间的对比；② 在有完整算子包的机器上做 torch_npu 基线（a2/910B3 有 ascend910b）；③ 确认是否存在 950PR 的算子包可安装。选哪条取决于基线要回答的问题 —— 要对比 ascriptor vs torch_npu 就必须有内置算子，换机器是最直接的。

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

### P2

#### `toy-case-shapes` — 现有算子 case 全是玩具形状，未在真实模型形状上验证

- **类别** validation · **适用于** 全部 · **阻塞** `phase 1 验收`
- **依据** kda_fwd 四个 case 为 B=1, H=1, HV=1~2, C=1~2；gdn_fwd 为 B=1, H=1~3, C=1~3。
- **影响** 正确性证据的形状覆盖与真实负载相距很远（Kimi-Linear 单层为 H=HV=32，T=1024 时 C=16）。多核分区、UB 压力、核归属路径在玩具形状下可能根本没被触发。
- **建议** 第一期的精度验收必须用 models.json 的 test_case_shapes.kimi_linear_layer，而不是沿用上游 case。

#### `block-dim-ceiling` — kda 的 block_dim 上限为 4，gdn 为 2，均远低于 delta_rule 的 32

- **类别** performance · **适用于** KDA / GDN · **阻塞** —
- **依据** kda_fwd / kda_bwd domain.block_dim: [1,2,3,4]；gdn domain.block_dim: [1,2]；delta_rule_fwd domain.block_dim: {min:1, max:32}。
- **影响** 多核扩展性可能成为性能天花板。A5 的核数远大于 4。
- **建议** 第一期测到性能数后判断这是声明限制还是实现限制。若是后者，是第四期优化的首要目标。

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
