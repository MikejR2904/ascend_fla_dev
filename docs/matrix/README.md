<!-- 由 tools/gen_matrix.py 生成，请勿手改。改 docs/matrix/*.json 后重新运行。 -->

# 支持矩阵

记录于 2026-09-10。本文件由 `docs/matrix/*.json` 生成。状态词汇沿用 ascriptor：`passed` / `untested` / `gap` / `failed`。

## 目标模型形状

ascriptor A5 定尺 ABI：`[B, H, C, L, D]`，L=64，D=128，q/k/v `bfloat16`，beta/g `float32`。

| 模型 | 算子族 | 优先级 | H | HV | head_k | head_v | dtype | 定尺匹配 | 阻塞缺口 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B-Instruct | gated_delta_rule | primary | 16 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ❌ | `gdn-no-gqa` |
| Kimi-Linear-48B-A3B-Instruct | kda | secondary | 32 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ✅ | — |
| fla GatedDeltaNetConfig 默认值 | gated_delta_rule | reference-only | 6 | 6 | 256 | 512 | — | ❌ ❌ ✅ | `fixed-kv-128`, `asymmetric-kv-dim` |
| fla KDAConfig 默认值 | kda | reference-only | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |
| fla DeltaNetConfig 默认值 | delta_rule | reference-only | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |

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
| `a5.kda_fwd` | kda | forward | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.kda_bwd` | kda | backward | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.delta_rule_fwd` | delta_rule | forward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |
| `a5.delta_rule_bwd` | delta_rule | backward | ✅ | ✅ | ✅ | ✅ | ⬜ | ✅ | ⬜ 未开始 |

> **compile** 一列是本仓 runtime 桥依赖的本地 CANN 编译路径 —— 全部 `untested`，这是第一期的首个里程碑（见 `gaps.json` 的 `aclnn-compile-untested`）。真机 `board` 全部 passed，但走的是 SSH 远端编译，不是同一条路。

### 缺失的算子

- **gdn_fused_recurrent**（gated_delta_rule）— decode 路径（逐 token 递推 + state 传递）。完全缺失，需要新写，不是接线。没有它就只有 prefill/训练，没有推理解码。
- **kda_fused_recurrent**（kda）— decode 路径。同上。

### 可复用原语

| 原语 | 对应 fla | 本仓状态 |
|---|---|---|
| `chunk_row_scan` | fla 的 chunk cumsum（ops/utils/cumsum.py 的 chunk_local_cumsum） | ⬜ 未开始 |
| `matrix_normalization.row_l2` | fla.modules.l2norm | ⬜ 未开始 |
| `gated_approximations` | fla.modules.activations / fused swiglu | ⬜ 未开始 |

### 全链路三层

> 全链路的上三层。窄切片原则：按算子倒推，用到哪个做哪个。

**modules**

- `causal_conv1d` — ⬜ 未开始 · none
- `rms_norm` — ⬜ 未开始 · partial — matrix_normalization 可借
- `fused_rms_norm_gated` — ⬜ 未开始 · partial
- `l2norm` — ⬜ 未开始 · matrix_normalization.row_l2

**layers**

- `gated_deltanet` — ⬜ 未开始
- `kda` — ⬜ 未开始

**models**

- `qwen3-next` — ⬜ 未开始 · 注入 — 用 HF transformers 的模型定义，替换 linear attention layer
- `kimi-linear` — ⬜ 未开始 · 注入

## 缺口

P0 3 项 · P1 7 项 · P2 7 项 · 共 17 项

**首个里程碑**：aclnn-compile-untested —— 在接任何线性注意力算子之前，先证明本地 aclnn 编译路径可用。

**建议的首个目标**：Kimi-Linear / fla-kda-default 的形状（H==HV，K=V=128），因为它们无阻塞缺口；Qwen3-Next 受 gdn-no-gqa 阻塞，留到第三期。

### P0

#### `aclnn-compile-untested` — runtime 桥依赖的本地 aclnn 编译路径从未验证

- **类别** runtime · **阻塞** `phase 1`
- **依据** 六个 a5 单元的 contract.json support 列表中 compile 与 cannsim stage 全部 untested；真机 passed 的是 board stage，走 SSH 推送 + 远端编译。
- **影响** 本仓 runtime/compile.py 的整个技术路线建立在一条未验证的路径上。若 aclnn 本地编译不通，第一期方案需要重新设计。
- **建议** 第一期的第一个里程碑：取最简单的单元（如 chunk_row_scan）走通 ascriptor 的 aclnn launcher 本地编译，产出 custom_opp_*.run 并安装为 vendor。在接任何线性注意力算子之前完成。

#### `runtime-bridge-missing` — ascriptor 无进程内 device tensor 调用能力

- **类别** runtime · **阻塞** `phase 1`, `phase 2`, `phase 3`
- **依据** ascriptor/runtime/opexec.py 的 __call__：aclnn 路径经 write_args 写二进制参数文件并跑独立 test_aclnnop；board/pypto 经 SSH 推送；返回值用 torch.frombuffer 重建 CPU tensor。
- **影响** 每次调用都有落盘与进程启动开销，且输出在 CPU 上 —— 无法作为训练/推理中的算子使用。
- **建议** 自建 runtime/{compile,cache,binding,autograd}.py：把生成的 CANN 自定义算子编译成常驻 .so，经 torch.library 注册，直吃 NPU device tensor。地基是 ascriptor 的 runtime/aclnn/template 与 build_custom_op()。

#### `gdn-no-gqa` — gdn_fwd/bwd 无独立 value-head 维度，不支持 GQA 分组

- **类别** abi · **阻塞** `qwen3-next-80b-a3b`
- **依据** gdn_fwd contract.json domain.shape："B,H,C are positive runtime dimensions" —— 没有 HV。对比 kda_fwd domain 有 "H_HV": "positive; HV % H == 0"。
- **影响** 直接阻塞首要目标模型 Qwen3-Next（16 key heads / 32 value heads）。
- **建议** 两条路：① 借鉴 kda_fwd 已有的 HV%H==0 实现，给 GDN kernel 加分组维度；② 第一期先用 H==HV 的形状（Kimi-Linear、fla 默认 KDA/DeltaNet）打通全链路，把 GQA 放到第三期。建议先走 ②，避免第一期同时扛 runtime 桥与 kernel 改写两个风险。

### P1

#### `gdn-fused-recurrent-missing` — decode 路径算子完全缺失

- **类别** coverage · **阻塞** `phase 3`
- **依据** ascriptor kernels catalog 中无 fused_recurrent 类单元；现有 gdn/kda/delta_rule 单元均为 chunk 路径。
- **影响** 只有训练与 prefill，没有推理解码。也失去了 chunk↔recurrent 互验这个最好的 oracle。
- **建议** 第三期新写（非接线）。写之前先用 fla 的 fused_recurrent 作语义基准。

#### `nonzero-initial-state` — gdn 与 delta_rule 只支持零初始 state

- **类别** abi · **阻塞** `phase 3`
- **依据** gdn_fwd / delta_rule_fwd contract.json domain.initial_state: "zero only"。kda_fwd 则支持 random（case 中有 initial_state: random）。
- **影响** 无法做 state 传递 —— 长序列分段训练、prefill→decode 交接、chunked prefill 都做不了。
- **建议** 借 kda_fwd 的实现方式给 gdn 加非零初始 state 入口。在此之前，门控必须对传入非零 initial_state 的调用报错。

#### `d-initial-state-absent` — backward ABI 不产出初始 state 的梯度

- **类别** abi · **阻塞** —
- **依据** gdn_bwd / delta_rule_bwd contract.json："no d_initial_state in the preserved backward production ABI" / "no trainable initial-state gradient in the production backward"。
- **影响** 可训练初始 state、序列并行（CP）、分段反向传播都不可用。
- **建议** 记录为已知限制。若训练场景不需要可训练初始 state，则不必修；需要时再扩 ABI。门控应在 initial_state.requires_grad 时报错。

#### `no-varlen` — 无变长序列（cu_seqlens）支持

- **类别** coverage · **阻塞** —
- **依据** ascriptor 各单元 ABI 均为规整的 [B,H,C,L,D]，contract 中无 cu_seqlens / varlen 概念。fla 的公共入口普遍带 cu_seqlens 与 chunk_indices。
- **影响** 训练常用的 sequence packing 无法使用，等长 padding 会浪费算力。
- **建议** 列为开放问题（见 docs/plan.md §7），代价与收益待评估。短期门控拒绝。

#### `scale-param-no-slot` — fla 的 scale 参数在 ascriptor ABI 中无入口

- **类别** abi · **阻塞** `phase 1 精度对齐`
- **依据** gdn_fwd contract 的 inputs 只有 query/key/value/beta/g，无 scale 标量。case parameters 里的 "scale": 0.05 是输入生成幅度（见 delta_rule_fwd domain.input_values: "generated q/k/v scale 0.05"），不是算子参数。kernels/recurrent.py 里的 scale*_reg 是 exp(g) 衰减，与此无关。
- **影响** fla 语义下 q 要乘 scale（默认 head_dim**-0.5 ≈ 0.0884）。无入口则只能 host 侧预乘，多一次 elementwise 全量遍历，与性能目标冲突。
- **建议** 优先在 kernel 内吸收 scale（preprocess 或 scores 阶段已有 q 的读取点，可顺带乘）。第一期若先用 host 预乘打通，必须在性能报告中标注这部分开销。

#### `state-dtype-bf16` — final_state 为 BF16，fla 惯例为 FP32

- **类别** precision · **阻塞** —
- **依据** gdn_fwd contract outputs.final_state: bfloat16 [B,H,128,128]。
- **影响** state 是跨 chunk 累积量，BF16 存储的误差会进入下一段递推。影响幅度未在 A5 上测过。
- **建议** 第一期就测：同形状下 BF16 state 与 FP32 state 的输出差异，长 C（如 C=64）下是否放大。不要沿用 A2 上的结论。

#### `no-tail-path` — L=64 固定且无 tail 路径，T 必须是 64 的整数倍

- **类别** abi · **阻塞** —
- **依据** 各单元 domain.tails: "No L/D tails" / "Full 64x128 tiles"。
- **影响** 任意序列长度的推理与训练都需要 host 侧 padding，或拒绝。
- **建议** 门控显式报错并给出最近的合法 T。padding 方案要在性能报告中算进开销。

### P2

#### `fixed-kv-128` — K=V=128 固定，不支持其他 head_dim

- **类别** coverage · **阻塞** `fla-gated-deltanet-default`
- **依据** 各单元 domain 固定 D=128 / K=128 V=128。
- **影响** fla GatedDeltaNetConfig 默认值（head_k=256 / head_v=512）不被支持。
- **建议** 不去支持它 —— 那是 fla 的参考默认值，不是落地模型规格。两个真实目标模型（Qwen3-Next、Kimi-Linear）都是 128。门控明确拒绝并在错误信息中说明。

#### `asymmetric-kv-dim` — K 与 V 共用同一个 D 维，不支持 head_k != head_v

- **类别** abi · **阻塞** `fla-gated-deltanet-default`
- **依据** gdn_fwd contract inputs：query/key/value 三者同为 [B,H,C,64,128]，domain 只声明单一 D=128。
- **影响** expand_v != 1.0 的配置不被支持（fla GatedDeltaNetConfig 默认 expand_v=2.0 → head_v 是 head_k 的两倍）。
- **建议** 与 fixed-kv-128 同样处置：两个真实目标模型都是 head_k == head_v == 128，不去支持非对称维度。门控拒绝并说明。

#### `toy-case-shapes` — 现有算子 case 全是玩具形状，未在真实模型形状上验证

- **类别** validation · **阻塞** `phase 1 验收`
- **依据** gdn_fwd 四个 case 为 B=1, H=1~3, C=1~3；kda_fwd 为 B=1, H=1, HV=1~2, C=1~2。
- **影响** 正确性证据的形状覆盖与真实负载相距很远（Qwen3-Next 单层为 H=16/HV=32，T=1024 时 C=16）。多核分区、UB 压力、尾块路径在玩具形状下可能根本没被触发。
- **建议** 第一期的精度验收必须用 models.json 的 test_case_shapes.qwen3_next_layer / kimi_linear_layer，而不是沿用上游 case。

#### `gdn-block-dim-ceiling` — gdn 的 block_dim 上限为 2，远低于 delta_rule 的 32

- **类别** performance · **阻塞** —
- **依据** gdn_fwd / gdn_bwd domain.block_dim: [1, 2]；delta_rule_fwd domain.block_dim: {min: 1, max: 32}。
- **影响** 多核扩展性可能成为 GDN 的性能天花板。A5 的核数远大于 2。
- **建议** 第一期测到性能数后再判断这是声明限制还是实现限制。若是后者，是第四期优化的首要目标。

#### `kda-no-local-evidence` — kda_fwd / kda_bwd 没有本地验证证据

- **类别** validation · **阻塞** —
- **依据** 两个单元目录下无 validation.json；contract support 中 reference/sim/pipesim/emit 全 untested，仅 board passed（证据为 kernels/docs/migration/fragments/a5-three-backends-20260907.json）。kda_bwd 的 pypto_pro 为 gap。
- **影响** 复用 KDA 时没有本地可复现的精度基线，出问题只能上真机查。
- **建议** 复用前先在本地补跑 reference 与 sim（ascriptor 的 run.py 即可，无需 CANN）。

#### `modules-layer-missing` — modules 层的三个必需组件全缺

- **类别** coverage · **阻塞** `phase 2`
- **依据** fla.layers.gated_deltanet 依赖 ShortConvolution/causal_conv1d、RMSNorm、FusedRMSNormGated；ascriptor 侧只有 matrix_normalization 与 gated_approximations 可部分借用，没有 causal_conv1d。
- **影响** 没有这三个，layer 层无法组装，端到端梯度检验做不了。
- **建议** 第二期做窄切片。causal_conv1d 需要新写；两个 norm 可从 matrix_normalization 的 row_l2 路径扩展。第一期可先用 torch 实现占位（但要在矩阵里标明是 torch 而非自有 kernel）。

#### `torch-npu-baseline-missing` — 缺少 torch_npu 组合实现作基线与第二 oracle

- **类别** validation · **阻塞** `phase 1 验收`, `phase 2 性能报告`
- **依据** 本仓尚无任何代码。
- **影响** 没有它就只有一个 oracle（fla naive），也没有性能对照 —— 无法判断"高效率"是否达成。
- **建议** 第一期与 runtime 桥并行做：reference/ 下用 torch_npu 原生算子拼出同语义 GDN，同时充当基线与第二 oracle。
