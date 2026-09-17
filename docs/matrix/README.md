<!-- 由 tools/gen_matrix.py 生成，请勿手改。改 docs/matrix/*.json 后重新运行。 -->

# 支持矩阵

记录于 2026-09-14。本文件由 `docs/matrix/*.json` 生成。状态词汇沿用 ascriptor：`passed` / `untested` / `gap` / `failed`。

## 目标模型形状

ascriptor A5 定尺 ABI：`[B, H, C, L, D]`，L=64，D=128，q/k/v `bfloat16`，beta/g `float32`。

| 模型 | 算子族 | 优先级 | 目标期 | H | HV | head_k | head_v | dtype | 定尺匹配 | 阻塞缺口 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B-Instruct | gated_delta_rule | secondary | 第 4 期 | 16 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ❌ | `gdn-no-gqa` |
| GDN-2 1.3B FineWeb-Edu 100B（95B checkpoint） | gdn2 | investigation | — | 16 | 16 | 128 | 128 | bfloat16 | ✅ ✅ ✅ | `gdn2-chunk-gate-range` |
| Kimi-Linear-48B-A3B-Instruct | kda | primary | 第 1 期 | 32 | 32 | 128 | 128 | bfloat16 | ✅ ✅ ✅ | — |
| fla GatedDeltaNetConfig 默认值 | gated_delta_rule | reference-only | — | 6 | 6 | 256 | 512 | — | ❌ ❌ ✅ | `fixed-kv-128`, `asymmetric-kv-dim` |
| fla KDAConfig 默认值 | kda | reference-only | — | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |
| fla DeltaNetConfig 默认值 | delta_rule | reference-only | — | 16 | 16 | 128 | 128 | — | ✅ ✅ ✅ | — |

定尺匹配三格依次为 head_k / head_v / head 分组。

### 算子测试应覆盖的形状

- **smoke** — B=1, H=1, C=1 · ascriptor 现有 case 的规模，仅用于接线冒烟
- **qwen3_next_layer** — B=1, H=16, HV=32, C=16, T=1024 · 单层真实形状，第一期精度验收目标
- **gdn2_1_3b_layer** — B=1, H=16, HV=16, C=16, T=1024 · GDN-2 1.3B 单层真实形状；输入槽语义与现有 a5.gdn_fwd 不同
- **kimi_linear_layer** — B=1, H=32, HV=32, C=16, T=1024 · KDA 单层真实形状
- **long_context** — B=1, H=16, HV=32, C=64, T=4096 · 覆盖 chunk 边界与 state 传递，不是为了测误差累积

> 算子测试矩阵应覆盖的形状。T 必须是 64 的倍数（L=64 无 tail 路径），C = T / 64。

### 待核实的内部规格

- **qwen3.5-9b**（gated_delta_rule）：32 heads / head_dim 128 / chunk 64, bf16, 24 layers — 规格需从权威 config 核实后再补入 models[]；当前仅作参考，不要当作已验证形状。

## 算子支持状态

ascriptor pin：`0.1.0.dev1` · library `77619116f9b3` · 支持硬件 a5 · deferred a2, a3

> ⚠️ 这个 library 修订 **unreachable** —— 2026-09-17 在权威来源上核过：该对象不存在（git cat-file 取不到）。本条自建仓 346cd8a 起未改动过，且从未被验证。推测上游历史重建过（版本号 0.1.0.dev1 → 0.1.0）。后果：第一、二期在 A5 上的精度数字目前无法按此 pin 复现 —— 数字本身有效，但『在哪个编译器修订上测的』这一维已经断了。重新钉 pin 需要所有者指认与之对应的修订，或接受在新 pin 上重跑回归。

> 动手用 `agent/compatibility.json` 的 pin（2026-09-17 读取）：release `0.1.0` · library `90cfcdc720bb` · kernels `b3b3f9c16df7`。来源 https://gitcode.com/ddddwe/ascriptor.git / https://gitcode.com/ddddwe/ascriptor-kernels.git / https://gitcode.com/ddddwe/ascriptor-agent.git

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
| `a5.kda_fused_recurrent` | kda | forward | ✅ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_fused_recurrent` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_fused_decode` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_short_conv_decode` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ 完成 |
| `a5.gdn2_norm2_w12_swiglu` | gdn2 | forward | ✅ | ✅ | ✅ | ✅ | ⬜ | ⬜ | done-explicit-opt-in |

★ 标记第一期的首个目标。

> **compile** 一列指 `ascriptor compile` CLI（纯源码发射+编译、不执行），**不是** aclnn launcher —— unit runner 把 board / aclnn / pypto 都记为 `board` stage。本仓的 aclnn 本地编译与零拷贝调用已独立实测通过，见下方 `our_runtime_bridge`。

### 缺失的算子

- **kda_fused_recurrent**（kda）— decode 路径（逐 token 递推 + state 传递）。**KDA 那半已经做完**（2026-09-11）：本仓自写 `a5.kda_fused_recurrent` 并接进 `layers/kda.py`，prefill/decode 一致性验过（见该条目的 our_status）。剩下的是 GDN / DeltaNet 的 decode（随各自扩族，第四期），以及 decode 的性能 —— 整层一步 458µs、瓶颈在层侧不在算子，见 gaps.json 的 decode-layer-overhead。chunk↔recurrent 互验这个 oracle 现在 KDA 上**已经有了**。
- **gdn_fused_recurrent**（gated_delta_rule）— decode 路径。同上。随 GDN 扩族（第四期）再补。
- **gdn2_chunk_fwd_bwd**（gdn2）— GDN-2 训练与长 prefill；channel-wise erase/write/decay。形状 K=V=128、H=HV=16 可复用定尺与分块经验，但输入必须新增长度 K 的 b/g 和长度 V 的 w；见 gdn2-abi-not-gdn。数值算法不能照搬 KDA：真实权重的有效 T=4096 stress case 已观测 64-token 累计衰减跨度 1461，见 gdn2-chunk-gate-range。

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


#### GDN-2 packed-inference

> 真实 95B checkpoint，B=1/H=HV=16/K=V=128，BF16；prompt T=6，decode T=1；warmup=100、iters=200、每轮末同步；host 打包后 H2D，canonical/packed 顺序加载并用相反顺序复测。绝对延迟对加载顺序敏感，因此报区间并取较小加速比作保守值

环境：Ascend950PR / CANN 9.1.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 · 记录于 2026-09-14

| 加载顺序 | canonical (ms/token) | packed (ms/token) | 加速 |
|---|---:|---:|---:|
| canonical-first | 11.625 | 9.822 | 1.184x |
| packed-first | 10.184 | 9.212 | 1.106x |

**数值一致性**：正式闸为同 dtype relative-L2 + cache offset：FP32 T=6 预算 1e-5，BF16 T=6 与确定性 T=64 预算 1e-2；prompt/下一 token logits、18 层 recurrent state、canonical 三份 conv cache 拼接 vs packed cache 均 passed。bitwise 与 argmax 仅作诊断。参数数均为 1,450,096,416；canonical state_dict 399 项，packed 在加入SwiGLU W1/W2 packing后为255项（此前只打包mixer时273项）。最新原生RMSNorm+W1/W2版本的真实BF16 canonical↔packed为prompt/step logits relative-L2=1.252e-3/3.937e-3，最差step recurrent/conv cache=7.239e-3/4.282e-3，仍过1e-2。另有 tiny FP32 CPU↔torch_npu数值测试，最大relative-L2=5.740e-4（预算1e-3）。

**端到端冒烟**：最终 host-packed loader 以同 prompt 与 greedy 配置生成预期的 32 token；单次 0.502s（63.7 tok/s）。这里含首次执行影响，只作端到端 smoke，不作为稳态加速比。

**打包边界**：q/k/v/b/w 五个无 bias 投影按输出行合成一次 F.linear；q/k/v depthwise short-conv 与 cache 沿 channel 合并。f/output-gate 低秩首层仅在 B=T=1 时合并：T>1 改用同一packed weight的两个连续row slice。SwiGLU W1/W2同样在host按行打包且不保留重复参数：T=1用一个[2304→12416] GEMV，prompt/T>1用两个连续row slice保持原尺寸路径。block/final RMSNorm weight保留模型dtype供原生NPU fused op；fused-decode的o_norm weight继续一次扩宽为FP32。所有打包在host完成后再H2D。

**内存**：同参数数证明没有常驻 canonical+packed 双份权重；host packing 后实测 forward allocated 为 canonical 2,980,944,896 bytes、packed 3,007,683,584 bytes，reserved 为 3,221,225,472 / 3,141,533,696 bytes，消除了目标设备逐层 cat 的临时 reserve。

**证据**：benchmarks/verify_gdn2_packed.py、benchmarks/verify_gdn2_cpu_npu.py；tmp/gdn2-95b/cpu-npu-fp32-final.json、verify-packed-host-pack-{bf16,fp32,t64-bf16}.json、verify-packed-steady-{canonical,packed}-first.json、packed-cache-chain.json、text-greedy-packed-host-pack-final.json 及原始日志（git-ignored）


**观察**：torch_npu 基线：数据量从 smoke 到 kimi_linear_layer 差 512 倍，耗时只差 ~15% —— 它完全被 kernel launch 开销支配（向量化后仍有 63 次求逆迭代 + NT 次 chunk 迭代的 python 循环），**不是硬件算力上限**。自编译侧相反：耗时随工作量近线性（T 从 1024 到 4096，bd4 下 1.311→5.138ms，正好 3.9 倍），是真正的算力账。这也解释了 smoke 上 19.7x 的加速 —— 那里 torch_npu 在付固定开销而我们不付。

**跨 CANN 版本一致性**：kda_fwd 经 runtime 桥在 CANN 9.1.0 与 9.2.0 两台机器上的 relL2 逐位相同（smoke 3.288e-03 / multi_chunk 3.383e-03 / gva 3.359e-03），说明这个偏差来自算子自身的数值路径（见 gaps.json 的 kda-fwd-bwd-dtype-mismatch），与 CANN 版本无关。

**测量注意**：torch_npu 基线的跑间波动约 20%（kimi_linear_layer 在四个子进程里测到 7.920 / 5.445 / 5.439 / 5.811ms），所以加速比带同等量级的不确定度。每行的比值用的是该子进程自己测的基线，不是跨进程平均。

**下一步**：① 抬高 block_dim 上限（gaps.json 的 block-dim-ceiling，已升 P1）—— 扩展性到 4 仍线性，物理上有 28 cube。② kernel 侧的 nd2nz 返工（kernel-nd2nz-suboptimal）。③ 反向的同类测量，第二期随 kda_bwd 一起做。

### 全链路三层

> 全链路的上三层。窄切片原则：按算子倒推，用到哪个做哪个。首个目标是 KDA 链路，其依赖面比 GDN 少一个 module。

> ``done-torch`` 表示功能完成但核心实现是 torch 原生算子拼的；``done-cce-inference`` 表示自编译 CCE 核心已接入推理、但训练/长 prefill 仍有显式缺口；``done`` 给范围内前后向均完整的自编译算子。

**modules**

- `causal_conv1d` — 🔶 完成（torch 实现） · none — 需新写
  - 证据：2026-09-11 第二期：ascend_fla/modules/convolution.py ShortConvolution。2026-09-14 GDN-2 架构优化新增 PackedShortConvolution：q/k/v 三路 depthwise conv 与 cache 沿 channel 合并；CPU 覆盖无 cache、短序列补零、分段 cache 与规格拒绝，tests/test_modules.py 共 13 项。真实 95B 权重 T=6/T=64 的 packed conv state 与 canonical 三份 state 拼接后 relative-L2=0；本次也逐位相同，但只作诊断。
- `fused_rms_norm_gated` — 🔶 完成（torch 实现） · partial — matrix_normalization 可借
  - 证据：2026-09-11 第二期：ascend_fla/modules/fused_norm_gated.py FusedRMSNormGated；tests/test_modules.py 纯 CPU 11 项通过。2026-09-15：GDN-2 真实 B1T1H16 BF16 packed 的 swish output-norm/gate 已并入 a5.gdn2_fused_decode，通用模块与 KDA sigmoid 变体仍是 torch；不能把模型专用融合外推为整项 CCE 完成。
- `rms_norm` — 🔶 完成（torch 实现） · partial — matrix_normalization 可借
  - 证据：2026-09-14：ascend_fla/models/gdn2.py 的 RMSNorm 在 fp32 归一化、回写输入 dtype；GDN-2 CPU 测试与 NPU 冒烟均经过它。2026-09-15 packed inference 将已按请求dtype量化的55个不变norm参数一次性扩到FP32，消掉每token weight Cast，但37个block/final RMSNorm本体仍是torch_npu。
- `l2norm` — 🔶 完成（torch 实现） · matrix_normalization.row_l2
  - 证据：2026-09-11：layers/kda.py 里用 F.normalize 在 fp32 下做。2026-09-14：models/gdn2.py 按训练 recurrent kernel 的 sum(x²)+1e-6 语义在 fp32 做，并保留随后 q/sqrt(K) 缩放。

**layers**

- `kda` — 🔶 完成（torch 实现）
  - 证据：2026-09-11 第二期：ascend_fla/layers/kda.py KimiDeltaAttention，参数名与 fla 逐项对齐（KDA 算子自编译，周边 modules 是 torch —— modules-are-torch-not-kernels）。层级梯度实测：三个形状下输出相对 L2 4.9e-03，全部 17 个参数的梯度在 3.6e-03~2.2e-02，预算 0.1（A_log/dt_bias/f_proj 用 0.25，因为它们的梯度直接由 dg 来）。参考是同一份权重的 CPU 层，只把 KDA 算子换成 fp32 逐 token 递推版。承担了 fla 放在 kernel 里的三件事（q/k 的 l2norm、门控变换、beta sigmoid）。默认初始化（跨度 ~94）另有两项：前向对递推 oracle 相对 L2 4.697e-03（已测）；整层反向（门控跨度校准到 94）对同一份权重的 CPU 层逐参数比对，18 项全在预算 0.25 内（output 4.694e-03、dx 9.024e-03、A_log 1.551e-01、dt_bias 6.542e-02、f_proj 5.0e-02/5.2e-02，其余 4.5e-03~1.1e-02），由 test_deep_gate_backward_matches_cpu_reference 盯，已在有 ascend950 算子包的机器上跑通。梯度对齐那三个形状是在 exp(A_log)=1 下测的 —— 为的是把「接线对不对」和「深衰减下 bf16 本来就糙」分开，不是因为默认初始化跑不了。decode 路径未接（fused-recurrent-missing）。 **decode 已接线**：prefill 走 chunk + 空 cache，之后每步 fused_recurrent 传同一个 cache；prefill 128 + 逐 token 解码 5 步对整段 CPU 参考 4.26e-03~5.03e-03，由 test_prefill_then_decode_matches_one_shot_reference 盯。性能另见 decode-layer-overhead。
- `gated_deltanet` — ⬜ 未开始
- `gdn2` — ✅ 完成（CCE 推理）
  - 证据：2026-09-14：先完成 layers/reference/checkpoint/packed-inference 架构，再新增本仓 Ascriptor CCE `a5.gdn2_fused_recurrent` 并通过显式 core_backend 接入；未知 backend、CCE 求导、T>16 与 packed 求导均报错，不静默 fallback。canonical 保持发布 checkpoint 的399 keys；packed-inference加入SwiGLU W1/W2 host packing后为255个内部entry，参数数仍为1,450,096,416。算子五个aclnn contract case全过，真实95B的BF16/FP32整网torch_npu↔CCE logits与两类cache均在relative-L2预算内，cache offset/argmax一致。随后fused decode、short-conv、stateful Graph、原生RMSNorm与W1/W2 packing把纯模型decode推进到375.37~376.86 token/s，并保留相同64-token greedy序列。训练和长prefill仍缺gdn2_chunk_fwd_bwd。

**models**

- `kimi-linear` — ⬜ 未开始 · 注入 — 用上游模型定义，替换 linear attention layer
- `qwen3-next` — ⬜ 未开始 · 注入 — 用 HF transformers 的模型定义，替换 linear attention layer
- `gdn2-1.3b-fineweb-edu-100b` — ✅ 完成（CCE 推理） · 独立 LitGPT-compatible canonical 基线 + inference-only packed 布局；core backend 显式分派，后续只替换 GDN-2 算子边界
  - 证据：2026-09-14 实权重完成：95B checkpoint 17,401,727,659 bytes / sha256 4ac729c6…f6d；canonical 399 项 strict load，packed-inference 参数数不变。通用 CCE recurrent 的 BF16/FP32 数值与文本均通过。2026-09-15 追加 a5.gdn2_fused_decode：真实 BF16 T1 把 raw gates、recurrence 与 output norm 合成一launch，Cast309→74、设备约5108.8→4020us/token；反序paired speedup 1.410x~1.650x。静态双槽进一步把kernel即时对照4.733/4.716us降到合并4.247/4.256us，四AIV pipe利用率和86.84%→111.55%，证明成对流水已生效。step logits=3.436e-3、最差recurrent cache=9.521e-3（预算1e-2），32-token续写token ids不变。单pipe约80%的目标仍受erase→delta全局join限制（最忙MTE2平均37.52%）；训练与长prefill仍被gdn2_chunk_fwd_bwd阻塞。证据：tmp/gdn2-recurrent/ 与 tmp/gdn2-cast-fusion/（git-ignored）。

## 缺口

P0 1 项 · P1 17 项 · P2 9 项 · 已解决 11 项 · 共 38 项

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
| KDA | `c1-multihead-o-corrupt` | `decode-call-overhead`<br>`decode-layer-overhead`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`no-tail-path`<br>`block-dim-ceiling`<br>`qk-l2norm-not-in-kernel`<br>`state-layout-k-first` | `kda-fwd-bwd-dtype-mismatch`<br>`npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`kernel-nd2nz-suboptimal`<br>`fwd-caches-not-emitted`<br>`modules-are-torch-not-kernels`<br>`stable-unit-no-harness`<br>`gate-span-still-bounded` |
| GDN | — | `gdn-no-gqa`<br>`layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`state-dtype-bf16`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path`<br>`block-dim-ceiling` | `npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim` |
| GDN-2 | — | `no-varlen`<br>`no-tail-path`<br>`gdn2-abi-not-gdn`<br>`gdn2-chunk-gate-range`<br>`gdn2-decode-fragmentation` | `npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim`<br>`modules-are-torch-not-kernels` |
| DeltaNet | — | `layout-not-token-major`<br>`nonzero-initial-state`<br>`d-initial-state-absent`<br>`fused-recurrent-missing`<br>`no-varlen`<br>`scale-param-no-slot`<br>`no-tail-path` | `npu-builtin-ops-missing`<br>`fixed-kv-128`<br>`asymmetric-kv-dim` |

### 待统一修复的 kernel 问题

> **kernel 源码层面的问题统一修一轮，不零散改。** 这是 2026-09-11 定的：发现一条就去改一条，会在 ascriptor 侧留下一串互相干扰的小改动，而且每改一次都要重跑全部 case。做法：发现时把它记进本表并打 `requires_kernel_change`，本仓侧先按 AGENTS.md §7 **装闸报错或记为声明限制**，保证不静默出错；等攒够一批再统一进 ascriptor 侧（§3：本仓不改那个仓，要改走那一侧的流程或建本仓派生单元）。当前队列 24 项，见下表。

| 缺口 | 级别 | 要在 kernel 侧改什么 |
|---|---|---|
| `c1-multihead-o-corrupt` | P0 | **根因已定位**（A2-04 / PR #60，joshjms 诊断，PM 独立复算）：`kda_fwd/kernels/recurrent.py` 的 `Aqk` L1 交接是**两信用配固定槽** —— `aqk_l1_valid = DEvent(Pipe.MTE1, Pipe.MTE2, preset=True)`（:130）给两个信用，而槽是 `aqk_slot = Var(c_idx % 2)`（:243 写、:373 读），按 chunk 取。一个头最后一个 chunk 的槽是 `(C-1)%2`，下一个头第一个 chunk 的槽是 `0` —— **当且仅当 C 为奇数时两者相撞**，写方领先一周期踩进还没被读走的槽（:245 的 MTE2 写 与 :398 的 MTE1 读无序）。每核最后一个头后面没有写，所以恰好是对的。**不是漏了某次 DEvent/Mutex 调用**，是信用数与实际轮转的槽数不匹配 —— 与 ascriptor `library/docs/defects/M10-076-mutex-credits-and-handoff-slots.md` 同型（那一条在 autosync 里已修成『depth <= j 才算有序』，但本 kernel 是手写同步，不过 autosync）。**修法**：让槽按每核周期序号轮转 `((pair_idx - pair_begin) * C + c_idx) % 2`，两信用配两槽；或把两个 DEvent 降成 SEvent（少一周期 run-ahead）。 **补充（来自 A2-04 的 delta）**：`l1_Aqk` 是两槽 `DBuff`（:146）。备选修法是把 `aqk_l1_valid`/`aqk_l1_ready` 改 `SEvent`（去掉一拍 run-ahead，**性能未测**）。落地按 AGENTS.md §3 走本仓派生单元、进 kernel 批次。A5 上的硬判据：失效表 12 格全对、偶数 C 与未修版逐位相同、bd=1 与 bd=4 逐位相同。 |
| `block-dim-ceiling` | P1 | kda_fwd/kda_bwd 的 contract domain.block_dim 上限由 4 抬高并补 case。物理 28 cube / 56 vec，实测到 4 仍是线性扩展，所以这是当前最大的单点性能头寸。 |
| `d-initial-state-absent` | P1 | gdn / delta_rule 的 backward 产出 dh0。 |
| `decode-call-overhead` | P1 | 若要消掉 host 侧 15.4µs 的布局转换：kda_fused_recurrent 改成直接吃 token-major [B,T,H/HV,128] 并在 kernel 内按 hv//groups 取 q/k 的头。桥侧那 25µs 不用改 kernel。 |
| `fused-recurrent-missing` | P1 | KDA 的 decode kernel 已自写完（kernels/projects/a5/kda_fused_recurrent）。剩下的是 GDN / DeltaNet 的 decode kernel，照 KDA 这个的结构做。 |
| `gdn-no-gqa` | P1 | gdn_fwd/bwd 加独立 value-head 维度。 |
| `gdn2-abi-not-gdn` | P1 | recurrent 已由本仓 a5.gdn2_fused_recurrent 解决；队列中只剩独立 gdn2 chunk fwd/bwd，正式 ABI 需要 channel-wise g[B,T,H,K]、b[B,T,H,K]、w[B,T,H,V]。 |
| `gdn2-chunk-gate-range` | P1 | 新建 gdn2 chunk fwd/bwd 时必须从一开始采用覆盖至少已观测 1461 局部跨度的数值表示；禁止直接复制 KDA stable 的 105/155 跨度实现。 |
| `gdn2-decode-fragmentation` | P1 | BF16 raw-gate recurrent+output-norm、qkv short-conv+cache、RMSNorm2+paired W1/W2+SiLU×Mul三个模型专用CCE单元均已通过整网/profile。后续大步优化需新建weight-only低精度GEMV及质量验证链。 |
| `layout-not-token-major` | P1 | gdn / delta_rule 的公开布局改 token-major。 |
| `no-tail-path` | P1 | kda kernel 加 partial chunk 的 tail 路径，解除 T % 64 == 0。 |
| `no-varlen` | P1 | kda kernel 支持 cu_seqlens（变长序列打包）。 |
| `nonzero-initial-state` | P1 | gdn / delta_rule 支持非零初始 state。 |
| `qk-l2norm-not-in-kernel` | P1 | 把 q/k 的 L2 归一化、门控变换、beta sigmoid 融进 kernel（fla 的 KDA 在 kernel 内做）。 |
| `scale-param-no-slot` | P1 | 给 scale 参数开 kernel 标量入口。 |
| `state-dtype-bf16` | P1 | gdn 的 final_state 由 bf16 改 fp32。 |
| `state-layout-k-first` | P1 | state 布局支持 v-first（fla 的 KDA layer 用 state_v_first=True）。 |
| `asymmetric-kv-dim` | P2 | 拆开 K 与 V 的维度，支持 head_k != head_v。 |
| `fixed-kv-128` | P2 | 解除 K=V=128 定尺。 |
| `fwd-caches-not-emitted` | P2 | kda_fwd 的 gate / wy / recurrent 各多写一个 GM 输出：g_cumsum（stable 已有）、v_new、h。搬进 kernel 后 _scan_states 与两处 CPU 绕行可以一起删掉。 |
| `gate-span-still-bounded` | P2 | 若要继续抬门控跨度：把 64×64 tile 按行列分块、每对子块用各自的中点（等价分块 log-sum-exp），有限性上限随分块数线性增长。**但反向的约束是精度不是有限性，这一项可能帮不上忙** —— 先做 kda-fwd-bwd-dtype-mismatch。 |
| `kda-fwd-bwd-dtype-mismatch` | P2 | kda_bwd 九个 kernel 的 g_cumsum 入参由 bf16 改 fp32（前向入口的 g 本来就是 fp32）。这是降低反向精度曲线的主要候选 —— 但**是推测，要测**。 |
| `kernel-nd2nz-suboptimal` | P2 | ascriptor lint 标出的访存低效点：nd2nz 展开、偶数 block stride 撞 UB bank。lint 信息里带了板上实测倍数与具体改法，照着做即可。 |
| `modules-are-torch-not-kernels` | P2 | GDN-2 B1T1 BF16 output norm/gate、packed qkv short-conv/cache与RMSNorm2+W1/W2+SiLU×Mul三个模型专用CCE单元均已通过整网/profile；通用causal_conv1d、FusedRMSNormGated、其他shape与训练路径仍是torch算子。 |

### P0

#### `c1-multihead-o-corrupt` — 【P0·静默错误】C 为奇数且一个 cube 核要处理多个头时，kda_sub45_fused_kernel 的 Aqk L1 交接竞争，写出内容错误的 o

- **类别** correctness · **适用于** KDA · **阻塞** —
- **依据** **这是真实形状精度验收的第一个产出，而且是最坏的一类缺陷：没有 NaN、没有报错、范数还正常。** 发现路径：按 models.json 的 kimi 形状扫 C=1…16，C=1 那档 `o` 的相对 L2 是 **1.06**（其余档 3.2e-03），而同一次运行的 `final_state` 正常（2.46e-03）。
**失效规律**（2026-09-11，Ascend950PR / CANN 9.2.0，B=1、H=1、跨度 46，正确 = 相对 L2 < 0.01）：
| block_dim | HV=2 | HV=4 | HV=8 | HV=16 |
|---|---|---|---|---|
| 1 | 仅头 1 | 仅头 3 | 仅头 7 | 仅头 15 |
| 2 | 全对 | 头 1,3 | 头 3,7 | 头 7,15 |
| 4 | 全对 | 全对 | 头 1,3,5,7 | 头 3,7,11,15 |
正确的恰好是**每个 cube 核分到的最后一个头**。kernel 的 `pair_begin = (B*HV * GetCubeIdx()) // GetCubeNum()` 按 cube 核切 `B*HV`，而 `GetCubeNum() == block_dim` —— 所以安全条件是 **`B*HV <= block_dim`**。
**只在 C=1 出现**：C=2/3/16/64 下全对（HV 到 32 都试过）。chunk 循环跑第二遍时补上了缺的那次同步。kernel 源码里 `recurrent.py:176` 写着 `auto sync is not used here because the nested for loops interfere with it` —— 同步是手写的，而手写的那份假设了 C≥2。
**范围已逐条核实**：① `upstream` 与 `stable` 的错误值**逐位相同**（都 2.721e-01）→ 是共享的 `kda_sub45_fused_kernel`，不是本仓的 stable 派生引入的；② `final_state` 不受影响；③ **反向不受影响** —— C=1/HV=8 的六项梯度 dq 2.48e-02 / dk 3.94e-02 / dv 3.22e-03 / dbeta 3.35e-03 / dg 7.16e-02 / dh0 2.40e-03，与 C=2 同量级，因为反向不消费 `o`，只消费 `do` 与九个检查点。
**为什么契约的 case 测不到**：`kda_fwd` 四个 case 里 C=1 的三个都是 HV=1，唯一 HV=2 的那个是 C=2 —— **`C=1 且 HV≥2` 一个 case 都没覆盖**。这正是 toy-case-shapes 说的那件事。

**2026-09-17 补：模型定位 + 逐格复算（a5 pipesim，library 90cfcdc / kernels b3b3f9c）。**
PM 在权威 workspace 上独立跑了 `benchmarks/diag_c1_multihead.py`，**上表被逐格复现**：bd=1/HV=2→仅头 1、bd=1/HV=4→仅头 3、bd=2/HV=2→全对、bd=2/HV=4→头 1,3。
**并且暴露面比上表宽：是奇数 C，不是只有 C=1。** B=1/HV=2/bd=1 扫 C=1…6，冒险条数 2 / 0 / 2 / 0 / 2 / 0，错头恒为头 0：
| C | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| pipesim 冒险 | 2 | 0 | 2 | 0 | 2 | 0 |
| 回放错头 | 头 0 | 无 | 头 0 | 无 | 头 0 | 无 |
候选补丁（槽按周期轮转）在 bd∈{1,2}×HV∈{2,4}×C∈{1,3,5} 共 12 格上冒险归零、全头逐位正确；负对照（只轮转 q/qg）缺陷原样保留。
**与真机记录的冲突要并排看**：真机 2026-09-11 测过 C=3 且『全对』。两者不矛盾 —— 无序 ≠ 必然发生，那一次时序没踩到。但**结构上 C=3/5/7… 同样暴露**，而闸只拦 C=1。对 kimi（C = T/64）来说 T=192、320 都是奇数 C。

**根因定位（A2-04，#30 / PR #60，library 627f55f / kernels c89f69b，a5 管线模型值）**：pipesim 报出的冒险全部是"本头的 Aqk L1 读（:398）↔ 同核下一头的 Aqk L1 写（:245）"无序，条数与归因逐格相等、无其它冒险。按调度回放 o：真机失效表 12/12 格逐头吻合（逐位正确的头 = 表中对的头）。用 verify_real_shapes.py 同款输入（跨度 46）预测：kimi H32/HV32/C1/bd4 整体 o 相对 L2 1.060（本条记录 1.06），H1/HV8/C1/bd1 为 2.721e-01（本条记录 2.721e-01），kimi final_state 2.458e-03（本条记录 2.46e-03）；错头 |o|/|ref| 0.9254~1.1319、全部有限。**C≥2 全对的原因不是"chunk 循环第二遍补上了同步"**：C 为偶数时相邻两次写的槽交替，信用数与槽数匹配；C 为奇数时每次换头相撞一次。L0C 输出握手各头一致，与缺陷无关。修补（aqk-only 槽轮转）在模型中 43/43 格与奇数 C 12/12 格冒险 0、全头正确，偶数 C 格与上游逐位相同；负对照（只改 q/qg 槽）不起作用。
**一条被推翻的旧解释**：本条原来写着 C≥2 全对是因为『chunk 循环跑第二遍时补上了缺的那次同步』。两半都错 —— 根因不是漏同步（是信用数与槽数不匹配），C≥2 也不全对（奇数 C 照样撞）。那句话是从『C=1 坏、C=2 好』这个症状规律倒推的。AGENTS.md §6 已同步更正。
- **影响** ① **T=64 的前向输出是错的**（HV>block_dim 时），错得没有任何信号：有限值、量级正常、`final_state` 还对。短 prompt 的 prefill 正好落在这里 —— kimi 形状 HV=32、bd=4 时 32 个头里只有 4 个对。
② 训练同样中招：梯度本身没问题，但**前向输出错 → loss 错**，所以 T=64 的训练步是垃圾。
③ 之前所有精度结论都不受影响 —— 它们用的形状要么 HV=1（安全），要么 C≥2（安全）。这也是它藏了两期没被发现的原因。
④ **实测到的一个具体后果（decode 接线时撞上）**：64 token 粒度的 prefill 用不了 —— T=64 就是 C=1，HV=2 且 bd=1 时就已经越界。prefill 要么一次 ≥128 个 token，要么按头分批。写 prefill→decode 的测试时被闸拦下，只能把 prefill 从 64 改成 128。
- **建议** **已做**：`ops/kda/chunk.py` 的 `_check` 里加了 `_check_single_chunk_heads`，`C==1 and B*HV > block_dim` 直接报错并给出两条绕法（AGENTS.md §7：绝不静默降级）。闸的边界照实测表逐个钉在 tests/test_kda_gating.py 里 —— 这类缺陷一旦闸被改松，没有别的东西会报警。
**闸目前是漏的（2026-09-17 发现，待所有者决定）**：`_check_single_chunk_heads` 只拦 `C==1 and B*HV > block_dim`，而模型显示奇数 C 全都暴露。收紧成 `C % 2 == 1 and B*HV > block_dim` 会拒掉一些现在能跑、真机上也确实跑过的形状（C=3 测过且过了），属于收紧可用面 —— 按 AGENTS.md §1「改变范围要显式决策」，等所有者放行再改。在那之前**这条缺口的影响面按奇数 C 记**，不要按 C=1。
**要做（按代价排序）**：
① 按头分批调用：C=1 时把 `B*HV` 切成每批 ≤ block_dim 个头，多发几次 kernel。数学完全不变（头之间独立，已由头独立性检查证明），代价是多几次发射。**这是可用性修复，但会悄悄改变性能特征，要显式声明而不是默默做掉。**
② 建本仓派生单元修手写同步（照 kda_fwd_stable / kda_bwd_stable 的先例，AGENTS.md §3 不改 ascriptor 仓）。要先读懂 `recurrent.py` 的 DEvent/Mutex 配对 —— 目前只掌握了**症状规律**（每核最后一个头对）而不是确切缺哪一次同步，动手前必须先把那个找出来，否则改了也不知道为什么好。
③ 上游补 case：`C=1 且 HV≥2`。这条不管我们怎么修都该做，否则上游下次改这个 kernel 还会踩。
⑤ **上游补 case**（A2-04 建议）：`kda_fwd` 契约要加 `C=1 且 HV≥2`，以及 `奇数 C≥3 且 B*HV > block_dim`。现有四个 case 一个都盖不到。
⑥ **真机侧要补的**：本条记的『C=3 全对』在仓里**没有对应的运行日志**。闸的范围要按奇数 C 定的话，得先在 A5 上补 C=3 / C=5 且 `B*HV > block_dim` 的逐 chunk 比对。

### P1

#### `decode-call-overhead` — decode 的瓶颈是每次调用约 48µs 的固定成本，不是 kernel 也不是带宽

- **类别** performance · **适用于** KDA · **阻塞** `phase 3`
- **依据** kimi decode 形状 B1/HV32/T1（warmup 5 / iters 50 / 同步）：整次 **53~67 µs**，其中 host 布局转换 15.4~15.7 µs（26~29%）；T=1→16 的**设备侧边际 2.7~4.8 µs/token**（两次运行 T=16 整次分别 101.0 与 125.4 µs —— 报区间而不是单点），于是**每次调用的固定成本约 48~58 µs**。边际与按「4 头/核 × 2 趟 × 128 行」估的设备时间同量级。block_dim 1/2/4/8/16/28 的总时长 54~67 µs **没有趋势** —— 固定成本主导的征兆。
**我原本预测 decode 是带宽瓶颈（按 state 64KB×2/头估下限约 2.5µs），那个预测错了。**而且如果不拆开量，bd 无效会被误判成「扩展性不行」—— AGENTS.md §6 铁律一说的就是这个。
复现：`benchmarks/verify_decode.py --check split`。
- **影响** 48 层模型按 55µs/层算是 **2.6ms/token**，不可接受（decode 一步的预算是几十 µs 量级）。所以 decode 算子虽然正确，但还不能用；这是 fused-recurrent-missing 接层之前必须先解的。
- **建议** 按占比从大到小：
① **桥侧（约 25µs）**：decode 期间形状固定，`aclCreateTensor` 的描述符与 workspace 查询可以按 (算子, 形状签名) 缓存复用，每步只换 data_ptr。要先 profile 确认 25µs 花在哪一段 —— **别再先推断后看数据**。
② **host 布局（15.4µs）**：让 kernel 直接吃 token-major 并在 kernel 内做 GQA 取头，就不用 permute/repeat_interleave/contiguous。这会改 ABI，归入 kernel 修复队列。
③ **输出（约 7µs）**：同理，让 kernel 直接写 token-major。
④ 设备侧 4.8µs/token 先不动 —— 它已经是最小的一项。

#### `decode-layer-overhead` — 整层 decode 一步 458µs，其中 KDA 算子只占 18%，层里那十几个小算子占 67%

- **类别** performance · **适用于** KDA · **阻塞** `phase 3`
- **依据** `benchmarks/bench_kda_decode_layer.py`，hidden=2048 / H16 / HV32 / bd1 / prefill 128，warmup 10 / iters 50 / 同步（2026-09-11，CANN 9.2.0）：
| 项 | 耗时 | 占比 |
|---|---|---|
| 整层一步 | 458.0 µs | 100% |
| 其中 KDA 算子 | 83.1 µs | 18% |
| 其中层里其余部分 | 305.1 µs | 67% |
**48 层外推 22 ms/token。**
层里 T=1 时要走的调用：7 个投影（q/k/v/f/b/g/o）、3 个短卷积、softplus、sigmoid、两次 fp32 l2norm（含 bf16↔fp32 往返）、FusedRMSNormGated —— 每个都几乎没有计算量却各要一次 launch。所以这 305µs 基本是**启动开销之和**，不是算力。
拆法是逐段替换而不是推断（AGENTS.md §6 铁律一）：`without_op` 那条跑完层里除 KDA 算子以外的全部步骤，`t_op` 只跑算子。
- **影响** decode 现在**功能可用但性能不可用**：22 ms/token 意味着 45 token/s，而同级别模型的目标是几十到上百倍于此。
注意优化的着力点：算子侧只占 18%，且其中设备时间只有几 µs —— **把 kernel 再优化一倍，整层只快 9%**。该动的是层这一侧。
- **建议** 按占比排序，且都要先 profile 再动手：
① **图捕获**（最大头）：decode 每步的形状完全固定，整层可以用 ACL/torch_npu 的 graph capture 录一次重放，把十几次 launch 压成一次。这是业界对 decode 的标准解法，但要确认 torch_npu 在本版本支持、且我们的自定义算子能进图。
② **合投影**：q/k/v 三个投影可以并成一个 `[hidden, 3*key_dim]`（权重拼接，数学不变）；f/b/g 同理。能把 7 次降到 3 次。
③ **去掉 l2norm 的 dtype 往返**：现在是 bf16→fp32→normalize→bf16。T=1 时这两次 cast 各是一次 launch。若 `qk-l2norm-not-in-kernel` 做掉（搬进 kernel），这一段连带消失。
④ 算子侧的 `decode-call-overhead` 只值 18% × 其中的固定成本，排在后面。

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

#### `fused-recurrent-missing` — decode 路径：KDA 已自写 kernel 并接进 layer，性能不可用；GDN/DeltaNet 仍整族缺失

- **类别** coverage · **适用于** KDA / GDN / DeltaNet · **阻塞** `phase 3`
- **依据** ascriptor kernels catalog 中无 fused_recurrent 类单元；六个 a5 单元均为 chunk 路径。
**KDA 这半已经补上**（2026-09-11）：本仓自写 `kernels/projects/a5/kda_fused_recurrent`，state 常驻 UB、两趟扫、按 B*HV 切给向量核（每头一核，核间无同步）。`ascriptor check` 0 error / 0 warning / 156 ops；真机八个形状下 o 的相对 L2 9.575e-08~1.789e-07、final_state 3.199e-08~1.440e-07（对 fp32 递推参考）；**逐 token 调 T 次并串接 state 与一次调 T 个 token 逐位相同**（decode 正确性就靠这条）；block_dim 1~28 全通（28 即 56 个向量核的物理上限）。
**它还顺带消掉一个约束**：逐 token 只用 `exp(g_i)`（~1.5），所以**门控跨度没有上限** —— 对比 chunk 路径的前向 155 / 反向 105（gate-span-still-bounded）。
- **影响** **KDA 的 decode 现在功能上可用了**（2026-09-11 接进 `layers/kda.py`）：prefill 走 `mode="chunk"` 并传一个空 `cache`，之后每步 `mode="fused_recurrent"` 传同一个 `cache`（原地更新，装 `recurrent_state` 与三份 `conv_state`）。实测 prefill 128 token 后逐 token 解码 5 步，对整段 CPU fp32 参考的相对 L2：prefill 4.595e-03、decode 4.628e-03 / 5.030e-03 / 4.650e-03 / 4.257e-03 / 4.783e-03 —— **逐 token 报数**而不是报平均，因为 conv_state 漏传只会坏前 conv_size−1 个 token，平均会掩盖它。
剩下三条：
① **性能不可用**：整层一步 458µs、48 层 22ms/token，见 decode-layer-overhead（层侧 67%）与 decode-call-overhead（算子侧的固定成本）。
② **不可求导**：本仓只有 chunk 的反向 kernel。层里会直接报错而不是静默不建图。
③ GDN / DeltaNet 的 decode 仍整族缺失，随各自扩族再补（第四期）。
（**此前我在这里写过「ShortConvolution 只有整段前向」，那是错的** —— `modules/convolution.py` 本来就支持 `cache` + `output_final_state` 的单步解码，还处理了 T < kernel_size 的补零。接线时直接用上了。）
- **建议** ① 先解 decode-layer-overhead（层侧占 67%），再看 decode-call-overhead（算子侧）。**别先去优化 kernel** —— 它只占 18%，其中设备时间才几 µs。
② GDN / DeltaNet 的 decode 随扩族做，照 KDA 这个单元的结构抄（两趟扫 + 每头一核）。
③ 接 HF/fla 模型时要一层 cache 适配器：我们的 `cache` 是本仓自己的两键字典，而 fla 的 KDA layer 用 `state_v_first=True`（V 在前），见 state-layout-k-first。

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

#### `gdn2-abi-not-gdn` — GDN-2 的 channel-wise erase/write/decay ABI 与现有 GDN kernel 不同

- **类别** abi · **适用于** GDN-2 · **阻塞** `gdn2_chunk_fwd_bwd`
- **依据** 目标模型 revision 86327354 的 gdn2_1.3B 配置是 H=HV=16、K=V=128，形状本身匹配 A5 定尺；但训练实现传给 recurrence 的张量是 q/k/v [B,T,H,128]、g [B,T,H,128] fp32、b [B,T,H,128]、w [B,T,H,128]，其中 b 是 key-channel erase gate，w 是独立的 value-channel write gate，g 也是 key-channel decay。现有 a5.gdn_fwd 的正式 ABI 只有 beta [B,H,C,64] 与 g [B,H,C,64] 两个 per-token 标量槽，没有 w，也没有 channel-wise b/g。
**recurrent 这一半已按独立 ABI 做完**（2026-09-14）：本仓新增 `a5.gdn2_fused_recurrent`，直接吃 token-major q/k/v/g/b/w 与非零 FP32 state，kernel 内做 q/k L2 norm 和 scale；Ascriptor unit runner 的 CCE/aclnn 五个 case 全过，o 最大 relative-L2=4.169e-6、state 最大 9.108e-8。真实 95B 的 BF16/FP32 整网 torch_npu↔CCE logits/cache 均在预算内，并由 CCE backend greedy 生成 32 个可读 token。现有 a5.gdn_* 仍未被误用或改名。
- **影响** 现有 a5.gdn_fwd/bwd 仍不能用于这个 checkpoint；强接会在形状层面丢掉 128 倍门控信息并无法表达独立 write gate，是确定性的语义错误。独立 recurrent kernel 已解除短 prompt/decode 的阻塞；剩余影响收窄为 T>16 的长 prefill 与训练反向，它们仍缺 gdn2_chunk_fwd_bwd。
- **建议** recurrent 已完成，不再改。下一步只做独立 gdn2 chunk fwd+bwd：L=64、K=V=128、直接吃 token-major；数值表示先解决 gdn2-chunk-gate-range 的 1461 跨度反例。g 变换与 b/w sigmoid 是否继续融合，等整层 profile 后决定；不能因名字相近复用 a5.gdn_*。
**2026-09-17**：forward-only 部分已排上看板 GD2-01（主机侧：量程设计 + kernel 单元）/ GD2-02（gated，A5 真机数 + 整网验证），D-PM-16 批准。backward 仍未排期。

#### `gdn2-chunk-gate-range` — GDN-2 的真实 chunk 衰减跨度远超 KDA stable 已验证域，chunk 算法必须单独做量程设计

- **类别** numerics · **适用于** GDN-2 · **阻塞** `gdn2_chunk_fwd_bwd`
- **依据** 2026-09-14 用真实 95B checkpoint、B=1/T=4096、合法 vocab token id 的确定性随机输入（seed=20260914）逐层抓取 g=-exp(A_log)*softplus(f+dt_bias)。18 层合计观测：单 token 的 -g 最大 60.926；64-token 局部累计跨度最大 1461.214，各层 chunk-span 均值再平均为 69.041；完整 4096-token 累计跨度最大 83768.180。logits 仍全有限，说明逐 token recurrence 本身可用；证据在 tmp/gdn2-95b/gate-domains-t4096.json 与原始日志（git-ignored）。该输入不是自然语料分布，不能拿均值外推任务分布，但它是模型公开 token 域内的有效反例，足以否定“照搬 KDA 155 的量程即可”。
- **影响** 现有 KDA stable chunk 的前向已验证跨度约 155、反向约 105，不能直接作为 GDN-2 的设计上限。任何在 64-token 区间端点或中点形成 exp(±span) / exp(±span/2) 的实现，在观测到的 1461 跨度下都可能出现 0、inf 或 inf×0；有限输出也不能替代梯度精度验证。decode recurrence 每步只使用 exp(g_t)，g_t≤0，向 0 下溢与数学极限一致，不受这个跨 token 配对量程问题影响。
- **建议** 冻结 q/k/v/g/b/w 的公共语义 ABI，但 chunk 实现暂缓。写 kernel 前先把 forward/backward 全链路所有 exp(g_i-g_j) 逐处列出，按 4/8/16/32/64 子块扫描实测量程，并同时对 fp32 recurrent oracle 验有限性与相对 L2；算法应采用不会构造大正指数的分段/归一化表示，而不是只把端点锚改成中点。测试至少覆盖本次 1461 反例、自然文本样本、chunk↔recurrent 与梯度。

#### `gdn2-decode-fragmentation` — GDN-2 host/elementwise fragmentation 基本收敛，BF16 decode 转为 GEMV 权重读取瓶颈

- **类别** performance · **适用于** GDN-2 · **阻塞** —
- **依据** 2026-09-15 初始真实95B/BF16/packed/B1T1 trace：torch_npu 6043.4us/1523 kernels；旧 CCE recurrent 5108.8us/1163 kernels，Cast 565.9us/309。去掉不变weight cast并新增 `a5.gdn2_fused_decode` 后，kernel内完成raw gate、q/k norm、FP32 recurrence/state、output RMSNorm+swish和最终BF16舍入；Cast降到74次/token、147.355us，launch 1163→640，反序整网paired speedup为1.410x~1.650x。
**动态双缓冲反例**：16/32/64-row DBuff虽无hazard/deadlock，pipesim却从整state 9444退化到14801/11917/10463 cycles；DB64真机4.804us且四AIV pipe和98.17%。Lowered IR显示auto_sync把动态 `slot[n]` 与 `slot[n-1]` 保守判为同root，插入read1→compute0和write0→compute1假依赖，因此只有存储双槽、没有流水。
**接受实现**：把两个64-row槽静态命名并展开，保持公式、VF算术、流量、block_dim和ABI不变。check 0 error/0 warning/404 surface ops，CCE emit 415；3/3 sim与3/3 pipesim数值通过且无hazard/deadlock，pipesim=9613 cycles，同核多pipe重叠3.68%→11.44%。两次独立真机profile各54样本，kernel mean=4.259/4.236us，合并mean/median=4.247/4.256us；夹心整state即时对照=4.733/4.716us，候选快1.114x/1.108x。候选Vector/Scalar/MTE2/MTE3合并平均34.73/17.60/37.52/21.70%，四pipe和111.55%（对照86.84%），证明真实重叠。aclnn与真实95B整网数值复验均过原预算；两轮profile与整网任务所有kernel/PID只在物理7、物理0活跃0次。另一次canonical aclnn fresh build在actual launcher前捕捉到物理0的python/0MB瞬时条目，时间对齐CANN opc编译；test_aclnnop只在7，0未留驻，后续严格单卡窗口不再fresh build。证据：tmp/gdn2-cast-fusion/{tile64-profile-v1,static2-v4-profile-v1,static2-control-profile-v1,static2-v4-profile-v2,static2-v4-fullmodel-v1,static2-v4-canonical-aclnn-v1}/（git-ignored）。
**NPU Graph 实测**：torch_npu 2.10 的 `NPUGraph` 能同时捕获整网内置算子与本仓 ctypes→aclnn custom op。固定prompt-cache对照为6735.3→3834.9us/token（148.47→260.76 token/s，1.756x）；可递推版本在图尾把18层新recurrent/conv cache拷回固定输入地址，v3~v5同轮eager为5911.9~6323.7us/token，graph稳定在3908.2~3911.9us（255.63~255.87 token/s，1.512x~1.617x）；逐token同步的graph为3915.6~3926.0us（254.71~255.39 token/s，对eager加速1.554x~1.736x）。四个不同token连续replay的logits、两类cache均与eager逐位相同。eager提交约5.67~6.78ms/token，graph提交仅2.8~7.4us/token，说明host发射已不再配速；stateful graph每token额外回写19,759,104 bytes，实测只比fixed graph多约75us。清空eager allocator历史后，捕获常驻allocated/reserved增量约39.6/130.0MB，一次捕获约13.6ms。证据：`benchmarks/bench_gdn2_graph_capture.py` 与 tmp/gdn2-graph-capture/{fixed-v1,stateful-v3,stateful-v4,stateful-v5}.json（git-ignored）。
**真实生成接线**：`GDN2NPUGraphDecodeRunner` 与 `generate_tokens(decode_backend="npu-graph")` 已接入。真实95B、64-token greedy在graph-first/eager-second与eager-first/graph-second两组独立进程中，四次token ids逐项相同。含logits D2H、CPU argmax、token H2D的graph decode为244.21/244.79 token/s，对应eager为173.02/148.84，两个顺序加速1.411x/1.645x；含冷prefill和每请求setup的64-token总吞吐graph为99.11/104.23、eager为96.73/91.60，两个顺序加速1.025x/1.138x。四个接受窗口物理0活跃均为0，物理7各只有一个本任务PID。证据：tmp/gdn2-graph-generation/{graph-greedy64-v2,eager-greedy64-v1,eager-greedy64-v2,graph-greedy64-v3}.json（git-ignored）；一轮与外部作业竞争的smoke已隔离为rejected，不作性能证据。
- **影响** host gap与大部分elementwise fragmentation已经消除：低内存默认纯模型Graph约2.659ms/376.11 token/s；保守mixed显式opt-in为2.589ms/386.32 token/s。其最新device时间90.175%是native GEMV+mixed W12权重流，mixed AIC MTE2平均91.88%。gamma-fold v1的56.24us/token投影已因递推精度作废；保守v2以1,029,832,704 bytes派生权重换取clean Graph约68.20us/token，必须连同内存决策。若要下一次大步收益，需要weight-only低精度/量化及独立任务质量实验。
- **建议** 精度、同卡stateful Graph A/B和profile均已完成。现在先决定：①保持vendor低内存默认；②用1.030GB派生布局换2.6%吞吐；③设计prompt/decode共用布局去掉重复权重。若继续追求大步收益，再决策weight-only低精度及真实生成/任务质量预算。明确不选：单独W1/W2 kernel、N=64小tile、减小block_dim、增加冗余工作，或已证伪的细粒度DMA双缓冲。W3融合不减少约85.8MB/layer的BF16 W12+W3权重字节，只是ceiling候选。

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

- **类别** performance · **适用于** KDA / GDN-2 · **阻塞** —
- **依据** 通用 `ascend_fla/modules/convolution.py` 仍用 F.conv1d+F.silu，`fused_norm_gated.py` 仍用rsqrt/mean/激活；KDA的投影/l2norm/门控也仍是torch算子。GDN-2真实B1T1 BF16 packed路径的模型专用CCE例外现在有 `a5.gdn2_fused_decode`、`a5.gdn2_short_conv_decode`，以及显式opt-in的 `a5.gdn2_norm2_w12_swiglu`。后者v1整网已拒绝；v2 reference2/2与random sim/pipesim通过，relative-L2=3.416e-7、无hazard/deadlock；`mlp_backend='cce'`才启用。通用卷积、其他projection、W3、prompt MLP及训练路径仍依赖torch_npu。
- **影响** ① 这些步骤在内置算子包不全的机器上不可用（需要 conv1d/silu/matmul），而自编译的 kda 算子本身不受影响 —— 所以层级验证比算子级验证对机器挑剔。② 层级耗时里有一部分不归本仓的"高效率算子"管，报层级性能数时必须拆开说，否则会把 torch 的开销算进算子账上。
- **建议** KDA保持原计划。GDN-2混合MLP的kernel、显式接线、真实95B递推、同卡Graph夹心与profile均已完成。下一步先决定是否用约1.030GB派生布局换2.6%吞吐，或设计不重复权重的prompt/decode统一布局；追求大幅提升则先决策weight-only低精度与任务质量预算。通用causal_conv1d与FusedRMSNormGated仍需独立算子，不能从模型专用T1 kernel外推为已完成。

#### `stable-unit-no-harness` — 本仓自有的三个单元都还不能用 ascriptor harness 独立跑

- **类别** verification · **适用于** KDA · **阻塞** —
- **依据** kernels/projects/a5/kda_fwd_stable/ 目前有 contract.json、README.md 与三个 kernel 文件，但缺 unit 协议要求的 unit.py（make_inputs/reference/execute）与 run.py —— 它们要对接 ascriptor 的 _unit_runner。现在的验证全部经本仓 runtime 桥 + pytest 做。
（2026-09-11 起这条覆盖三个单元：`kda_fwd_stable`、`kda_bwd_stable`、以及本仓自写的 `kda_fused_recurrent`。三者的 `ascriptor check` 与 runtime 桥都过了，缺的是 `unit.py` + `run.py` 对接 `_unit_runner`。）
- **影响** ① 拿不到 ascriptor harness 的 sim / pipesim / cannsim 几个 stage 的证据，也就用不上它的逐 stage checkpoint 比对（那对定位 kernel 内部错误很有用）。② 这个单元不能被 ascriptor 侧的人独立复现，不利于把修法推回上游。contract.json 的 support 里已如实标注证据来源，没有假装有 harness 证据。
（2026-09-11 起这条覆盖三个单元：`kda_fwd_stable`、`kda_bwd_stable`、以及本仓自写的 `kda_fused_recurrent`。三者的 `ascriptor check` 与 runtime 桥都过了，缺的是 `unit.py` + `run.py` 对接 `_unit_runner`。）
- **建议** 补 unit.py 与 run.py。reference 可以直接用 ascend_fla/reference/kda.py 的逐 token 递推版（它没有跨度上限，正是宽域下唯一可用的 oracle）。做完后把 contract.json 的 support 按 harness 实际结果更新。
（2026-09-11 起这条覆盖三个单元：`kda_fwd_stable`、`kda_bwd_stable`、以及本仓自写的 `kda_fused_recurrent`。三者的 `ascriptor check` 与 runtime 桥都过了，缺的是 `unit.py` + `run.py` 对接 `_unit_runner`。）

#### `gate-span-still-bounded` — 稳定化把门控跨度上限从 80 抬到前向 155 / 反向 105，但没有去掉上限

- **类别** numerics · **适用于** KDA · **阻塞** —
- **依据** `kda_fwd_stable` / `kda_bwd_stable` 走的是**对称分解**：把 `exp(a_i − a_j)` 拆成两个以中点为锚的因子，各压到 ±span/2，有限性的理论上限正好翻倍 —— 前向 `2 × -ln(FLT_MIN_NORMAL) ≈ 174.7`，反向 `2 × ln(BF16_MAX) ≈ 177.4`。
**但两条链的闸不是同一回事，这是实测出来的**：
* 前向的约束是**有限性**。实测跨度到 155.97 时 `o` 的相对 L2 仍稳定在 2.85e-03~3.19e-03，完全不随跨度退化 —— 所以闸就设在实测最深点 155。
* 反向的约束是**精度，而且它先于有限性到来**。梯度到 169.76 都还是有限值，但对 fp32 递推参考的相对 L2 随跨度单调上升：dq 在 46/94/105/110/130/169 处是 2.89e-02 / 4.55e-02 / 4.86e-02 / 4.99e-02 / 6.17e-02 / 6.88e-02，**130 处越过契约预算 0.05**；dg 在 169 处崩到 6.49e-01（预算 0.25）。所以反向的闸设在 105。
**105 的下界是 fla 初始化本身的上界，这点此前被我写错了。** 跨度不是常量 ~94，它是**随机变量** —— 跨度 ∝ `max_hv exp(A_log)`，而 fla 取 `A_log = log(U(1,16))`，所以 `exp(A_log) ∈ [1,16]`。实测 12 个 seed：HV=1 给 21.5~96.2、HV=2 给 15.3~94.8、**HV=8 给 63.1~100.9**（头数越多越稳地顶到上界，因为取 max 的样本更多）。甚至同一个 seed 下，建层与抽 x 的先后顺序不同就从 64.6 变成 94.0（RNG 消耗顺序不同）。上界是 `exp(A_log)≤16 × dt≤0.1 × 63 步 ≈ 100.8`。**闸必须覆盖 100.8**，否则默认初始化的层会被我们自己的门控拒掉 —— 这就排除了 100。上限那头是契约预算还成立的最深实测点：110 处 dq=4.99e-02 只剩 0.2% 余量，不取；105 处 dq=4.861e-02，余量 2.8%。
于是 `MAX_GATE_SPAN` 是二维的：`{impl: {forward, backward}}`，纯推理用前向那条、训练用反向那条。fla 默认初始化的层跨度约 94，两条都满足。
**跨度不随 T 增长** —— 它是 chunk 内（64 token）的量，cumsum 每 chunk 重置。推高它的是 `exp(A_log)` 与 `dt` 的乘积。
- **影响** ① fla 默认初始化的跨度上界是 100.8，反向的闸 105 只剩 4% 余量，而实测 HV=8 时 8 个 seed 里就有一个到 100.6。`A_log` 与 `dt_bias` 都是可训练参数，训练中 `dt` 变大就会撞上限 —— 届时是**报错**（设计如此），但会中断训练。要继续得显式 `check_gate_range=False` 并接受超预算的梯度，或者压 `dt`。
② 门控检查默认开，每次前向对 g 做一次 cumsum + 两次规约，实测 0.212ms / 步（bd=4 / kimi_linear_layer，占训练步 4%）。
③ **有限性上限（反向 169.76）比声明上限宽**。需要更深跨度又能接受精度退化的调用方，可以自己关掉检查 —— 曲线在 `kda_bwd_stable/contract.json` 的 `accuracy_vs_span` 里，照着选，不要瞎试。
- **建议** 现在不做。真撞上限时按代价排序：
① **先查 dq 为什么是约束项**。它在跨度 46 时就已用掉 58% 预算，说明深衰减下 `dq` 的主项（`d_qg · exp(g) · scale`，见 inverse_epilogue）对 bf16 的 `g` 最敏感。若把 `g_cumsum` 检查点从 bf16 升到 fp32（kda_bwd 的 ABI 问题，见 kda-fwd-bwd-dtype-mismatch），这条曲线可能整体下移 —— **这是推测，要测**。
② 把 64×64 的 tile 再按行列分块，每对子块用各自的中点（等价于分块 log-sum-exp），有限性上限随分块数线性增长。但若约束是精度而不是有限性，这一项帮不上忙。
③ fla 的 `lower_bound` / `safe_gate`：给门控设下界。那**会改变数学**，属于模型侧决策，不能当数值修补悄悄加上（本仓目前显式拒绝这两个开关）。
