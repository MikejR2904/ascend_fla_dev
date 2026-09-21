# FMT-01 host 算子审计：工具 + 全仓公共算子入口清单（A5 真机）

> 结论只对下面这一套环境成立（`AGENTS.md` §6「结论不跨 SoC / 环境继承」）。本仓其它 BF / FMT 任务用的是
> CANN 9.1.0-beta.1 / 9.2.0 与 torch 2.12；**本清单是 CANN 9.3.0 / torch 2.7.1 上的观测**，aten 层的算子名与分解
> 会随版本变化，逐条对照时先看环境行。

## 0. 摘要（先看这张表）

审计对象：`ascend_fla/ops/**` 的公共算子入口，每条 dtype 路径各跑一遍。判据是 `docs/pm/bf16-kernel-side.md`
（D-PM-35 / D-PM-37）：host 侧只允许分配输出、不搬数据的元数据操作、只读校验、取指针与 launch；
dtype 转换与格式（布局）转换、产出进入计算的数据的算术一律禁止。

| 入口 × 路径 | 判定 | 调用数 | 分配 | 元数据 | 校验 | 禁止:dtype | 禁止:格式 | 禁止:算术 |
|---|---|---|---|---|---|---|---|---|
| `chunk_kda` 推理（raw flags 关，无梯度） | **clean** | 71 | 24 | 15 | 32 | 0 | 0 | 0 |
| `chunk_kda` 推理（raw flags 开，无梯度） | **clean** | 62 | 31 | 25 | 6 | 0 | 0 | 0 |
| `chunk_kda` 训练（raw flags 关，含反向） | violations | 253 | 84 | 113 | 32 | 10 | 2 | 12 |
| `chunk_kda` 训练（raw flags 开，含反向） | violations | 280 | 84 | 115 | 6 | 18 | 2 | 55 |
| `chunk_kda_fwd_with_caches`（stable） | violations | 130 | 33 | 71 | 6 | 8 | 2 | 10 |
| `chunk_kda_fwd_with_caches`（upstream impl） | **当前 pin 编译不过**（上一个 pin 可编可跑，见 §3.1） | — | — | — | — | — | — | — |
| `chunk_kda_bwd` | violations | 61 | 43 | 17 | 0 | 0 | 0 | 1 |
| `fused_recurrent_kda` BF16 | **clean** | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `fused_recurrent_kda` FP32 | **clean** | 3 | 3 | 0 | 0 | 0 | 0 | 0 |
| `chunk_gdn` FP32 / BF16 | **clean** | 18 / 18 | 18 | 0 | 0 | 0 | 0 | 0 |
| `chunk_gdn_bwd` FP32 / BF16 | **clean** | 14 / 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| `chunk_pgdn` FP32 / BF16 | **clean** | 71 / 71 | 24 | 2 | 45 | 0 | 0 | 0 |
| `chunk_precond_kda` BF16 | **clean** | 29 | 28 | 1 | 0 | 0 | 0 | 0 |
| `chunk_precond_kda` FP32 | 工具报 violations（6 次算术）→ **按 REVIEW 口径归为"未登记的只读校验"** | 91 | 22 | 5 | 58 | 0 | 0 | 6 |
| `chunk_gdn2` FP32（仓主轨道） | **clean** | 18 | 18 | 0 | 0 | 0 | 0 | 0 |
| `chunk_gdn2` BF16（仓主轨道） | violations | 22 | 18 | 0 | 0 | 4 | 0 | 0 |

**五处违规 + 一处未登记的只读校验，逐处定位**（分类按 REVIEW 07:16Z 的口径：PKDA FP32 那处产出的是判定、
不进入计算，属 D-PM-42 ① 的"只读校验"，只是写在未登记的函数里，**不与 D-PM-37 的 dtype / 格式 / 算术违规混为一类**）（详细行见 §3；原始 JSON 见 `benchmarks/evidence/host_op_audit/run-a5-final/`，
另有一份在上一个 library pin 上补测的 `run-a5-lib627f55f/`，用途见 §3.1；每轮还有一份未过滤的
`console.log`——**只把绝对路径换成占位符**（仓库公开），其余未删未过滤）：

| # | 位置 | 内容 | 已登记？ |
|---|---|---|---|
| 1 | `ops/kda/chunk.py` `_scan_states`（约 :700–:712） | Cast + matmul + stack 的 host 复算（`h` / `v_new`） | 是，D-PM-42 存量例外 |
| 2 | `ops/kda/chunk_bwd.py:286` `chunk_kda_bwd` | `dw = -d_vh` 一次 `neg` | 是，D-PM-42 存量例外 |
| 3 | `ops/kda/chunk.py` `_prepare_inputs` / `_l2norm`（raw flags + 需要梯度） | 训练路径保留的 host 前处理计算图 | 是，BF-08 在做 |
| 4 | autograd 引擎（反向节点，Python 栈上无我们的帧） | 对第 3 条那段 host 算术求导产生的反向算子 | 随第 3 条一起消失 |
| 5* | `ops/pkda_chunk_fwd.py:126` `chunk_precond_kda`（仅 FP32 路径） | **未登记的只读校验**（`ref.reference.validate_inputs` 里的 `pow` / `sum` / `neg`，算 key 行 L2 范数与域判定，产出只是 bool） | 不是 D-PM-37 违规；建议登记或搬 kernel（§6 建议 A，P2） |
| 6 | `ops/gdn2_chunk_fwd.py:111` 与 `:128` `chunk_gdn2`（仅 BF16 路径） | BF16→FP32 加宽 ×3、FP32→BF16 输出回转 ×1 | 是，仓主轨道 / BF-05 在做 |

## 1. 环境行（每份 JSON 自带一份同样的）

| 项 | 值 |
|---|---|
| SoC | Ascend950PR_9589（a5），两张卡；本任务只用一张，另一张有他人任务 |
| CANN | `ASCEND_TOOLKIT_HOME` 的 `ascend_toolkit_install.info`：`version=9.3.0`、`innerversion=V100R001C12B116`；**同一棵树下另一个 `version.info` 报 `Version=9.2.0`（子包版本号，非 toolkit 版本）**，两者都记进 JSON 并带 sha256 |
| 内置算子包目录 | `ascend910_93`、`ascend910b`、`ascend950`、`config` → torch_npu 计算算子可用 |
| Python / torch | 3.11.15 / torch 2.7.1+cpu / torch_npu 2.7.1.post4 |
| ascriptor | library `90cfcdc` / kernels `b3b3f9c`（`agent/compatibility.json` 当前 pin） |
| 被审计提交 | `8e5da4c7ed25d8cc905601b1c1154fae96aabb05`（真机上跑的是该提交的 `git archive` 副本，不是 git checkout，所以提交号由 `HOST_OP_AUDIT_COMMIT` 显式声明） |
| 工具 | `benchmarks/host_op_audit.py`，每份 JSON 带 `audit_tool_sha256` 与被审计入口源文件的 sha256 |
| 卡的健康 | 该机**没有 npu-smi 二进制**，只能用 `torch_npu.npu.mem_get_info` 间接看占用；审计前后都确认所用卡空闲 |

## 2. 工具（`benchmarks/host_op_audit.py`）

一条命令重跑：

```bash
python benchmarks/host_op_audit.py --list                       # 19 个 case（入口 × dtype 路径）
python benchmarks/host_op_audit.py --self-check                 # 纯 CPU 自检分类器（11 项）
python benchmarks/host_op_audit.py --case <id> --output <dir>   # 审计一个 case
python benchmarks/host_op_audit.py --all  --output <dir>        # 全部 case，每个一份 JSON + summary.json
```

设计上的四个决定，都是踩出来的：

1. **分类表显式**（`CATEGORY_BY_OP`）：逐个算子写死；表里没有的落 `unclassified` 并在报告里单列，**不猜**。
2. **拷贝按 storage 判，不按算子名判**：每行记 `copy`（输出的 storage 不在输入 storage 集合里且不是分配类算子）。
   `contiguous` 在已连续时是 no-op；在 aten 层，真的要搬数据时会看到 `clone`。
3. **只读校验按调用点认**（`CHECK_SITE_FUNCTIONS`）：`torch.isfinite(x).all()` 在 aten 层分解成 `abs` / `ne` / `mul` / `all`，
   只看算子名会把校验误报成算术。登记的校验函数名里发出的调用归 `readonly_check`；**同样的算式不在校验函数里就照报不误**
   （`tests/test_host_op_audit.py::test_arithmetic_outside_a_check_site_is_still_a_violation` 钉住这条）。
4. **一个 case 一个进程**：runtime 桥只在首次算子解析时注册 vendor 树（`ASCEND_CUSTOM_OPP_PATH` 只被读一次），
   同一进程里换一个算子族再 `prepare()` 会报"已经执行过 aclnn 算子"。`--all` 因此按 case 派子进程。
   （实测踩到两次：第一次是 `--all` 单进程，第二次是交叉核对脚本一次跑三个 case。）

CPU 单测：`pytest tests/test_host_op_audit.py -q` → **27 passed**（分类表、正反例、拷贝 vs 元数据、
校验按调用点、按调用点改判要留痕、归因与脱敏、JSON 可解析、入口报错时 verdict 不读成 clean、redact 去绝对路径）。

## 2.1 "只读校验按调用点认"这条机制的全部改判（REVIEW 07:16Z 第 4 条）

这条规则会把**登记函数内**发出的 dtype / 格式 / 算术类算子改判成 `readonly_check`。它是"允许"的例外，所以要能被审：
每一行在 JSON 里带 `check_by_site: true`，每份回执的 `summary.reclassified_by_site` 列出 函数 × 算子 × 次数，
`summary.check_site_functions` 带每个名字的理由。**本轮全部 20 份回执里被改判的只有下面这些**：

| 函数（调用点） | 算子 | 次数（全部 case 合计） | 出现在哪些 case |
|---|---|---|---|
| `_check_input_domain`（`ascend_fla/ops/kda/chunk.py:187`） | `aten._to_copy.default` | 4 | kda.prepared_grad, kda.prepared_nograd |
| `_check_input_domain`（`ascend_fla/ops/kda/chunk.py:187`） | `aten.linalg_vector_norm.default` | 4 | kda.prepared_grad, kda.prepared_nograd |
| `_gate_span`（`ascend_fla/ops/kda/chunk.py:475`） | `aten.cumsum.default` | 7 | kda.bf16, kda.prepared_grad, kda.prepared_nograd, kda.rawflags_grad, kda.rawflags_nograd, kda.upstream_impl |
| `_gate_span`（`ascend_fla/ops/kda/chunk.py:476`） | `aten.sub.Tensor` | 7 | kda.bf16, kda.prepared_grad, kda.prepared_nograd, kda.rawflags_grad, kda.rawflags_nograd, kda.upstream_impl |
| `_validate`（`ascend_fla/ops/pgdn_chunk_fwd.py:112`） | `aten.sum.dim_IntList` | 4 | pgdn.bf16, pgdn.fp32 |

合计 26 次，全部是判定用的算式：`cumsum` / `sub` 在算门控跨度，`linalg_vector_norm` + `_to_copy` 在算 q/k 行范数，
`sum.dim_IntList` 在 GDN / PGDN 的 `_validate` 里做有限性与域判定。**没有一次是产出进入计算的张量的算术**
（读者可自行核对：每条都能在对应 case 的 `rows` 里按 `check_by_site` 过滤出来）。

名单本身（只收 `ascend_fla/ops/**` 里真实存在的函数，每个带理由）：

| 名字 | 为什么算只读校验 |
|---|---|
| `_gate_span` | kda/chunk.py:467 —— 算 chunk 内门控跨度用于判定上限，返回 float 判据，不产出进入计算的张量 |
| `_check_gate_range` | kda/chunk.py:479 —— 拿 _gate_span 的值与 MAX_GATE_SPAN 比，超了报错 |
| `_check_input_domain` | kda/chunk.py:164 —— 判 g<=0 / 0<beta<1 / q,k 行范数，只产出 bool |
| `_validate_raw_inputs` | kda/chunk.py:47 —— raw 输入的 dtype / 形状 / 连续性检查 |
| `_check` | kda/chunk_bwd.py:145 —— 反向入口的形状 / dtype / caches 校验，返回维度元组 |
| `_validate` | gdn_chunk_fwd.py:58、gdn_chunk_bwd.py、pgdn_chunk_fwd.py:62 —— 各入口的 ABI 与域校验 |
| `_check_options` | gdn2_chunk_fwd.py:45 —— device / block_dim 取值检查 |
| `_constant` | pgdn_chunk_fwd.py:57 —— 常量参数与期望值比对 |

反过来的保证也有单测钉住：**同样的算式写在名单外的函数里，照报违规**
（`tests/test_host_op_audit.py::test_arithmetic_outside_a_check_site_is_still_a_violation`），
所以这条机制不会把真的计算藏起来。PKDA FP32 的那 6 次就是例子：它的校验写在入口函数体里、不在名单内，
工具照报（§3.2）。

## 3. 清单（逐处违规）

### 3.1 KDA

- **推理路径干净**：raw flags 关 / 开两条都是 `clean`。分配 + 元数据 + 只读校验，没有 dtype / 格式 / 算术。
  FMT-02 与 BF-07 的效果在这里是可见的：布局与 dtype 转换已经不在 host。
- **decode 干净**：`fused_recurrent_kda` 两条 dtype 路径各只有 3 次分配。
- **训练路径仍有三处**（都是已登记的）：
  - `_scan_states`：`h` / `v_new` 的 host 复算——8 次 `_to_copy`、10 次算术（`bmm` / `mul` / `add` 等）、2 次 `stack`
    （在本工具里记为格式转换，因为它搬数据）。这是 D-PM-42 登记的整体例外。
  - `chunk_kda_bwd:286` 的 `dw = -d_vh`：一次 `neg`。
  - raw flags + 需要梯度时，`_prepare_inputs` / `_l2norm` 的前处理计算图（BF-08 在做），另外 autograd 引擎对这段
    算术求导又产生 34 次反向算子——它们在 Python 栈上没有我们的帧，工具标成
    `<autograd engine: backward of host-side math>`，**不是第二处违规，是第一处的导数**。
- **`impl="upstream"` 在当前 pin 上编译不过，但在上一个 pin 上能编能跑 —— 这是 pin 的问题，不是这条路径本身坏**：
  `PassError: local_mutex: cube needs 34 mutex IDs (maximum 32) at #18 loc(".../kda_bwd/kernels/inverse_mm.py:65:15")`。
  注意报错的是**上游反向单元** `kda_bwd/kernels/inverse_mm.py`：`impl` 同时选前向与反向两条链（`chunk.py` 的 docstring），
  所以给带缓存前向选 upstream 会把上游 `kda_bwd` 一起编。
  为了区分"这条链本身超预算" / "新 pass 拒绝了它" / "我的调用方式不对"，固定 kernels 解出件（`b3b3f9c`）只换 library：

  | library | block_dim | 结果 |
  |---|---|---|
  | `90cfcdc`（当前 pin） | 1 | FAILED，34 > 32，`inverse_mm.py:65` |
  | `90cfcdc`（当前 pin） | 2 | FAILED，同一 kernel、同一计数 |
  | `627f55f`（上一个 pin，D-PM-13 时用的那个） | 1 | **OK，编过且跑出结果**（`o` 形状 `(1,128,2,128)`） |

  两个 block_dim 同样的计数 → 与核组数无关，不是调用方式问题。老 library 能编能跑、新 library 拒绝 → 是
  `627f55f..90cfcdc` 之间**新加的 `ascriptor/passes/local_mutex.py`** 在拒绝它（该文件在这两个修订之间才出现，
  `git diff --stat` 可见）。**没有改任何 kernel 源码，也没有绕过这个 pass。**
  影响：当前 pin 下 `impl="upstream"` 整条路径不可用（不只是审计）。
  **`log2(eg)` 分支到底在哪条路径上（PM 04:25Z 要求核实，以文件:行为准）**：`chunk.py:551–563` —— `impl == "stable"` 时
  gate kernel **同时**写出 `g_cumsum` 与 `eg`；否则 `g_cumsum = None`、只给 kernel 传 `eg`。
  而 `chunk.py:784–787` 的分支是 `if chain["g_cumsum"] is not None: ... else: tok(chain["eg"].log2())`。
  所以 **`log2(eg)` 只在 `impl="upstream"` 上走，stable 的带缓存前向永不经过它**。
  两份 trace 与这个读法一致：stable 的 `chunk_kda_fwd_with_caches`（130 次）里没有 `aten.log2.default`；
  upstream 那份（在 `627f55f` 上补测）有且仅有 1 次，在 `chunk.py:787`。
  因此"当前 pin 上无法实测"这句话**只针对 `impl="upstream"` 这条对照路径**，与公开 stable 路径无关。
  **补测**：既然 `627f55f` 能编能跑，就在那个 library 上把这个 case 审了一遍
  （`benchmarks/evidence/host_op_audit/run-a5-lib627f55f/`，**library 修订与主清单不同，单独放、不与主清单混用**）：
  130 次调用，D-PM-42 登记的 `log2(eg)` 分支如期被查出——`aten.log2.default` 在 `chunk.py:787`
  `chunk_kda_fwd_with_caches` 1 次；其余 20 次违规与 stable 路径一样落在 `_scan_states`。
  机制（读 pass 源码 + 报错自带的 census，供 PM 判）：新 pass 按"每个需要保护的分配各占一段 ID、不复用"累加，
  `count += slots` 超过 32 就报错；老 library 没有这个 pass，A5 默认走 events 策略，不存在 ID 预算。
  census（`run-a5-final/console.log:639`，16 个分配合计 34）：`_l0a` 2、`_l0b` 2、`l1_do` 2、`l1_vnew` 2、`l1_dv` 2、
  `l1_h` 2、`l1_dh` 2、`l1_Akk` 3、`l1_dw` 3、`l0c_dqg` 1、`l0c_dkg` 1、`l0c_dvh` 2、`l0c_dvbeta` 2、`l0c_dkbetag` 2、
  `dvh_f_ub` 3、`out_f_ub` 3。
  所以两种可能：① pass 的计数是上界（不复用生命周期不重叠的 buffer）→ 库侧问题；
  ② 这些 buffer 真的同时活着 → 上游单元要改写。**区分这两条要做 buffer 生命周期分析，属于 ascriptor 库侧，
  不在本任务写集**。另外新 library 保留了 `a5_sync_strategy="events"` 选项，PM 可先用它验证。

### 3.2 GDN / PGDN / PKDA

- `chunk_gdn` / `chunk_gdn_bwd` 两条 dtype 路径都 **clean**（各 18 / 14 次，全是分配）。BF-02 之后符合通则。
- `chunk_pgdn` 两条路径 **clean**：71 次里 45 次是只读校验（`_validate` 里的有限性 / 域判定），24 次分配、2 次元数据。
- `chunk_precond_kda`：
  - **BF16 路径 clean（29 次，28 次分配 + 1 次元数据）**。
  - **FP32 路径 6 次算术**（工具按算子表报 `forbidden_arithmetic`，因为调用点 `chunk_precond_kda` 不在
    `CHECK_SITE_FUNCTIONS` 里；**按 REVIEW 07:16Z 的口径，它们是 D-PM-42 ① 的只读校验，不算 D-PM-37 违规**），
    全在 `pkda_chunk_fwd.py:126`，即 `_module('ref.reference').validate_inputs(inputs)`：
    `pow` / `sum` / `neg` 在算 key 行的 L2 范数与域判定。
  - **两条路径的差别值得单独看**：BF16 在 `:102` 直接进 `_bf16_module().execute(...)`，域校验由**kernel 侧的状态字**完成
    （`kernels/projects/a5/pkda_chunk_fwd_bf16/kernels/pipeline.py` 的 `ERRORS` 1–7，含 "key row L2 norm must be <=1+1e-5"），
    所以 host 上一个校验算子都看不到；FP32 仍在 host 上算。证据就是这两份 trace 的差别。
    **这正是 `bf16-kernel-side.md` 里 D-PM-42 ① 说的"控制回读"已经在本仓落地的一个例子**，
    建议把 FP32 路径也改成同一形态（见 §6 建议 A）。

### 3.3 GDN-2（仓主轨道，只跑不改）

`chunk_gdn2` BF16 路径 4 次 dtype 转换：`:111` 三次 BF16→FP32 加宽、`:128` 一次 FP32→BF16 回转；FP32 路径 clean。
与 `gaps.json` 的 "GDN-2 BF16 加宽" 一致，BF-05 在做。按写集要求**只跑不改**，结果照列。

## 4. 阳性 / 阴性对照（对着规格 2026-09-21 更新那一节）

| 期望 | 实测 | 一致？ |
|---|---|---|
| 阳性：D-PM-42 的 KDA 存量例外（`_scan_states` 整体、`dw` 取负） | `_scan_states` 20 次、`neg` 1 次，逐行定位 | 是 |
| 阳性：`log2(eg)` 分支（**只在 `impl="upstream"` 上，见 §3.1 的 `chunk.py:551–563` / `:784–787`**） | 当前 pin 编译不过；**在 `627f55f` 上补测到了**：`aten.log2.default` @ `chunk.py:787` ×1 | 是（library 修订不同，证据单独目录） |
| 阳性：KDA 训练 raw-flag 前处理（BF-08 在做） | `_prepare_inputs` / `_l2norm` + 引擎侧反向，共 55 次算术 / 18 次 cast | 是 |
| 阳性：GDN-2 BF16 加宽（仓主轨道，BF-05 在做） | 4 次 cast，定位到两行 | 是 |
| 阴性：PKDA FP32 应干净 | **6 次算术**：是只读校验，但在 host 上算、且写在未登记的函数里（§3.2） | 与期望不符 —— 新发现（归类为"未登记的只读校验"，非 D-PM-37 违规） |
| 阴性：KDA 推理 / decode（raw flags 关 / 开） | clean | 是 |
| 阴性：GDN / PGDN / PKDA 的 BF16 | 全 clean | 是 |

规格明确要求"报出的与期望不符的地方如实列出，不要为了对上期望去调分类表"——§3.2 的 PKDA FP32 就是这一条，
分类表没有为它开口子。

## 5. 交叉核对（与已合入任务自带的只读审计驱动）

在同一次入口调用里同时开本工具与 `kernels/projects/a5/kda_prep/native_audit.py` 的 `Audit`（只读引用，未改动），
逐算子比对（`tmp/` 下的一次性脚本，不在写集内）：

| case | 本工具 | native_audit | 逐算子差异 |
|---|---|---|---|
| `chunk_kda` 推理（raw flags 关） | 71 | 71 | 无 |
| `chunk_kda` 训练（raw flags 关） | 256 | 253 | 只有 `aten.detach.default`：12 vs 9 |
| `chunk_kda_fwd_with_caches`（stable） | 130 | 130 | 无 |

**那 3 次 `detach` 是两个 dispatch mode 叠在一起的产物，不是分歧**：本工具单独跑同一个 case 时总数是 **253**
（`run-a5-final/kda.chunk_kda.prepared_grad.json` 的 `total_calls`），与 native_audit 单独记的 253 相同；
两个 mode 同时开时，内层 mode 调 `func(*args)` 会再经过外层 mode，`detach` 这种会被多记几次。

**类别口径不同是设计差异**：

- `native_audit` 按**调用点**给 D-PM-42 的例外各起一个专名，并把该函数内的**全部**调用都算进去——
  所以它在训练路径上报 `scan_states` 52 次（含该函数里的分配与 view），`dw` 取负 1 次；
  本工具只把其中真正违规的算子计入禁止类（dtype 10 + 格式 2 + 算术 12 = 24），分配与 view 仍归允许面。
  两种读法都对，用途不同：它回答"这个例外一共发出多少调用"，本工具回答"哪些调用违规、违哪一类"。
- 边界差一条：`native_audit` 把一次 `aten.view.default` 归进只读校验（它发生在校验函数里），
  本工具按算子表把 view 归 `meta_view`。三个 case 里都是这同一条。

## 6. 建议的改正任务拆分（供 PM 建任务）

- **建议 A（新，P2）**：PKDA FP32 路径的域校验搬成 kernel 侧状态字 + host 回读，与它自己的 BF16 兄弟单元一致。
  （它本身不是 D-PM-37 违规，所以不是 P0/P1；收益是少一次 host 侧同步开销、并让审计不需要人工判断。）
  范围：`ascend_fla/ops/pkda_chunk_fwd.py` 的 `:126` 一处调用 + 可能的派生单元改动；验收：审计报 clean，
  且拒绝面不变（同样的非法输入仍然报错，错误信息与现在等价）。工作量估计：小（半天到一天），需要 kernel 侧支持状态字。
- **建议 B（已在飞）**：KDA 训练 raw-flag 前处理（BF-08）；合入后应看到 `_prepare_inputs` / `_l2norm` 与引擎侧反向一起消失。
- **建议 C（已在飞）**：GDN-2 BF16 加宽（BF-05，仓主轨道）。
- **建议 D（kernel 批次）**：`_scan_states` 与 `dw` 取负这两处 D-PM-42 存量例外，按 §6.5 进 kernel 批次统一修。
- **建议 E（pin）**：已按 `RISK contradicts-handoff` 上报；PM 04:25Z 回复这是已登记条目
  `gaps.json` 的 `kda-bwd-inverse-mm-mutex-over-budget`（P1，根因由 A5K-02 定位：`l0c_dvh` / `l0c_dvbeta` 两块 L0C 是 `DBuff`，
  改单槽 `Tensor` 即 34 → 32；本仓 stable 反向已用资源受限版，公开路径不触发）。本节保留是为了记下**这台机器上的对照数据**。
  当前 compatibility pin（library `90cfcdc`）新加的
  `local_mutex` pass 拒绝上游 `kda_bwd/kernels/inverse_mm.py`（34 > 32），而上一个 pin `627f55f` 能编能跑。
  也就是说**换 pin 让 `impl="upstream"` 整条路径在当前 pin 下不可用**。要判的是：上游单元改写降 ID 占用、
  pass 的预算算法是否偏保守、还是把 `impl="upstream"` 正式退役。这条超出本任务写集，交 PM / 用户。

## 7. 局限（每条都影响怎么读这份清单）

1. **只看得到 aten 层**：runtime 桥用 ctypes 直调 aclnn 的那一步不经过 dispatch，本工具看不到——那是我们自编译
   kernel 的 launch，属于允许，JSON 里以 `bridge_launch_invisible` 说明。`torch_npu` 的 `npu_*` 自定义算子会经过
   dispatch，单列为 `npu_custom_op`；本次清单里没有出现。
2. **归因到行依赖 Python 栈**：autograd 引擎的反向节点在 C++ 里跑，只能标到
   `<autograd engine: backward of host-side math>`，没有行号。
3. **"只读校验"的边界是人定的**：`CHECK_SITE_FUNCTIONS` 是显式名单，新增要写理由。PKDA FP32 的校验写在入口函数体里
   （不是具名校验函数），所以本工具按算术报——这是有意的，不是误报：D-PM-42 ① 的允许面是"只读校验"，
   而"写在哪里"决定了它能不能被自动识别。
4. **环境特定**：CANN 9.3.0 / torch 2.7.1 的 aten 分解与 9.1.0-beta.1 / 9.2.0 + torch 2.12 可能不同；
   跨任务对照时以各自的环境行为准。
5. **形状固定**：每个 case 用小形状（B=1、T=128、H/HV ≤ 2，GDN-2 H=1）。host 侧发出的算子与形状基本无关，
   但**这是假设不是实测**；真实形状下是否多出别的 host 调用没有扫。
6. **不含 layer / module 层**：按通则范围只审 `ascend_fla/ops/**`。`layers/` 与 `modules/` 本来就是 torch 算子拼的
   （见 `gaps.json` 的 `modules-are-torch-not-kernels`），不在本清单内。
