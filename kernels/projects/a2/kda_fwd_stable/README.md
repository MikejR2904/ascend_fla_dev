# a2.kda_fwd_stable：A2（c220）上的 KDA chunk 前向

> 结论范围：Ascend 910B3 / CANN 9.0.0 / ascriptor library `90cfcdc`、kernels `b3b3f9c`。
> 按 `AGENTS.md` §2，A2-11 之前 A2 真机数字**只作观测**，契约的 board stage 因此标 `untested`。

## 这是什么

五个 kernel 组成的 KDA 前向，数学与精度边界沿用本仓 `kernels/projects/a5/kda_fwd_stable`。
inverse 来自 ascriptor 上游 `a5/kda_fwd`，只读引用。代码按 A2 facade 重写：

| 阶段 | kernel | 文件 | 输入 → 输出 |
|---|---|---|---|
| gate | `kda_sub1_gate_a2_kernel` | `kernels/gate.py` | `g_raw` → `g_cumsum`、`eg` |
| scores | `kda_sub2_score_a2_kernel` | `kernels/intra.py` | `q`、`k`、`g_cumsum`、`beta` → `Aqk`（BF16）、`strict`（FP32） |
| inverse | `tril_inverse64_a2_kernel` | `kernels/triangular_inverse.py` | `strict` → `Akk = (I − strict)⁻¹`（BF16） |
| wy | `kda_sub3_wy_a2_kernel` | `kernels/wy.py` | `q`、`k`、`v`、`beta`、`Akk`、`g_cumsum` → `w`、`u`、`qg`、`kg` |
| recurrent | `kda_sub45_a2_kernel` | `kernels/recurrent.py` | `q`、`Aqk`、`kg`、`w`、`u`、`eg`、`h0` → `o`（BF16）、`final_state` |

**公共 ABI 是 token-major**：`q`/`k` 为 `[B, T, H, 128]` BF16，`v` 为 `[B, T, HV, 128]` BF16，`g_raw` 为 `[B, T, HV, 128]` FP32，
`beta` 为 `[B, T, HV]` FP32，`initial_state` 为 `[B, HV, 128, 128]` FP32；输出 `o` 为 `[B, T, HV, 128]` BF16。
kernel 把这些连续张量当 2-D 视图（如 `[B*T, H*128]`）直接读写。BF16→FP32 转换、`scale`、GQA 头映射
（`h = hv // (HV // H)`）都在 kernel 里做。host 只做两件事：输出用 NaN 预填的分配，以及不拷贝的 `view`（D-PM-35/37）。
链内中间量（`g_cumsum`、`eg`、`Aqk`、`strict`、`Akk`、`w`、`u`、`qg`、`kg`）保持 A5 的 `[B, HV, C, 64, *]`。

## 与 A5 版本的差别，以及原因

在 pin 版库上实测（`ascriptor check` / verifier / CCE 后端）：

1. **a2 facade 没有 `@vf`**（只有 A5 有寄存器形式）。向量体全部改成 UB 指令，逐行标量用 `Var.GetValueFrom`。运算顺序照 A5，gate 的 `g_cumsum` 与顺序 FP32 累加逐位相同。
2. **device b3 上 `dma.ub_to_l1` 与 `dma.l0c_to_ub` 不可用**（verifier 原话 "not available on device b3"）。每个跨侧交接改成两槽 `GMBuff` 环：
   - 生产者用 beat 索引，读者用 beat − K；库里的 GMBuff pass 要求一个环只用一个计数器。
   - 由 `VcMutex`（MTE3→MTE2）或 `CvMutex`（FIX→MTE2）保护，写入落在所在分支自己的 lock..ready 窗口里。
3. **c220 没有 FP32 的 L0C→L1**。CCE 后端原话："fp32 -> fp32 l0c_to_l1 has no c220 spelling … quant to half/bf16, or route through UB"。
   inverse 的 FP32 中间积因此改走 GM workspace，没有量化成 BF16，保住 A5 的 FP32 精度。
   同核 GM 的写后读（FIX 写 → MTE2 读）autosync 不排，pipesim 会报 RAW hazard，所以显式加 `DEvent(FIX→MTE2)`；深度为 1 时 autosync 会拒，理由是独立发布可能让 FIX 连发两次。
4. **A2-01 / ascriptor M10-081**：所有 `is_init=False` 累加前都显式 `barrier(Pipe.M)`，不分 dtype。
   - inverse 的 FP32 链，正是 M16 失效形状。
   - recurrent 的 BF16 输出链。
   - 不使用 BF16/FP16 的 `splitk`。
   - scores 的 FP32 `splitk=64` 由 pin 的 desugar 规则自动 settle。
   - a2 lint 0 trap。
5. **recurrent** 的五个跨侧交接（`h`、`qg'`、`v_new` 给 cube；`prod`、`delta` 回向量）都走 GM 环，由 `auto_sync` 排序，取代 A5 手写的 31 个事件。
   A5K-01 修的 Aqk L1 轮转信用在这个设计里不存在：`Aqk` 每个 chunk 载入 cube 本地缓冲。C=1 多头、奇数 C 在真机上与仿真逐位一致。
6. A2 的 `dup` 没有 BF16 形式，零块先用 FP32 dup 再 cast。下三角掩码由 kernel 内建的列号行加 `compare_scalar` + `select` 生成，不由 host 造掩码张量。

## 运行

```bash
PYTHONPATH=<ascriptor>/library python run.py reference
PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher sim
PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher pipesim
ASCEND_RT_VISIBLE_DEVICES=<card> PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher aclnn
```

契约共 20 个 case：
- 沿用 a5 单元的 8 个：窄门控、深门控 80×/140×、分组头、C=1/2/3/5。
- A2-03 的边界网格 12 个：(C ∈ {1, 2}) × (HV ∈ {1, 2, 4}) × (bd ∈ {1, 2})，H=1，所以 HV>1 都是分组。

比较预算沿用 a5 单元：allclose rtol/atol 0.02，`max_relative_l2` 0.05。o 对 FP32 参考；a5 上实测 2.85e-3 ~ 3.19e-3。

## 证据

`evidence/` 下是脱敏的原始日志与回执，数字的汇总见任务 issue #34 的 STATUS。内容：
- 每个 kernel 单独的 sim、pipesim、真机逐项比对，含 bd1 与 bd2 的输出哈希；
- 整链的 `run.py check` 结果：sim、pipesim、aclnn。

## 没有确立的

- `block_dim` 只跑过 1 与 2；A5 的上限 4 不继承。
- 门控跨度的 A2 上限没测，A5 的 155 / 174.7 不继承，留给 A2-12。
- 性能没有测。
- 反向（`kda_bwd_stable`）是 A2-09。
