# a2.kda_fused_recurrent：A2（c220）上的 KDA decode

> 结论范围：Ascend 910B3 / CANN 9.0.0 / ascriptor library `90cfcdc`。
> 按 `AGENTS.md` §2，A2-11 之前 A2 真机数字**只作观测**，契约的 board stage 因此标 `untested`。

## 这是什么

逐 token 的 KDA 递推，覆盖 decode（T=1）与投机解码（T ≤ 16）。state 常驻 UB，每个 token 做两趟行扫描：

```
state <- state * exp(g)            # 沿 K 的逐行衰减
delta  = v - k^T state
state <- state + (beta k) (x) delta
o      = (scale q)^T state         # 用更新后的 state
```

由本仓 `kernels/projects/a5/kda_fused_recurrent/kernels/step.py` 改写，只有一个 kernel：`kernels/step.py` 的 `kda_fused_recurrent_a2_kernel`。

## ABI（与 A5 版不同）

A5 版的 ABI 是 FP32、BHV-major、q 预乘 scale、GQA 在 host 侧展开。这些 host 侧转换现在都被禁止（D-PM-35/37）。
A2 版直接吃公共张量：

| 名称 | dtype | 形状 |
|---|---|---|
| `q`、`k` | BF16 | `[B, T, H, 128]` |
| `v` | BF16 | `[B, T, HV, 128]` |
| `g` | FP32 | `[B, T, HV, 128]`（log 空间） |
| `beta` | FP32 | `[B, T, HV]` |
| `initial_state` | FP32 | `[B, HV, 128, 128]`，key-major |
| `o`（输出） | BF16 | `[B, T, HV, 128]` |
| `final_state`（输出） | FP32 | `[B, HV, 128, 128]` |

kernel 把 token-major 张量当 2-D 视图读写。BF16→FP32 转换、`scale`、GQA 头映射都在 kernel 里做，`o` 用 round-to-nearest-even 转成 BF16。
host 只做两件事：NaN 预填的分配，以及不拷贝的 `view`。

## 向量体

A2 没有 `@vf`。每个 state 行用一条 `muls` 加一条 `add`，标量由 `Var.GetValueFrom` 从 UB 读出，累加顺序同 A5：先乘进临时量，再加。

## 预算

在跑之前写定：
- `final_state` 沿用 A5 decode 的 1e-5。真机实测 4e-8 ~ 1e-7。
- `o` 是 BF16，地板是 FP32 结果本身舍入到 BF16 的误差，实测 1.63e-3 ~ 1.71e-3。预算取 `max_relative_l2` 5e-3，约为地板的 3 倍。

## 运行

```bash
PYTHONPATH=<ascriptor>/library python run.py reference
PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher sim
ASCEND_RT_VISIBLE_DEVICES=<card> PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher aclnn
```

契约 8 个 case：单 token；GVA；T=4；T=16；Kimi 形状 H=HV=32；GQA H16/HV32；B2 T8 GQA；零初始 state。

## 没有确立的

- `block_dim` 只跑过 1 与 2。
- 性能没有测。这个版本逐行发指令，是正确性优先的写法。
