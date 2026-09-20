# a2.kda_bwd_stable：A2（c220）上的 KDA chunk 反向

> 结论范围：Ascend 910B3 / CANN 9.0.0 / ascriptor library `90cfcdc`、kernels `b3b3f9c`。
> 按 `AGENTS.md` §2，A2-11 之前 A2 真机数字**只作观测**，契约的 board stage 因此标 `untested`。

## 这是什么

九个 kernel 组成的 KDA 反向，数学与精度边界沿用上游 `a5/kda_bwd`；`inverse_mm` 取本仓
`a5/kda_bwd_stable` 的 bounded 版（D-PM-24），`finalize_pre` / `finalize_post` 取本仓的中点锚定版。

| 阶段 | kernel | 输入 → 输出 |
|---|---|---|
| scan | `scan_fused_a2_kernel` | 反向 chunk 扫描，`dstate` 常驻 UB → `dAqk`、`dh`、`dv`、`dh0` |
| inverse_mm | `inverse_mm_a2_kernel` | 五个 WY 逆梯度积 → `d_qg`、`d_kg`、`d_w`、`d_v_beta`、`d_k_beta_g` |
| epilogue | `inverse_epilogue_a2_kernel` | 去门控 → `dq_hv`、`dk_hv`、`dv`、`dbeta`、`dg_core`、`k_exp` |
| dainv | `inverse_dainv_a2_kernel` | `dtri = beta[col] * (dv@v^T + d_w@k_exp^T)`，严格下三角 |
| dakk | `inverse_dakk_fused_a2_kernel` | `dAkk = -tril(Akk^T dtri Akk^T, -1)` |
| pre | `finalize_pre_a2_kernel` | 门控锚定的 q/k 缩放与三个 mask 矩阵 |
| pair | `finalize_pair_a2_kernel` | 四个成对衰减积（纯 cube） |
| post | `finalize_post_a2_kernel` | 折回 dq/dk/dbeta/dg，dg 是行反向累加 |
| reduce | `finalize_reduce_a2_kernel` | GQA 组内求和 → 公共 `dq`、`dk` |

**ABI 全是 BF16**，与 a5 单元一致：公共输入 `q k v g beta initial_state do dht`，加九个前向检查点
（`g_cumsum Aqk Akk w u qg kg v_new h`，`g_cumsum` 是 chunk 内 cumsum 除以 ln2，即 log2 空间）；
输出六项梯度 `dq dk dv dbeta dg dh0`。token-major 张量由 kernel 按 2-D 视图直读直写，
host 只做 NaN 预填分配与不拷贝的 `view`。

## 与 a5 版本的差别，以及原因

1. **没有 `@vf`**。向量体全部是 UB 上的 tile 指令，逐行标量用 `Var.GetValueFrom`。
2. **b3 没有 `dma.ub_to_l1` 与 `dma.l0c_to_ub`**。跨侧交接全改成两槽 `GMBuff` 环，一个环一个 beat
   计数器，各自的 `VcMutex`（MTE3→MTE2）或 `CvMutex`（FIX→MTE2）。`scan_fused` 有七个这样的环。
3. **L0C 只有 128KB**。`scan_fused` 的五个积共用三个累加器（`[128,128]` + `[64,128]` + `[64,64]`
   = 112KB），每次复用前显式 `barrier(Pipe.M)`；`finalize_pair` 的四个 `[64,128]` 累加器正好占满，
   不共用（共用过，真机错）。
4. **UB 只有 192KB**。`finalize_pre` / `finalize_post` / `inverse_epilogue` 把 chunk 拆成两半、
   每半 32 行处理，逐行输入以 BF16 常驻 UB、逐行 cast。`finalize_post` 先跑上半，好让 dg 的
   反向累加仍然是 63→0；`inverse_epilogue` 的 dg 例外地整 chunk 常驻，因为末行的修正要等所有行。
5. **D-PM-37**：a5 在 host 做的两件事挪进 kernel —— `g_last` 不再由 host 从 `g_cumsum` 切片
   （`scan_fused` 直接跨步读该 chunk 的门控末行），`inverse_mm` 自己取负并直接输出 `d_w`。
   a5 折在 host 的 qg 预乘 `1/sqrt(128)` 也进了 `scan_fused` 的 seed。

## 真机上踩到的事（只在 c220 上出现，功能模拟器全绿）

这些是本单元绕行的记录，不是 ascriptor 的修复；已在任务 issue #112 报 RISK。

1. **BF16 标量读编译不过**。`Var.GetValueFrom(bf16_gm[...])` 要一次标量 bf16→float 转换，
   bisheng 直接报 `not support bf16 type cast`。绕行：`gm_to_ub_pad` 进 UB，向量侧 cast，再读标量。
2. **整块 cast 一个 32 字节行包会被降解成 `srcBlk=0`**，于是第 0 行的 block 被复制到每一行，
   真机上整个 chunk 的每个 token 都拿到 `beta[0]`。改成逐行 `count=1` 的 cast。
3. **`auto_sync` 的保护列表不完整（观测，未观察到后果）**：`finalize_pair` 的 `l1_q`、
   `inverse_mm` 的 `l1_akk`/`l1_dv`/`l1_vnew`、`scan_fused` 的 `l1_w` 都不在生成代码的
   MTE2→MTE1 ready 列表里；几个输出暂存缓冲也缺 V→MTE3。
   **但受控 A/B（同一 case、同一 build，只差这两条栅栏）在 910B3 上输出逐位相同** ——
   所以这只是生成代码的观测，不是已证实的缺陷。栅栏留着当保险，代码里按「预防」注明。
   我一度把它当成根因，是因为同一个 commit 里还改了判据；真正的根因是第 2 条。
4. **功能模拟器不校验物理地址分配**，pipesim 与真机会。四个 kernel 在 sim 下全绿，
   pipesim 才报 UB / L0C 溢出。**报 sim 通过之前，每个 kernel 至少跑一次
   `ascriptor compile --backend cce`。**

## 运行

```bash
PYTHONPATH=<ascriptor>/library python run.py reference
PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher sim
PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher pipesim
ASCEND_RT_VISIBLE_DEVICES=<card> PYTHONPATH=<ascriptor>/library python run.py check --device a2 --backend cce --launcher aclnn
```

契约 26 个 case：a5 的五个 case 各跑 bd1/bd2；(C ∈ {1,2}) × (HV ∈ {1,2,4}) × (bd ∈ {1,2}) 的
边界网格；一个 C=3（奇数 chunk 数是结构性分档，见 `AGENTS.md` §6）；一个零初始 state；一个 B=2。

## 预算

六项输出沿用 a5 的预算：allclose rtol/atol 1e-3 加 `max_relative_l2` 0.05，`dk` 放宽到 0.15、
`dg` 放宽到 0.25（理由见契约，是 a5 的实测理由，不是 A2 的新结论）。

逐 kernel 的 33 个 checkpoint 里，`finalize_pair` 的四个积单独一档。它们是 64 项的 FP32 点积，
操作数还带着**未抵消**的成对衰减：910B3 实测操作数跨 13 个数量级（`|k_scaled|` 到 1.8e+13），
和到 3.3e+08，项与项之间灾难性相消。真机与 torch 的求和顺序不同，于是相消附近的元素能差几十个
BF16 ulp，而向量范数不受影响。四个 case（C=1/2/3、HV=1/2/4）实测：最大 58 ulp，
超过 1 ulp 的元素占比 ≤ 0.41%，相对 L2 最大 1.06e-07。
所以这四个的判据取**相对 L2 1e-5**（比实测宽两个数量级），逐元素的 rtol 取 1.0 只当粗错闸
（58 ulp 约等于值的 0.23~0.45，1.0 是两倍余量）。`finalize_pre` 的三个缩放量**一点都不用放宽**
（0 ulp，相对 L2 ≤ 3.7e-27），用默认预算。

这一档是**看到真机结果之后定的**，如实记在这里；六项输出的预算一个字没动。

## 没有确立的

- `block_dim` 只跑过 1 与 2；a5 的上限不继承。
- 门控跨度的 A2 上限没测，a5 反向的 105 不继承，留给 A2-12。
- 性能没有测。这个版本逐行发指令，是正确性优先的写法。
