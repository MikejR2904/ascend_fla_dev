"""runtime — 把 ascriptor kernel 变成常驻的、torch 可调用的 NPU 算子。

ascriptor 自身的执行模型是"落盘 + 独立进程"（aclnn 写参数文件跑独立 harness，
board 经 SSH 推送），输出是 CPU tensor。本模块补的就是缺失的那一层：

    compile.py   ascriptor kernel -> CANN 自定义算子 .so
    cache.py     按 (kernel, 形状签名, SoC, 版本) 缓存产物
    binding.py   torch.library 注册，直吃 NPU device tensor、零拷贝
    autograd.py  fwd/bwd 组装 autograd.Function

⚠️ 前置风险：六个 a5 单元的 compile stage 全为 untested
（见 docs/matrix/gaps.json 的 aclnn-compile-untested）。
"""
