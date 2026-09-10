"""ascend_fla — fla 系列线性注意力算子的昇腾 NPU 高效实现。

本包是独立算子库，不是 fla 的后端插件：公共 API 由本包自己定义，
不注册进 fla 的 BackendRegistry。定位与决策见根目录 AGENTS.md。

当前状态：phase 0（支持矩阵已建立，尚无实现）。见 docs/matrix/README.md。
"""

__version__ = "0.0.1.dev0"
