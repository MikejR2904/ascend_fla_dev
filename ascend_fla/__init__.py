"""ascend_fla — fla 系列线性注意力算子的昇腾 NPU 高效实现。

本包是独立算子库，不是 fla 的后端插件：公共 API 由本包自己定义，
不注册进 fla 的 BackendRegistry。定位与决策见根目录 AGENTS.md。

当前状态：KDA chunk fwd/bwd 与 decode 已接线；GDN-2 有 canonical 与 packed-inference
两种完整模型布局、CCE recurrent/fused-decode/packed-short-conv kernel 与 stateful NPU
Graph decode。
权威状态见 docs/matrix/README.md。
"""

__version__ = "0.0.1.dev0"
