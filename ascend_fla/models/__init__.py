"""models — 注入式模型支持。

不重写 modeling_*.py：用 HF transformers / fla 的模型定义，只把本仓的 layer
替换进去，使模型规格自动跟随上游。
"""
