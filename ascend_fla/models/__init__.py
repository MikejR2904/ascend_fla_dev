"""模型层。

生产方向仍采用注入式集成；``gdn2`` 额外提供一个纯 torch / torch_npu 的完整基线，
用于先严格加载原始 LitGPT checkpoint、跑通模型，再按实测调用链替换昇腾算子。
"""

from .gdn2 import (
    GDN2Config,
    GDN2ForCausalLM,
    GDN2LayerCache,
    GatedDeltaNet2,
    checkpoint_state_dict,
)
from .generation import (
    GDN2_DECODE_BACKENDS,
    GDN2GenerationResult,
    GDN2GenerationTimings,
    GDN2NPUGraphDecodeRunner,
    generate_tokens,
)

__all__ = [
    "GDN2Config",
    "GDN2_DECODE_BACKENDS",
    "GDN2ForCausalLM",
    "GDN2GenerationResult",
    "GDN2GenerationTimings",
    "GDN2LayerCache",
    "GDN2NPUGraphDecodeRunner",
    "GatedDeltaNet2",
    "checkpoint_state_dict",
    "generate_tokens",
]
