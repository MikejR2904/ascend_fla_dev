"""CCE kernel entry for the model-specific GDN-2 BF16 decode unit."""

from .step import gdn2_fused_decode_kernel

__all__ = ["gdn2_fused_decode_kernel"]
