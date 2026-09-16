"""CCE kernel entry for the GDN-2 recurrent unit."""

from .step import gdn2_fused_recurrent_kernel

__all__ = ["gdn2_fused_recurrent_kernel"]
