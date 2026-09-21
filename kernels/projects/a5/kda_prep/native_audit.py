"""Actual TorchDispatch provenance including prep and retained training math."""
import collections
import traceback
from pathlib import Path

from torch.utils._python_dispatch import TorchDispatchMode

REPO=Path(__file__).resolve().parents[4]


class Audit(TorchDispatchMode):
    allowed={'aten.empty.memory_format','aten.empty_strided.default','aten.view.default',
             'aten._unsafe_view.default','aten.as_strided.default','aten.slice.Tensor',
             'aten.select.int','aten.detach.default','aten.alias.default',
             'aten.unsqueeze.default','aten.squeeze.dim','aten.expand.default'}

    def __init__(self):
        super().__init__()
        self.rows=collections.Counter()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        scope=[f for f in traceback.extract_stack() if '/ascend_fla/ops/kda/' in f.filename
               or '/kda_layout/runtime.py' in f.filename or '/kda_prep/runtime.py' in f.filename]
        origin=scope[-1] if scope else None
        category='operator'
        if any(f.name=='_prepare_inputs' for f in scope):
            category='BF08_pending_training_host_preparation_not_compliant'
        elif any(f.name=='_scan_states' for f in scope):
            category='DPM42_scan_states_whole_exception_not_compliant'
        elif any(f.name in ('_gate_span','_check_gate_range','_check_input_domain') for f in scope):
            category='readonly_validation_DPM42_provisional'
        elif origin and origin.name=='chunk_kda_bwd' and str(func)=='aten.neg.default' and origin.line=='dw = -d_vh':
            category='DPM42_dw_negation_exception_not_compliant'
        elif origin and origin.name=='chunk_kda_fwd_with_caches' and str(func)=='aten.log2.default':
            category='DPM42_upstream_log2_exception_not_compliant'
        file=str(Path(origin.filename).relative_to(REPO)) if origin else '<runtime bridge>'
        self.rows[(category,file,origin.lineno if origin else 0,str(func))]+=1
        return func(*args,**(kwargs or {}))

    def report(self):
        return [dict(category=c,file=f,line=l,operator=o,count=n) for (c,f,l,o),n in sorted(self.rows.items())]

    def unexpected(self):
        return [r for r in self.report() if r['category']=='operator' and r['operator'] not in self.allowed]
