"""Literal pinned FLA FP32 naive; dependency exists only for validation."""
import functools,hashlib,importlib.util,os
from pathlib import Path
import torch

PIN='e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'
SHA256='d1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb'


@functools.lru_cache(maxsize=1)
def load():
    path=Path(os.environ['FLA_GDN_NAIVE'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SHA256:
        raise ValueError(f'FLA_GDN_NAIVE must match FLA pin {PIN}')
    spec=importlib.util.spec_from_file_location('_bf01_pinned_gdn_naive',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.naive_recurrent_gated_delta_rule


def reference(inputs):
    q,k,v,beta,g=(inputs[n] for n in ('q','k','v','beta','g'))
    ratio=v.shape[2]//q.shape[2]
    if ratio!=1:q,k=(x.repeat_interleave(ratio,dim=2) for x in (q,k))
    with torch.no_grad():
        o,s=load()(q,k,v,beta,g,scale=128**-.5,output_final_state=True)
    return dict(o=o,final_state=s)
