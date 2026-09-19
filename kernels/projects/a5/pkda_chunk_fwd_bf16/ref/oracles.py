"""Load the explicitly selected FLA naive file without importing Triton."""
from __future__ import annotations
import hashlib
import importlib.util
import os
from pathlib import Path

NAIVE_SHA256 = 'ec18fab93598982c41447b8e0b3b3e3e1916b71945b131a4557033646ce187c5'


def fla_naive():
    path = Path(os.environ['FLA_PKDA_NAIVE']).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest() != NAIVE_SHA256:
        raise ValueError('FLA_PKDA_NAIVE must match the pinned e52dbc0e PKDA naive source')
    spec = importlib.util.spec_from_file_location('_pinned_fla_pkda_naive', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_precond_kda


def oracle(inputs):
    result = fla_naive()(**inputs, output_final_state=True)
    return dict(zip(('o','final_state','final_A_state'), result))


def metrics(actual, expected):
    if set(actual) != set(expected):
        raise ValueError('missing or extra outputs')
    out = {}
    for n in expected:
        a, e = actual[n], expected[n]
        if a.shape != e.shape or a.dtype != e.dtype:
            raise ValueError(f'{n}: dtype/shape mismatch')
        d = a.double() - e.double()
        out[n] = dict(relative_l2=float(d.norm()/e.double().norm().clamp_min(1e-30)),
                      max_abs=float(d.abs().max()))
    return out
