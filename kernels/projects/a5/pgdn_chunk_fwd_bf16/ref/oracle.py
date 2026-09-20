"""Literal pinned FLA PGDN oracle, used only in CPU validation."""
import functools
import hashlib
import importlib.util
import os
from pathlib import Path

import torch

PIN = 'e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'
SHA256 = '3baa67a5f35dc7230698e3f1761ec8675131318c15d4a27ed7f2fce11e84b5e8'


@functools.lru_cache(maxsize=1)
def load():
    path = Path(os.environ['FLA_PGDN_NAIVE'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f'FLA_PGDN_NAIVE must match FLA pin {PIN}')
    spec = importlib.util.spec_from_file_location('_bf03_pinned_pgdn_naive', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.naive_recurrent_precond_gated_delta_rule


def reference(inputs):
    names = ('q', 'k', 'v', 'g_atk', 'g', 'beta_atk', 'beta')
    with torch.no_grad():
        result = load()(*(inputs[name] for name in names), output_final_state=True)
    return dict(zip(('o', 'final_state', 'final_A_state'), result))
