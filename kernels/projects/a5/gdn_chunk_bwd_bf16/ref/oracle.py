"""Test-only A oracle: explicitly selected, hash-checked pinned FLA source."""
import ast
import functools
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
import torch

SHA256 = 'd1cf17992349fd3e94af999b22e3d3a81be4a2d1881ce5b70a3457257166e0cb'
PIN = 'e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2'


@functools.lru_cache(None)
def load(fp64=False):
    path = Path(os.environ['FLA_GDN_NAIVE'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError('FLA_GDN_NAIVE does not match the accepted FLA source')
    spec = importlib.util.spec_from_file_location('_gda03_pinned_naive', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.naive_recurrent_gated_delta_rule
    if not fp64:
        return fn
    # The literal pinned function always casts to FP32. Finite differences at
    # double epsilon therefore require a qualification-only precision lift.
    # Rewrite ONLY torch.float32 -> torch.float64 in its isolated function AST.
    # Never change the pinned file, equations, order, scale or FP32 A oracle.
    tree = ast.parse(inspect.getsource(fn))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'torch' and node.attr == 'float32':
            node.attr = 'float64'
            changed += 1
    assert changed == 2, changed
    namespace = dict(module.__dict__)
    exec(compile(tree, '<qualification-only-FLA-fp64>', 'exec'), namespace)
    return namespace[fn.__name__]


def outputs(q, k, v, g, beta, *, fp64=False):
    ratio = v.shape[2] // q.shape[2]
    return load(fp64)(q.repeat_interleave(ratio,2), k.repeat_interleave(ratio,2),
                      v, beta, g, output_final_state=True)


def autograd(q, k, v, g, beta, do=None, dht=None):
    xs = [x.detach().float().requires_grad_() for x in (q,k,v,g,beta)]
    o, ht = outputs(*xs)
    loss = (o * (torch.zeros_like(o) if do is None else do.float())).sum()
    loss = loss + (ht * (torch.zeros_like(ht) if dht is None else dht)).sum()
    grad = torch.autograd.grad(loss, xs)
    return dict(zip(('dq','dk','dv','dg','dbeta'), grad))
