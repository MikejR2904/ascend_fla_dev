"""Small diagnostics of the production DD helpers; no production dispatch path."""
from pathlib import Path
import importlib.util

from ascriptor.a5 import *

_path = Path(__file__).parent / 'kernels/backward.py'
_spec = importlib.util.spec_from_file_location('_bf09_arithmetic_source', _path)
_source = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_source)
_dd_mul = _source._dd_mul
_dd_add_constant = _source._dd_add_constant


def make_probe(kind):
    if kind not in ('product', 'bounded_product', 'constant'):
        raise ValueError(kind)

    @vf
    def calculate(a: Tensor, b: Tensor, high: Tensor, low: Tensor):
        x = Reg(DT.float)
        y = Reg(DT.float)
        z = Reg(DT.float)
        h = Reg(DT.float)
        l = Reg(DT.float)
        x <<= a[0, 0:64]
        y <<= b[0, 0:64]
        z.fill(0.)
        if kind == 'constant':
            _dd_add_constant(h, l, x, z, 1., 1.4901161193847656e-8)
        elif kind == 'bounded_product':
            _dd_mul(h, l, x, z, y, z, protect_overflow=False)
        else:
            _dd_mul(h, l, x, z, y, z)
        high[0, 0:64] <<= h
        low[0, 0:64] <<= l
        vf_barrier(VfPipe.STORE, VfPipe.LOAD)

    @kernel(mode='vec')
    def probe(a: GM[f32, (1, 64)], b: GM[f32, (1, 64)],
              high: GM[f32, (1, 64)], low: GM[f32, (1, 64)]):
        x = Tensor(DT.float, [1, 64], Position.UB)
        y = Tensor(DT.float, [1, 64], Position.UB)
        h = Tensor(DT.float, [1, 64], Position.UB)
        l = Tensor(DT.float, [1, 64], Position.UB)
        if GetVecIdx() == 0:
            with auto_sync():
                x <<= a
                y <<= b
                calculate(x, y, h, l)
                high <<= h
                low <<= l
        return high, low

    probe.name = 'kda_prep_backward_arithmetic_' + kind
    return probe


def inputs_and_reference(kind):
    """FP64 represents these binary32 products and selected sums exactly."""
    import torch

    if kind == 'constant':
        pairs = [(v, 0.) for v in (-.75, -.5, -.25, -2.**-25,
                                   2.**-25, .25, .5, .75)]
    else:
        pairs = [(1.+2.**-23, 1.-2.**-23), (-1.-2.**-23, 1.-2.**-23),
                 (1.234567, 3.456789), (2.**-40*(1.+2.**-23), 2.**-40*(1.-2.**-23)),
                 (2.**50*(1.+2.**-23), 2.**50*(1.-2.**-23)),
                 (-.75, .375), (2.**100, 2.**40), (-2.**100, 2.**40)]
        if kind == 'bounded_product':
            pairs = [((1.+2.**-23)/4, (1.-2.**-23)/4),
                     (-(1.+2.**-23)/4, (1.-2.**-23)/4),
                     (.1234567, .3456789),
                     (2.**-20*(1.+2.**-23), 2.**-20*(1.-2.**-23)),
                     (.125, .375), (-.75, .375), (.875, .625), (-.875, .625)]
    a, b = (torch.tensor([float(p[i]) for p in pairs] * 8, dtype=torch.float32).view(1, 64)
            for i in (0, 1))
    exact = a.double() + (1.+2.**-26) if kind == 'constant' else a.double()*b.double()
    high = exact.float()
    low = torch.where(high.isfinite(), (exact-high.double()).float(), torch.zeros_like(high))
    return a, b, high, low


def check_outputs(high, low, expected_high, expected_low):
    import torch

    assert torch.equal(high.view(torch.int32), expected_high.view(torch.int32)), 'DD high bits'
    assert torch.equal(low.view(torch.int32), expected_low.view(torch.int32)), 'DD residual bits'
