"""Independent CPU references and detailed backward comparison receipts."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def numeric_ulps(actual, reference):
    rounded = reference.to(actual.dtype)
    integer = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    sign = 32768 if actual.dtype == torch.bfloat16 else 2147483648
    mask = sign - 1
    a = actual.contiguous().view(integer).long()
    b = rounded.contiguous().view(integer).long()
    # Signed magnitude maps both zero encodings to zero numeric distance.
    aa = torch.where(a < 0, -(a & mask), a & mask)
    bb = torch.where(b < 0, -(b & mask), b & mask)
    distance = (aa - bb).abs()
    finite = torch.isfinite(actual) & torch.isfinite(rounded)
    values, counts = torch.unique(distance[finite], return_counts=True)
    distribution = [dict(ulp=int(u), count=int(n)) for u, n in zip(values, counts)]
    return distance, finite, distribution


def raw_reference(raw, function, prepare_inputs, flags):
    """Checkpointed CPU FP32 KDA recurrence plus the explicit preparation graph."""
    from torch.utils.checkpoint import checkpoint

    names = ('q', 'k', 'v', 'g', 'beta', 'A_log', 'dt_bias', 'h0')
    leaves = {n: raw[n].float().detach().clone().requires_grad_(True) for n in names}
    prepared = prepare_inputs(*(leaves[n] for n in ('q', 'k', 'g', 'beta')),
                              A_log=leaves['A_log'], dt_bias=leaves['dt_bias'], **flags)
    q, k, g, beta = prepared
    state, outputs = leaves['h0'], []
    for start in range(0, q.shape[1], 64):
        section = slice(start, start+64)

        def segment(q, k, v, g, beta, state):
            return function(q.float(), k.float(), v.float(), g.float(), beta.float(),
                            initial_state=state, output_final_state=True)

        o, state = checkpoint(segment, q[:, section], k[:, section], leaves['v'][:, section],
                              g[:, section], beta[:, section], state, use_reentrant=False)
        outputs.append(o)
    output = torch.cat(outputs, 1)
    gradients = torch.autograd.grad((output, state), [leaves[n] for n in names],
                                    (raw['do'].float(), raw['dht'].float()), allow_unused=True)
    return dict(o=output.detach(), final_state=state.detach(),
                **{'d'+n: t.detach() if t is not None else None for n, t in zip(names, gradients)})


def gradient_record(key, actual, fp64, old, precision, budget, detail_root,
                    source=None, sensitivity=None, condition=None, environment=None,
                    endpoint_zero_mask=None, endpoint_authority=None):
    """Both-reference metrics plus every BF16 discrepancy beyond one ULP."""
    actual, fp64, old = (t.detach().cpu().contiguous() for t in (actual, fp64, old))
    high = precision.metrics(actual, fp64)
    previous = precision.metrics(actual, old)
    u64, finite64, histogram64 = numeric_ulps(actual, fp64)
    uold, finite_old, histogram_old = numeric_ulps(actual, old)
    ordinary = torch.ones_like(actual, dtype=torch.bool)
    endpoint = None
    ordinary_high = high
    ordinary_old = previous
    if endpoint_zero_mask is not None:
        mask = endpoint_zero_mask.detach().cpu()
        assert mask.dtype == torch.bool and mask.shape == actual.shape
        assert endpoint_authority, 'Endpoint classification needs an owning decision'
        ordinary = ~mask
        endpoint = dict(authority=endpoint_authority, elements=int(mask.sum()),
            candidate_zero=bool((actual[mask] == 0).all()),
            old_cpu_zero=bool((old[mask] == 0).all()),
            candidate_old_cpu_numeric_equal=bool(torch.equal(actual[mask], old[mask])),
            fp64_discrepancies_retained_in_full_metrics=True)
        ordinary_high = precision.metrics(actual[ordinary], fp64[ordinary]) if ordinary.any() else None
        ordinary_old = precision.metrics(actual[ordinary], old[ordinary]) if ordinary.any() else None
    m = ordinary_high
    passed = (m is None or (m['finite_pairs'] == int(ordinary.sum())
              and m['relative_l2'] is not None
              and m['relative_l2'] <= budget['relative_l2_limit']
              and m['max_relative_nonzero'] <= budget['elementwise_relative_limit']
              and m['zero_reference_nonzero_actual'] == 0))
    if budget['ulp_limit_each_reference'] is not None:
        passed = passed and bool(finite64[ordinary].all() and finite_old[ordinary].all()) and not bool(((u64[ordinary] > 1) | (uold[ordinary] > 1)).any())
    ordinary_passed = bool(passed)
    if endpoint is not None:
        passed = passed and endpoint['candidate_zero'] and endpoint['old_cpu_zero'] and endpoint['candidate_old_cpu_numeric_equal']
    details = []
    if actual.dtype == torch.bfloat16:
        locations = ((finite64 & (u64 > 1)) | (finite_old & (uold > 1))).nonzero().tolist()
        rounded = fp64.bfloat16()
        for index in locations:
            ix = tuple(index)
            row = dict(index=index, candidate_bits=int(actual.view(torch.int16)[ix]) & 65535,
                       old_bits=int(old.view(torch.int16)[ix]) & 65535,
                       rounded_fp64_bits=int(rounded.view(torch.int16)[ix]) & 65535,
                       fp64_value=float(fp64[ix]), ulp_to_fp64=int(u64[ix]), ulp_to_old=int(uold[ix]))
            if condition is not None:
                value = float(condition[ix])
                row['condition_number'] = value if torch.isfinite(torch.tensor(value)) else None
                row['zero_derivative_condition_undefined'] = not torch.isfinite(torch.tensor(value)).item()
            elif source is not None and sensitivity is not None:
                at = ix[:-1] if source.ndim == 4 else ix
                row['input_row'] = source[at].detach().double().reshape(-1).tolist()
                row['sensitivity_row'] = sensitivity[at].detach().double().reshape(-1).tolist()
            else:
                raise AssertionError('BF16 >1ULP detail lacks sensitivity or condition')
            details.append(row)
    detail_root.mkdir(parents=True, exist_ok=True)
    pages = []
    for start in range(0, len(details), 64):
        path = detail_root / f'{start//64:05d}.json'
        path.write_text(json.dumps(dict(environment=environment, key=key, locations=details[start:start+64]), indent=2, allow_nan=False)+'\n')
        pages.append(path.name)
    return dict(key=key, to_fp64=high, to_old_cpu=previous, budget=budget,
                ordinary_to_fp64=ordinary_high, ordinary_to_old_cpu=ordinary_old,
                ordinary_elements=int(ordinary.sum()), ordinary_criteria_passed=ordinary_passed,
                endpoint_zero=endpoint,
                ulp_to_fp64_distribution=histogram64, ulp_to_old_distribution=histogram_old,
                bf16_over_one_locations=len(details), detail_pages=pages, passed=bool(passed))
