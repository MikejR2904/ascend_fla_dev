"""Recompute pre-implementation budget arithmetic from text receipts."""
from pathlib import Path
import hashlib
import json


def verify():
    root = Path(__file__).resolve().parent
    raw = root / 'evidence/calibration/pre-kernel.json'
    data = json.loads(raw.read_text())
    budgets = json.loads((root / 'budgets.json').read_text())
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == budgets['calibration_sha256']
    assert hashlib.sha256((root/'ref/calibrate.py').read_bytes()).hexdigest() == data['script_sha256']
    for name, item in data['predecessor']['files'].items():
        assert hashlib.sha256((root/'baseline'/name).read_bytes()).hexdigest() == item['sha256']
    assert len(data['cases']) == 100
    observed = {}
    for row in data['cases']:
        m = row['metrics']
        if m['finite_pairs'] != m['elements']:
            assert m['relative_l2'] is None, 'nonfinite/empty subset cannot count as numeric pass'
        if not row['numerical_budget_population']:
            continue
        assert m['finite_pairs'] == m['elements']
        assert m['reference_norm'] > 0 or (m['residual_norm'] == 0 and m['relative_l2'] == 0)
        if m['reference_norm']:
            assert m['relative_l2'] == m['residual_norm'] / m['reference_norm']
        key = row['op'] + ':' + row['types']
        floor = observed.setdefault(key, [0., 0., 0, 0])
        floor[0] = max(floor[0], m['relative_l2'])
        floor[1] = max(floor[1], m['max_relative_nonzero'])
        floor[2] += 1
        floor[3] = max(floor[3], m['rounded_reference_max_ulp'])
    assert set(observed) == set(budgets['limits']) == set(data['groups'])
    for key, (l2, point, cases, ulp) in observed.items():
        group = data['groups'][key]
        assert (l2, point, cases, ulp) == (group['floor_relative_l2'], group['floor_max_relative'], group['cases'], group['max_ulp'])
        limit = budgets['limits'][key]
        assert limit['relative_l2'] == (min(.01, 3*l2) if key.startswith('norm:') else 3*l2)
        assert limit['max_relative_nonzero'] == 3*point
        assert limit['relative_l2'] < .01 and limit['max_relative_nonzero'] < .02
    assert len(data['negative_controls']) == 66
    for row in data['negative_controls']:
        assert row['rejected'] and row['relative_l2'] > 3 * row['calibration_floor']
    manifest = root / 'evidence/calibration/files.sha256.json'
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    print('PRE-KERNEL CALIBRATION VERIFIED: 100 records, 14 budget groups, 66 rejected controls')


if __name__ == '__main__':
    verify()
