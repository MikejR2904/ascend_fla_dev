"""Recompute grid claims from the bounded, public original-data review package."""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(root, budgets):
    root = Path(root)
    load = lambda name: json.loads((root/name).read_text())
    manifest = load('manifest.json')['sha256']
    for name, expected in manifest.items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert digest((root/path).read_bytes()) == expected, name
    report = load('summary.json')
    assert report['passed'] and report['cases_per_bd'] == 1620 and report['gradient_selections'] == 9
    cases = []
    for path in sorted((root/'output-hashes').glob('*.json')):
        page = json.loads(path.read_text())
        def decode(value):
            if isinstance(value, dict):
                if set(value) == {'$sha256'}:
                    return page['sha256_dictionary'][value['$sha256']]
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        cases.extend(decode(page['cases']))
    assert len(cases) == len({c['id'] for c in cases}) == 1620
    header = load('original-summary-header.json')
    fields = dict(header['original_metadata'], cases=cases)
    summary = {key: fields[key] for key in header['original_key_order']}
    original = (json.dumps(summary, indent=2, allow_nan=False)+'\n').encode()
    assert digest(original) == header['original_summary_sha256']
    assert {x['summary_sha256'] for x in report['original_summaries'].values()} == {digest(original)}
    assert digest(Path(budgets).read_bytes()) == report['frozen_budget_sha256']
    limits = json.loads(Path(budgets).read_text())['groups']
    maxima = {}; metric_rows = 0
    for path in sorted((root/'raw-metrics').glob('*.json')):
        page = json.loads(path.read_text())
        for values in page['rows']:
            row = dict(zip(page['columns'], values))
            high = dict(zip(page['metric_fields'], row['to_fp64']))
            budget = limits[row['key']]
            assert row['passed'] and high['finite_pairs'] == high['elements']
            assert high['zero_reference_nonzero_actual'] == 0
            values = dict(relative_l2=high['relative_l2'], max_relative=high['max_relative_nonzero'],
                relative_l2_ratio=high['relative_l2']/budget['relative_l2_limit'],
                max_relative_ratio=high['max_relative_nonzero']/budget['elementwise_relative_limit'])
            assert values['relative_l2_ratio'] <= 1 and values['max_relative_ratio'] <= 1
            for metric, value in values.items():
                maxima[(row['key'], metric)] = max(maxima.get((row['key'], metric), 0.), value)
            metric_rows += 1
    assert metric_rows == 7992
    assert set(k for k, _ in maxima) == set(report['groups'])
    for (key, metric), maximum in maxima.items():
        assert report['groups'][key][metric]['value'] == maximum, (key, metric)
    uses = collections.Counter()
    for case in load('audit-uses.json')['cases']:
        assert len(case['table_by_selection']) == 9
        uses.update(case['table_by_selection'].values())
    union = collections.Counter()
    for table in report['audit_tables']:
        assert digest(json.dumps(table['operations'], sort_keys=True, separators=(',', ':')).encode()) == table['sha256']
        assert set(table['uses_per_bd'].values()) == {uses[table['sha256']]}
        for row in table['operations']:
            key = tuple(row.get(k) for k in ('category', 'file', 'line', 'operator'))
            union[key] += row['count'] * uses[table['sha256']] * 4
    assert sum(uses.values()) == 1620 * 9
    assert union == {tuple(row[k] for k in ('category', 'file', 'line', 'operator')): row['count']
                     for row in report['audit_operation_union']}
    for path in (root/'selected-original').glob('*.json'):
        row = json.loads(path.read_text())
        raw = (json.dumps(row['receipt'], indent=2, allow_nan=False)+'\n').encode()
        assert len(raw) == row['original_bytes'] and digest(raw) == row['original_sha256']
        assert row['receipt']['passed']
    return dict(passed=True, reconstructed_summary_sha256=digest(original), metric_rows=metric_rows,
        groups=len(report['groups']), public_training_calls=sum(uses.values())*4,
        all_four_original_summary_hashes_equal=True, audit_tables=len(uses),
        audit_union_recomputed=True, selected_original_receipts=len(list((root/'selected-original').glob('*.json'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--budgets', type=Path, default=Path(__file__).with_name('backward_budgets.json'))
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.budgets)))


if __name__ == '__main__':
    main()
