"""Audit native grids and compare every retained stage/public output byte hash."""
import argparse
import hashlib
import json
import math
from pathlib import Path

GRADIENTS = ('dq', 'dk', 'dv', 'dg', 'dbeta')
STAGES = set(GRADIENTS) | {'checkpoints', 'final_state', 'tape', 'dq_parts', 'dk_parts'}


def load_grid(path, block_dim):
    data = json.loads(path.read_text())
    assert data['passed'] and data['block_dim'] == block_dim
    expected = json.loads((Path(__file__).parent / 'contract.json').read_text())['cases']
    expected = {row['id']: row for row in expected if row['block_dim'] == block_dim}
    indexed = {}
    for row in data['cases']:
        case = expected[row['case']]
        assert row['seed'] == case['seed'] and row['parameters'] == case['parameters']
        assert row['inputs_unchanged']
        assert set(row['stage_hashes']) == STAGES
        assert set(row['public_hashes']) == set(GRADIENTS)
        assert row['dtype'] in ('torch.float32', 'torch.bfloat16')
        for group in ('composition', 'leaf', 'fp32_A', 'fp32_B', 'public_A', 'public_B'):
            for name, metric in row[group].items():
                limit = 5e-3 if group.startswith('public') and row['dtype'] == 'torch.bfloat16' and name in ('dq', 'dk', 'dv') else 1e-4
                assert metric['finite'] and math.isfinite(metric['relative_l2'])
                assert metric['relative_l2'] <= limit, (row['case'], group, name, metric)
        key = (row['case'].removesuffix(f'_bd{block_dim}'), row['dtype'])
        assert key not in indexed
        indexed[key] = row
    assert len(indexed) == 2 * len(expected)
    return indexed


def compare(bd1, bd2):
    grids = (load_grid(bd1, 1), load_grid(bd2, 2))
    assert grids[0].keys() == grids[1].keys()
    pairs = []
    for key in sorted(grids[0]):
        first, second = (grid[key] for grid in grids)
        assert first['parameters'] == second['parameters'] and first['seed'] == second['seed']
        for field in ('stage_hashes', 'public_hashes'):
            assert first[field] == second[field], (key, field)
        pairs.append(dict(case=key[0], dtype=key[1], stage_hashes=first['stage_hashes'],
                          public_hashes=first['public_hashes'], byte_identical=True))
    maxima = {}
    for group in ('fp32_A', 'fp32_B', 'public_A', 'public_B', 'composition', 'leaf'):
        maxima[group] = {
            name: max(row[group][name]['relative_l2'] for grid in grids for row in grid.values())
            for name in next(iter(grids[0].values()))[group]
        }
    return dict(passed=True, method='SHA256 of contiguous returned tensor bytes, including signed zero',
                inputs=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (bd1, bd2)],
                case_dtype_pairs=len(pairs), stage_arrays_per_pair=10, public_arrays_per_pair=5,
                total_array_pairs=len(pairs)*15, relative_l2_maxima=maxima, pairs=pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bd1', type=Path, required=True)
    parser.add_argument('--bd2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.bd1, args.bd2)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"PASS: {result['case_dtype_pairs']} case/dtype pairs, {result['total_array_pairs']} arrays byte-identical")


if __name__ == '__main__':
    main()
