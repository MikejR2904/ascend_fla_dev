"""Recompute BF-09 timing and closeout claims from restored original JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def verify(package, restored):
    package, restored = Path(package), Path(restored)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = json.loads((package / 'manifest.json').read_text())
    for name, expected in manifest.items():
        path = package / name
        assert path.resolve().is_relative_to(package.resolve())
        assert digest(path) == expected, name
    report = json.loads((package / 'summary.json').read_text())
    reconstructed = json.loads((package / 'reconstruction.json').read_text())
    for name, expected in reconstructed['original_sha256'].items():
        path = restored / name
        assert path.resolve().is_relative_to(restored.resolve())
        assert digest(path) == expected, name
    rows_by_stage = {}
    for stage in ('baseline-r2-perf-bd4', 'final-perf-bd4'):
        folder = restored / 'receipts' / stage
        raw = json.loads((folder / 'raw-timings.json').read_text())
        measured = json.loads((folder / 'measurements.json').read_text())
        assert raw['complete'] and len(raw['rows']) == 36
        assert measured['complete'] and len(measured['rows']) == 4
        rows = []
        for row in measured['rows']:
            samples = [s for s in raw['rows'] if (s['tokens'], s['baseline']) ==
                       (row['tokens'], row['baseline'])]
            assert len(samples) == 9
            for number in (1, 2, 3):
                assert [s['phase'] for s in samples if s['round'] == number] == [
                    'baseline_before', 'candidate', 'baseline_after']
            a = [s['wall_ms'] for s in samples if s['phase'] != 'candidate']
            b = [s['wall_ms'] for s in samples if s['phase'] == 'candidate']
            baseline, candidate = statistics.median(a), statistics.median(b)
            assert row['passed'] and all(row['inputs_unchanged'].values())
            assert baseline == row['baseline_median_ms']
            assert candidate == row['candidate_median_ms']
            assert candidate / baseline == row['candidate_over_baseline']
            rows.append(dict(tokens=row['tokens'], baseline=row['baseline'],
                baseline_median_ms=baseline, candidate_median_ms=candidate,
                ratio=candidate / baseline, baseline_samples_ms=a, candidate_samples_ms=b))
        assert rows == report['performance'][stage]
        rows_by_stage[stage] = rows
    original = next(r for r in rows_by_stage['baseline-r2-perf-bd4']
                    if (r['tokens'], r['baseline']) == (4096, 'baseline'))
    final = next(r for r in rows_by_stage['final-perf-bd4']
                 if (r['tokens'], r['baseline']) == (4096, 'baseline'))
    reduction = 1 - final['candidate_median_ms'] / original['candidate_median_ms']
    assert reduction == report['t4096_latency_reduction_vs_same_device_merged_bf08']
    assert report['advisory_target_met'] == (final['ratio'] <= 1.25)
    for profile in report['profiles'].values():
        for route in profile['routes'].values():
            groups = route['custom_kernels']
            assert sum(g['launches'] for g in groups.values()) == route['custom_launches']
            for group in groups.values():
                assert sum(group['per_launch_device_ms']) == group['device_ms']
            assert sum(g['device_ms'] for g in groups.values()) == route['profiled_custom_device_sum_ms']
    arithmetic = json.loads((restored / 'receipts/final-arithmetic-bd4/summary.json').read_text())
    assert arithmetic['passed'] and arithmetic['all53vendors_before_firstcustom']
    assert len(arithmetic['rows']) == 3
    assert all(r['actual'] == r['expected'] for r in arithmetic['rows'])
    assert not report['global_training_host_compliance']
    for case, row in report['actual_benchmark_audit'].items():
        path = package / 'raw-benchmark' / (case + '.json')
        raw = json.loads(path.read_text())
        assert digest(path) == row['raw_sha256']
        assert raw['summary']['forbidden_calls'] == row['forbidden_calls']
        assert raw['summary']['verdict'] == row['verdict']
        assert not any(r['category'].startswith('forbidden') and
                       r['site_file'].startswith('kernels/projects/a5/kda_prep/')
                       for r in raw['rows'])
    return dict(passed=True, raw_timing_samples=72, native_arithmetic_kinds=3,
                t4096_latency_reduction=reduction,
                advisory_target_met=report['advisory_target_met'],
                global_training_host_compliance=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--restored', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.restored), indent=2))
