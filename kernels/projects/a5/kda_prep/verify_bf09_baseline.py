"""Recompute the frozen, pre-optimization BF-09 native baseline (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, digest in manifest.items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve()), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    summary = json.loads((root / 'summary.json').read_text())
    source = summary['environment']['production_source_sha256']
    repo = Path(__file__).resolve().parents[4]
    for name, digest in source.items():
        data = subprocess.check_output([
            'git', '-C', str(repo), 'show', summary['base_commit'] + ':' + name])
        assert hashlib.sha256(data).hexdigest() == digest, name
    for stage in ('full', 'perf'):
        result = json.loads((root / stage / 'summary.json').read_text())
        assert result['complete'] and result['passed'], stage
        compiled = json.loads((root / stage / 'compile.json').read_text())
        assert compiled['complete'] and compiled['first_custom_launch_not_started']
        assert len(compiled['entries']) == 50
        assert sum(e['family'] == 'backward' for e in compiled['entries']) == 9
    for path in root.glob('*/*.json'):
        value = json.loads(path.read_text())
        assert value['environment']['production_source_sha256'] == source, path.name
    full = json.loads((root / 'full/full.json').read_text())
    assert full['passed'] and len(full['cpu_references']) == 2
    measured = json.loads((root / 'perf/measurements.json').read_text())
    raw = json.loads((root / 'perf/raw-timings.json').read_text())
    assert raw['complete'] and measured['complete']
    assert len(raw['rows']) == 36 and len(measured['rows']) == 4
    rows = []
    for row in measured['rows']:
        samples = [s for s in raw['rows'] if
                   (s['tokens'], s['baseline']) == (row['tokens'], row['baseline'])]
        assert len(samples) == 9 and {s['round'] for s in samples} == {1, 2, 3}
        for round_number in (1, 2, 3):
            assert [s['phase'] for s in samples if s['round'] == round_number] == [
                'baseline_before', 'candidate', 'baseline_after']
        baseline = statistics.median(s['wall_ms'] for s in samples if s['phase'] != 'candidate')
        candidate = statistics.median(s['wall_ms'] for s in samples if s['phase'] == 'candidate')
        assert row['passed'] and all(row['inputs_unchanged'].values())
        assert baseline == row['baseline_median_ms']
        assert candidate == row['candidate_median_ms']
        assert candidate / baseline == row['candidate_over_baseline']
        rows.append(dict(tokens=row['tokens'], baseline=row['baseline'],
                         baseline_median_ms=baseline, merged_bf08_median_ms=candidate,
                         ratio=candidate / baseline))
    assert rows == summary['rows']
    return dict(verified=True, source_commit=summary['base_commit'], raw_samples=36,
                rows=rows, scope='Pre-optimization bd4 baseline only; no candidate qualification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path,
                        default=Path(__file__).parent / 'evidence/backward/bf09-baseline-v1')
    print(json.dumps(verify(parser.parse_args().evidence), indent=2))
