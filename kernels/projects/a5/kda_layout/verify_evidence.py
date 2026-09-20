"""Recompute the text evidence using only Python's standard library.

This checks recorded measurements and identities; it does not execute a kernel
or promote historical qualification to the current source.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics

from aggregate import verify


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(root, name):
    records = read(root / name)
    for relative, row in records.items():
        path = root / relative
        assert path.resolve().is_relative_to(root.resolve()), relative
        assert sha(path) == row['sha256'], relative
    return len(records)


def performance(root):
    report = read(root / 'perf-bd4/performance.json')
    assert report['passed'] and report['rounds'] == 3 and len(report['measurements']) == 8
    for measurement in report['measurements']:
        assert len(measurement['rounds']) == 3
        phases = collections.defaultdict(list)
        for row in measurement['rounds']:
            assert len(row['samples']) == 3
            for sample in row['samples']:
                assert math.isfinite(sample['milliseconds']) and sample['milliseconds'] > 0
                phases[sample['phase']].append(sample['milliseconds'])
        if 'baseline_before' in phases:
            assert set(phases) == {'baseline_before', 'candidate', 'baseline_after'}
            baseline = statistics.median(phases['baseline_before'] + phases['baseline_after'])
            candidate = statistics.median(phases['candidate'])
            assert baseline == measurement['baseline_median_ms'] and candidate == measurement['candidate_median_ms']
            assert candidate / baseline == measurement['candidate_over_baseline']
            assert measurement['extra_launches'] == measurement['launches']['candidate']['total'] - measurement['launches']['baseline']['total']
        else:
            assert set(phases) == {'torch_npu_before', 'candidate', 'torch_npu_after'}
    return len(report['measurements'])


def attribution(path):
    data = read(path)
    assert data['complete'] and data['block_dim'] == 4 and data['device'] == 'dev-B'
    assert {(r['tokens'], r['scope']) for r in data['rows']} == {
        (t, s) for t in (1024, 4096) for s in ('plain_forward', 'cached_forward', 'backward_existing_caches')}
    result = []
    for row in data['rows']:
        assert len(row['clean_rounds']) == 3
        phases = collections.defaultdict(list)
        for run in row['clean_rounds']:
            assert [s['phase'] for s in run] == ['old_before', 'candidate', 'old_after']
            for sample in run:
                assert sample['milliseconds'] > 0 and math.isfinite(sample['milliseconds'])
                phases[sample['phase']].append(sample['milliseconds'])
        assert not any(row['hot_three_call_counters'].values())
        layouts = [event for event in row['event_rows'] if event['operator'].startswith('KdaLayout')]
        assert len(layouts) == {'plain_forward': 6, 'cached_forward': 14, 'backward_existing_caches': 1}[row['scope']]
        for event in row['event_rows']:
            assert event['device_event_ms'] > 0 and event['host_dispatch_ms'] >= 0
            assert event['input_bytes'] >= 0 and event['output_bytes'] > 0
        result.append(dict(tokens=row['tokens'], scope=row['scope'],
            clean_candidate_median_ms=statistics.median(phases['candidate']),
            clean_baseline_median_ms=statistics.median(phases['old_before'] + phases['old_after']),
            layout_device_event_sum_ms=sum(e['device_event_ms'] for e in layouts),
            layout_host_dispatch_sum_ms=sum(e['host_dispatch_ms'] for e in layouts),
            hot_three_call_io_counts=row['hot_three_call_counters']))
    return result


def main():
    unit = Path(__file__).resolve().parent
    evidence = unit / 'evidence'
    native = evidence / 'native'
    all_files = read(evidence / 'all-files.sha256.json')
    actual = {str(p.relative_to(evidence)) for p in evidence.rglob('*') if p.is_file() and p.name != 'all-files.sha256.json'}
    assert set(all_files) == actual
    for name, digest in all_files.items():
        assert sha(evidence / name) == digest, name
    native_files = check_manifest(evidence, 'native-manifest.json')
    check_manifest(evidence, 'local-manifest.json')
    environment = dict(cann='9.1.0-beta.1', compiler_timestamp='20260509_173000235',
                       opp_directories=['ascend910_93', 'ascend910b', 'ascend950'],
                       python='3.12.14', torch='2.12.0+cu130', torch_npu='2.12.0', soc='a5')
    assert all(r['runtime_environment'] == environment for r in read(evidence / 'native-manifest.json').values())
    aggregate = verify(native)
    # Every fresh native process must carry the delivered production identity.
    for path in native.glob('*/environment.json'):
        env = read(path)
        for name in ('runtime.py', 'kernels/move.py'):
            assert sha(unit / name) == env['source_sha256'][name], str(path)
        for name, digest in env['wrapper_sha256'].items():
            assert sha(unit.parents[3] / 'ascend_fla/ops/kda' / name) == digest
        compile = read(path.parent / 'compile.json')
        assert compile['complete'] and len(compile['entries']) == 22
        assert sum(r['family'] == 'backward' for r in compile['entries']) == 9
    tails_common = None
    for bd in (1, 2, 3, 4):
        tails = read(native / f'tails-bd{bd}/tails.json')
        assert tails['complete'] and tails['passed'] and len(tails['cases']) == 6
        hashes = {}
        for row in tails['cases']:
            assert row['block_dim'] == bd and row['passed'] and row['canaries_intact'] and row['input_unchanged']
            value = row['comparison']['destination']
            assert value['passed'] and value['sha256'] == value['before_sha256']
            hashes[row['operator']] = value['sha256']
        if tails_common is None:
            tails_common = hashes
        else:
            assert hashes == tails_common
    seed_path = evidence / 'repro-bundle/seed-manifest.json'
    seed = read(seed_path)
    raw = read(evidence / 'repro-bundle/raw-tensor-manifest.json')
    assert seed['schema'] == 'fmt02.scan-seeded-inputs/1'
    assert sha(seed_path) == raw['seed_manifest']['sha256']
    assert sha(unit.parents[3] / seed['generator']['file']) == seed['generator']['sha256']
    assert len(seed['public_inputs']) == 8 and len(seed['scan_inputs']) == 8
    historic = seed['historical_bf16_dh0']
    assert len(historic) == 36 and all(r['dtype'] == 'bfloat16' for r in historic.values())
    assert all('hex' not in row for group in ('public_inputs', 'scan_inputs', 'historical_bf16_dh0') for row in seed[group].values())
    repeat_summary = {}
    for label, folder in [('dev-A', 'old-public-repeats-bd4'), ('dev-B', 'old-public-repeats-dev-B-bd4')]:
        data = read(native / folder / 'repeats.json')['rows']
        assert len(data) == 24 and all(r['input_unchanged'] for r in data)
        names = set(data[0]['sha256']) - {'h0'}
        assert len(names) == 7 and all(len({r['sha256'][n] for r in data}) == 1 for n in names)
        baseline = [r for r in data if r['path'] == 'before']
        assert len(baseline) == 12
        hashes = dict(collections.Counter(r['sha256']['h0'] for r in baseline))
        if label == 'dev-B':
            assert len(hashes) == 1 and len({r['sha256']['h0'] for r in data}) == 1
            assert all(r['h0_cpu_fp32']['budget'] == .05 and r['h0_cpu_fp32']['relative_l2'] <= .05 for r in data)
        repeat_summary[label] = dict(old_h0_hash_distribution=hashes, other_seven_exact=True)
        widen = read(native / f'sample-widen-{label}-bd4/widening.json')
        assert widen['complete'] and widen['passed'] and len(widen['samples']) == 36
        observed = {}
        for row in widen['samples']:
            # The original acquisition index calls these replay/N; the widening
            # runner labels the same captured tensor scan-replay/N.
            sample = row['sample']
            if sample.startswith('scan-replay/'):
                sample = sample.removeprefix('scan-')
            name = row['origin'] + '/' + sample
            assert name not in observed
            observed[name] = row['bf16_sha256']
            assert row['passed'] and row['cpu_f32_sha256'] == row['old_f32_sha256'] == row['new_f32_sha256']
        assert observed == {name: row['sha256'] for name, row in historic.items()}
        replay_dir = native / f'seed-replay-{label}-bd4'
        reproduction = read(replay_dir / 'input-reproduction.json')
        for group in ('public_inputs', 'scan_inputs'):
            assert set(reproduction[group]) == set(seed[group])
            assert all(row['actual'] == row['expected'] == seed[group][name]['sha256'] for name, row in reproduction[group].items())
        replay = read(replay_dir / 'replay.json')
        assert replay['passed'] and replay['input_unchanged'] and not replay['scan_dh0_fixed']
        assert replay['seed_manifest_sha256'] == sha(seed_path)
        assert len(replay['repeats']) == 12 and len(replay['same_tensor_widening']) == 12
        distributions = {name: dict(collections.Counter(row[name] for row in replay['repeats'])) for name in replay['repeats'][0]}
        assert distributions == replay['distributions']
        assert all(len(distributions[name]) == 1 for name in ('dAqk', 'dh', 'dv'))
        for index, row in enumerate(replay['same_tensor_widening']):
            assert row['sample'] == index and row['bf16_sha256'] == replay['repeats'][index]['dh0']
            assert row['input_unchanged'] and row['cpu_fp32_sha256'] == row['old_fp32_sha256'] == row['new_fp32_sha256']
        repeat_summary[label]['direct_scan_dh0_hash_distribution'] = distributions['dh0']
    model = read(evidence / 'model/summary.json')
    assert model['passed'] and len(model['cases']) == 30 and all(r['passed'] for r in model['cases'])
    assert model['kernel_sha256'] == sha(unit / 'kernels/move.py')
    history = evidence / 'history/v1'
    check_manifest(history, 'native-manifest.json')
    check_manifest(history, 'local-manifest.json')
    historical_aggregate = verify(history / 'native')
    assert sha(history / 'source/kernels/move.py') != sha(unit / 'kernels/move.py')
    old_attribution = attribution(history / 'native/perf-attribution-v1-bd4/attribution.json')
    new_attribution = attribution(native / 'perf-attribution-v2-bd4/attribution.json')
    print(json.dumps(dict(passed=True, evidence_files=len(all_files), native_files=native_files,
        aggregate=aggregate, tail_canaries=24, repeats=repeat_summary, conversion_checks=96,
        bounded_models=30, performance_records=performance(native), attribution=new_attribution,
        history=dict(aggregate=historical_aggregate, performance_records=performance(history / 'native'),
                     attribution=old_attribution, qualifies_current_source=False)), indent=2))


if __name__ == '__main__':
    main()
