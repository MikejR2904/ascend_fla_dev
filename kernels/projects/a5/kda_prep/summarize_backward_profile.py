"""Aggregate a private NPU profiler capture without publishing machine metadata."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def summarize(root, measured, output):
    root, measured = Path(root), Path(measured)
    metadata=json.loads((root/'profile-summary.json').read_text())
    assert metadata['complete'] and len(metadata['rows'])==2
    builds=json.loads((root/'compile.json').read_text())
    assert builds['complete'] and len(builds['entries'])==50
    assert sum(r['family']=='backward' for r in builds['entries'])==9
    # CANN uses the title-cased vendor name emitted from the registered entry.
    by_vendor={''.join(p.title() for p in r['entry'].split('_')):r for r in builds['entries']}
    clean=json.loads((measured/'measurements.json').read_text())
    clean=next(r for r in clean['rows'] if r['tokens']==4096 and r['baseline']=='baseline')
    actual=next(r for r in metadata['rows'] if r['route']=='candidate')['output_sha256']
    assert actual==clean['rounds'][0]['samples'][1]['output_sha256']
    routes={}
    for route in ('baseline','candidate'):
        paths=list((root/'profiler'/route).rglob('kernel_details.csv'));assert len(paths)==1
        path=paths[0];rows=list(csv.DictReader(path.open()))
        groups={};builtin=collections.Counter()
        for row in rows:
            name=row['Name'];duration=float(row['Duration(us)'])/1000
            if name not in by_vendor:
                builtin[row['Type']]+=duration;continue
            entry=by_vendor[name];group=groups.setdefault(entry['entry'],dict(
                family=entry['family'],launches=0,device_ms=0.,per_launch_device_ms=[],
                separate_event_window_ms=0.,host_dispatch_ms=0.))
            group['launches']+=1;group['device_ms']+=duration;group['per_launch_device_ms'].append(duration)
            if 'PrepBackwardGateBf16F32F32' in name:
                group['profiled_vector_time_us']=float(row['aiv_vec_time(us)'])
                group['profiled_vector_ratio']=float(row['aiv_vec_ratio'])
                group['profiled_scalar_ratio']=float(row['aiv_scalar_ratio'])
                group['profiled_mte2_ratio']=float(row['aiv_mte2_ratio'])
                group['profiled_mte3_ratio']=float(row['aiv_mte3_ratio'])
        attribution=clean['attribution'][route]
        by_signature={r['signature']:r['entry'] for r in builds['entries']}
        for row in attribution['launches']:
            group=groups[by_signature[row['signature']]]
            group['separate_event_window_ms']+=row['device_event_ms']
            group['host_dispatch_ms']+=row['host_dispatch_ms']
        assert sum(g['launches'] for g in groups.values())==len(attribution['launches'])
        routes[route]=dict(custom_kernels=groups,custom_launches=sum(g['launches'] for g in groups.values()),
            total_npu_kernel_rows=len(rows),profiled_custom_device_sum_ms=sum(g['device_ms'] for g in groups.values()),
            profiled_builtin_by_type_ms=dict(builtin),profiled_all_kernel_duration_sum_ms=sum(float(r['Duration(us)'])/1000 for r in rows),
            original_kernel_csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            separate_instrumented_wall_ms=attribution['wall_ms'],separate_whole_stream_event_ms=attribution['whole_stream_event_ms'],
            separate_custom_dispatch_sum_ms=attribution['custom_host_dispatch_sum_ms'])
    report=dict(environment=metadata['environment'],tokens=4096,block_dim=4,
        scope='Full forward+backward training including raw preparation; separate Level1 hardware profiler after complete training warmup.',
        profile_candidate_matches_clean_run_output_hashes=True,
        interpretation='Profiler counters perturb durations. CSV device durations exclude queue wait but are not an additive decomposition of the separately measured clean median. The separate event windows may include host idle gaps; host dispatch overlaps device execution.',
        source_summary_sha256=hashlib.sha256((root/'profile-summary.json').read_bytes()).hexdigest(),
        routes=routes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return {k:dict(launches=v['custom_launches'],npu_rows=v['total_npu_kernel_rows'],sum_ms=v['profiled_all_kernel_duration_sum_ms']) for k,v in routes.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--measured',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(summarize(a.root,a.measured,a.output)))


if __name__=='__main__':main()
