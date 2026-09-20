"""Recompute medians and sandwich ratios from every retained same-card sample."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
d=json.loads(a.input.read_text());assert d['passed'] and (d['rounds'],d['warmup'],d['repeat'])==(3,10,50)
rows=[]
for case in d['cases']:
 assert case['inputs_unchanged'] and len(case['sandwiches'])==3
 for checkpoint in ('checks_before','checks_after'):
  for path,values in case[checkpoint]['independent_B'].items():
   assert set(values)=={'dq','dk','dv','dg','dbeta'}
   assert all(m['finite'] and m['passed'] and m['relative_l2']<=m['budget'] for m in values.values())
 rounds=[]
 for sandwich in case['sandwiches']:
  segments=sandwich['segments'];assert [s['name'] for s in segments]==['baseline','candidate','baseline']
  medians=[]
  for segment in segments:
   samples=segment['samples_ms'];assert len(samples)==50 and min(samples)>0
   value=statistics.median(samples);assert value==segment['median_ms'];medians.append(value)
  baseline=(medians[0]+medians[2])/2
  rounds.append(dict(round=sandwich['round'],baseline_before_ms=medians[0],candidate_ms=medians[1],baseline_after_ms=medians[2],baseline_midpoint_ms=baseline,baseline_over_candidate=baseline/medians[1]))
 rows.append(dict(case=case['case'],rounds=rounds,median_baseline_ms=statistics.median(r['baseline_midpoint_ms'] for r in rounds),median_candidate_ms=statistics.median(r['candidate_ms'] for r in rounds),median_baseline_over_candidate=statistics.median(r['baseline_over_candidate'] for r in rounds)))
assert {r['case']['parameters']['T'] for r in rows}=={1024,4096}
out=dict(passed=True,stage=d['stage'],block_dim=d['block_dim'],baseline=d['baseline'],candidate=d['candidate'],synchronization=d['synchronization'],total_measured_invocations=900,total_warmup_invocations=180,cases=rows)
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
