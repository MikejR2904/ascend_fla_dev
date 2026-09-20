"""Recheck the two complete native grids; never supplies reference tensors."""
import argparse,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--bd1',type=Path,required=True);p.add_argument('--bd2',type=Path,required=True);p.add_argument('--contract',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
contract=json.loads(a.contract.read_text());reports={1:json.loads(a.bd1.read_text()),2:json.loads(a.bd2.read_text())}
names={'dq','dk','dv','dg','dbeta'};stages=names|{'checkpoints','final_state','tape','dq_parts','dk_parts'}
matrices={};maxima={};audits={};omitted={};zero_budgets=0
for bd,report in reports.items():
 assert report['passed'] and report['block_dim']==bd
 expected={(c['id'],d) for c in contract['cases'] if c['block_dim']==bd for d in ('torch.bfloat16','torch.float32')}
 rows={(r['case']['id'],r['dtype']):r for r in report['cases']}
 assert len(rows)==len(report['cases']) and set(rows)==expected and len(rows)==138
 matrices[bd]={}
 for (identifier,dtype),row in rows.items():
  assert row['passed'] and row['inputs_unchanged'] and not row['forbidden_operators']
  assert set(row['public_hashes'])==names and set(row['stage_hashes'])==stages
  assert set(row['inputs'])=={'q','k','v','g','beta','do','dht'}
  if dtype=='torch.float32':assert row['fp32_original_byte_equal'] and row['public_hashes']==row['fp32_original_hashes']
  else:assert row['fp32_original_hashes'] is None
  mode=row['case']['parameters']['mode']
  expected_poison=[] if dtype=='torch.float32' or mode=='both' else ['dout' if mode=='dht' else 'dht']*2
  assert row['absent_cotangent_poison']==expected_poison
  omitted[(dtype,mode)]=omitted.get((dtype,mode),0)+1
  operators=set(row['host_operators'])
  assert operators<= {'aten.empty.memory_format','aten.empty_strided.default','aten.zeros.default','aten.zeros_like.default','aten.view.default','aten.unsqueeze.default','aten.alias.default'}
  if dtype=='torch.bfloat16':assert operators<={'aten.empty.memory_format','aten.empty_strided.default'}
  audits.setdefault(dtype,set()).update(operators)
  for category in ('public_A','public_B','composition','independent_leaf'):
   assert set(row[category])==(names if category.startswith('public_') else stages)
   for n,value in row[category].items():
    error=value['relative_l2'];assert value['finite'] and value['passed'] and isinstance(error,(int,float)) and math.isfinite(error)
    limit=min(.01,3*value['floor']) if dtype=='torch.bfloat16' and n in names else .0001
    assert value['budget']==limit and error<=limit
    if limit==0:zero_budgets+=1;assert error==0
    key=(dtype,category,n);maxima[key]=max(maxima.get(key,0.),error)
  matrices[bd][(identifier.rsplit('_bd',1)[0],dtype)]=row
assert set(matrices[1])==set(matrices[2])
for key,left in matrices[1].items():
 right=matrices[2][key]
 assert left['case']['parameters']==right['case']['parameters'] and left['case']['seed']==right['case']['seed']
 assert left['inputs']==right['inputs']
 assert left['stage_hashes']==right['stage_hashes'],(key,'stage bytes')
 assert left['public_hashes']==right['public_hashes'],(key,'public bytes')
report=dict(passed=True,records=sum(len(x['cases']) for x in reports.values()),cross_block_pairs=len(matrices[1]),stage_arrays_per_pair=10,public_arrays_per_pair=5,zero_budget_comparisons=zero_budgets,
 maxima={'/'.join(key):value for key,value in sorted(maxima.items())},host_operators={k:sorted(v) for k,v in audits.items()},cotangent_records={'/'.join(k):v for k,v in omitted.items()},original_fp32_byte_equal_records=138)
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
