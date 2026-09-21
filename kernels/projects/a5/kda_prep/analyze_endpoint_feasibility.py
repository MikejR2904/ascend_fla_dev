"""Necessary error-bound checks on saved native-run CPU FP32 references."""
from pathlib import Path
import argparse,hashlib,json,platform
import torch
parser=argparse.ArgumentParser(description='Check necessary feasibility conditions using retained actual CPU FP32 goldens; never changes acceptance.')
parser.add_argument('--tensor-root',type=Path,required=True)
parser.add_argument('--tensor-manifest',type=Path,required=True)
parser.add_argument('--case-table',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();restore=args.tensor_root;table=json.loads(args.case_table.read_text());expected=json.loads(args.tensor_manifest.read_text())['files'];rows=[];seen=set();loaded={};source_hashes={};torch.set_num_threads(1)
def tensor_hash(t):
 return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
for entry in table['failed_slices']:
 route,label,scope,c,h=(entry[k] for k in ('route','case','scope','chunk','head'));key=(route,label,scope,c,h)
 if key in seen:continue
 seen.add(key);run='nearzero-observe-v1-chunk-bd1' if route=='chunk' else 'nearzero-dpm51-v1-decode-bd1';path=restore/run/(label+'.private.pt')
 if (route,label) not in loaded:
  source_hashes[route,label]=hashlib.sha256(path.read_bytes()).hexdigest();assert source_hashes[route,label]==expected[str(path.relative_to(restore))]['sha256']
  loaded[route,label]=torch.load(path,map_location='cpu',weights_only=True)
 data=loaded[route,label];refs=data['reference'] if route=='chunk' else data['native_preparation_reference'];idx=(slice(None),slice(c*64,(c+1)*64),h) if scope=='head_chunk' else (...,)
 actual=data['actual']['o'][idx];targets={name:refs[name]['o'][idx] for name in ('independent','fla')};norms={name:float(t.double().norm()) for name,t in targets.items()};limits={};floors={};zero_special={}
 for name,e in targets.items():
  rounded=e.to(actual.dtype);norm=norms[name];floor=float((rounded.double()-e.double()).norm())/norm if norm else 0.;floors[name]=floor
  zero_special[name]=actual.dtype==torch.bfloat16 and bool((e.bfloat16()==0).all())
  limits[name]=.05 if route=='chunk' else min(.01,3*floor) if actual.dtype==torch.bfloat16 else 1e-5
 distance=float((targets['independent'].double()-targets['fla'].double()).norm());radius=sum(limits[n]*norms[n] for n in targets)
 applicable=not any(zero_special.values());disjoint=applicable and distance>radius
 impossible_rounding={name:applicable and floors[name]>limits[name] for name in targets}
 rows.append(dict(route=route,case=label,native_tensor_file=str(path.relative_to(restore)),native_tensor_archive_sha256=source_hashes[route,label],reference_sha256={n:tensor_hash(t) for n,t in targets.items()},scope=scope,chunk=c,head=h,actual_dtype=str(actual.dtype),golden_dtype='torch.float32',reference_norms=norms,relative_limits=limits,nearest_representable_relative_l2=floors,zero_representation_exception=zero_special,error_ball_test_applicable=applicable,reference_distance=distance,sum_allowed_error_radii=radius,distance_over_radius=distance/radius if radius else None,reference_error_balls_disjoint=disjoint,minimum_rounding_already_exceeds_limit=impossible_rounding,
  independent_to_fla_relative_l2=distance/norms['fla'] if norms['fla'] else None))
summary={}
for route in ('chunk','decode'):
 rr=[x for x in rows if x['route']==route];bad=[x for x in rr if x['reference_error_balls_disjoint'] or any(x['minimum_rounding_already_exceeds_limit'].values())]
 summary[route]=dict(failed_locations=len(rr),locations_with_disjoint_reference_error_balls=sum(x['reference_error_balls_disjoint'] for x in rr),locations_with_unrepresentable_tolerance=sum(any(x['minimum_rounding_already_exceeds_limit'].values()) for x in rr),cases_with_any_impossible_criterion=len(set(x['case'] for x in bad)),impossible_locations=len(bad),maximum_reference_distance_over_allowed_radii=max((x['distance_over_radius'] or 0 for x in rr),default=0))
report=dict(analysis='Necessary feasibility conditions only; does not change or pass any original criterion.',source_table_sha256=hashlib.sha256(args.case_table.read_bytes()).hexdigest(),analysis_environment=dict(python=platform.python_version(),torch=torch.__version__),analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),goldens='Retained actual runtime CPU FP32 tensors, never replaced by FP64; FP64 used solely for metric accumulation.',bounds='Triangle inequality: both L2 errors can fit only if norm(ref1-ref2)<=limit1*norm(ref1)+limit2*norm(ref2). Nearest output-dtype rounding minimizes elementwise absolute error and therefore L2; its error is a lower bound for all representable outputs.',summary=summary,locations=rows)
args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2));print('examples',json.dumps([x for x in rows if x['reference_error_balls_disjoint'] or any(x['minimum_rounding_already_exceeds_limit'].values())][:2],indent=2))
