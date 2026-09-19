"""CPU-only BF-04 operand-precision study; never hardware qualification."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, os, sys, time
import torch

ROOT=Path(__file__).resolve().parents[1]
UNIT=ROOT
sys.path.insert(0,str(UNIT))
from ref.reference import make_inputs
from ref.stages import reference_stages
from ref.oracles import oracle,NAIVE_SHA256

def rounded(x):return x.to(torch.bfloat16).float()
def metric(x,y):
 d=x.float()-y.float()
 return dict(relative_l2=float(d.norm()/y.float().norm().clamp_min(1e-30)),max_abs=float(d.abs().max()))

def stages(data,variant):
 q=data['q'];batch,length,heads,_=q.shape;n=(length+63)//64
 a=data['initial_A_state'].clone();center=data['log_atk_scale'].view(1,heads,1);kn=[]
 for t in range(length):
  a=data['g_atk'][:,t].exp().unsqueeze(-1)*a+data['beta_atk'][:,t].unsqueeze(-1)*data['k'][:,t].square()
  if variant=='atk_state_bf16':a=rounded(a)
  r=(a+1e-6).log()-center;m=(-math.log(1.5)*r/(1+r.abs())).exp();kn.append(data['k'][:,t]*m)
 def pack(x):
  pad=x.new_zeros(batch,n*64-length,heads,128)
  return torch.cat((x,pad),1).reshape(batch,n,64,heads,128).transpose(2,3).contiguous()
 qn=pack(q*float(data.get('scale',128**-.5)));kn=pack(torch.stack(kn,1));gc=pack(data['g']).cumsum(-2)
 bk=pack(data['beta'].unsqueeze(-1)*data['k']);wv=pack(data['beta'].unsqueeze(-1)*data['v'])
 if variant=='prefix_bf16':gc=rounded(gc)
 lower=q.new_zeros(batch,n,heads,64,64);score=torch.zeros_like(lower)
 def operands(stage,x,y):return (rounded(x),rounded(y)) if variant in (stage,'all_matmul_bf16') else (x,y)
 for i in range(64):
  decayed=kn[...,:i+1,:]*(gc[...,i:i+1,:]-gc[...,:i+1,:]).exp()
  lhs,rhs=operands('scores_bf16',qn[...,i:i+1,:],decayed);score[...,i,:i+1]=(lhs*rhs).sum(-1)
  if i:
   lhs,rhs=operands('scores_bf16',bk[...,i:i+1,:],decayed[...,:i,:]);lower[...,i,:i]=(lhs*rhs).sum(-1)
 rhs_w=bk*gc.exp();lhs=lower
 if variant in ('wy_bf16','all_matmul_bf16'):lhs=rounded(lhs);rhs_w=rounded(rhs_w);rhs_v=rounded(wv)
 else:rhs_v=wv
 system=lhs+torch.eye(64);u=torch.linalg.solve_triangular(system,rhs_v,upper=False,unitriangular=True);wy=torch.linalg.solve_triangular(system,rhs_w,upper=False,unitriangular=True)
 s=data['initial_state'].clone();states=[];deltas=[];outputs=[]
 for c in range(n):
  states.append(s)
  lhs,rhs=operands('scan_bf16',wy[:,c],s);delta=u[:,c]-lhs@rhs;deltas.append(delta)
  lhs,rhs=operands('output_bf16',qn[:,c]*gc[:,c].exp(),s);o=lhs@rhs
  lhs,rhs=operands('output_bf16',score[:,c],delta);outputs.append(o+lhs@rhs)
  tail=kn[:,c]*(gc[:,c,:,-1:,:]-gc[:,c]).exp()
  lhs,rhs=operands('scan_bf16',tail.transpose(-1,-2),delta)
  s=gc[:,c,:,-1,:].exp().unsqueeze(-1)*s+lhs@rhs
 o=torch.stack(outputs,1).transpose(2,3).reshape(batch,n*64,heads,128)[:,:length].contiguous()
 return dict(qn=qn,kn=kn,gc=gc,bk=bk,wv=wv,lower=lower,score=score,u=u,wy=wy,states=torch.stack(states,1),delta=torch.stack(deltas,1),o=rounded(o),final_state=s,final_A_state=a)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'precision-study.json');args=p.parse_args()
 torch.set_num_threads(1)
 cases=json.loads((UNIT/'contract.json').read_text())['cases']
 for span in (0.,1e-4,50.,105.,155.):
  cases.append(dict(id=f'gate_uniform_{span}',seed=704,parameters=dict(B=1,T=192,H=2,gate_span=span)))
 for seed in (71,72,73):cases.append(dict(id=f'gate_strong_weak_{seed}',seed=seed,parameters=dict(B=1,T=192,H=8),adversarial=True))
 variants=('fp32_internal','scores_bf16','wy_bf16','scan_bf16','output_bf16','all_matmul_bf16','atk_state_bf16','prefix_bf16')
 results=[];start=time.time()
 for case in cases:
  data=make_inputs(case)
  # Generation keeps rounded keys strictly within the existing norm domain;
  # this is not operator preprocessing or a change of the public contract.
  data={n:(x.float() if isinstance(x,torch.Tensor) else x) for n,x in data.items()}
  if case.get('adversarial'):
   gate=data['g'].reshape(1,3,64,8,128);gate.fill_(-1e-5);gate[:,:,0].fill_(-154.9992)
  norm=float(data['k'].norm(dim=-1).max());assert norm<=1+1e-5
  refa=oracle(data);refb=reference_stages(data)
  floors={n:dict(A=metric(rounded(refa[n]),refa[n])['relative_l2'],B=metric(rounded(refb[n]),refb[n])['relative_l2']) for n in ('o','final_state')}
  thresholds={n:min(.01,3*min(f.values())) for n,f in floors.items()};thresholds['final_A_state']=1e-4
  row=dict(case=case['id'],parameters=case['parameters'],key_norm_max=norm,floors=floors,thresholds=thresholds,oracle_B_vs_A={n:metric(refb[n],refa[n]) for n in refa},variants={})
  for variant in variants:
   got=stages(data,variant)
   errors={oracle_name:{n:metric(got[n],ref[n]) for n in refa} for oracle_name,ref in [('A',refa),('B',refb)]}
   passed=all(v['relative_l2']<=thresholds[n] for e in errors.values() for n,v in e.items())
   stage_errors={n:metric(got[n],refb[n]) for n in ('qn','kn','gc','bk','wv','lower','score','u','wy','states','delta')}
   row['variants'][variant]=dict(passed=passed,errors=errors,stage_errors=stage_errors)
  assert row['variants']['fp32_internal']['passed'],row
  results.append(row)
  print(json.dumps(dict(case=row['case'],floors=floors,passes={k:v['passed'] for k,v in row['variants'].items()},output_errors={k:v['errors']['A']['o']['relative_l2'] for k,v in row['variants'].items()})),flush=True)
  report=dict(stage='CPU precision study only; no hardware execution',torch=torch.__version__,fla_revision='e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2',fla_naive_sha256=NAIVE_SHA256,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),variants=variants,cases=results,elapsed_seconds=time.time()-start)
  args.output.write_text(json.dumps(report,indent=2)+'\n')
 print('COMPLETE',len(results),'cases',time.time()-start,flush=True)
if __name__=='__main__':main()
