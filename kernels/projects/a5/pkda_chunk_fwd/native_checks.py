"""Additional native API boundaries, sharing only generated inputs with goldens."""
import torch

def extra_cases():
 specs=[('b2_t65_h32',2,65,32,{}),('b2_t130_h3',2,130,3,{}),('t129_h14',1,129,14,{}),
        ('raw_norm_1e-2',1,65,3,{'row_norm':1e-2}),('raw_norm_1e-4',1,65,3,{'row_norm':1e-4}),
        ('zero_qk',1,65,3,{'row_norm':0.}),('no_main_decay',1,192,3,{'gate_scale':0.}),
        ('span155_multichunk',2,130,3,{'gate_span':155.}),('zero_A_multichunk',2,130,3,{'zero_atk':True,'A_scale':0.})]
 cases=[dict(id=n,seed=9100+i,block_dim=1,parameters=dict(B=b,T=t,H=h,N=(t+63)//64,**kw)) for i,(n,b,t,h,kw) in enumerate(specs)]
 cases.extend(dict(id=f'adversarial155_seed{seed}',seed=seed,block_dim=1,parameters=dict(B=1,T=192,H=8,N=3,adversarial_span=155.)) for seed in (71,72,73))
 return cases

def public_boundaries(api,ref,oracle,compare,contract,bd):
 case=dict(id='api_states',seed=9351,parameters=dict(B=2,T=130,H=3,N=3))
 data=ref.make_inputs(case);rows=[]
 def to_npu(inp):return {n:(t.npu() if isinstance(t,torch.Tensor) else t) for n,t in inp.items()}
 def call(inp,**kw):
  result=api.chunk_precond_kda(**inp,block_dim=bd,output_final_state=True,**kw);torch.npu.synchronize()
  return dict(zip(ref.OUTPUTS,(t.cpu() for t in result)))
 for use_s in (False,True):
  for use_a in (False,True):
   gold=dict(data);args=dict(data)
   if not use_s:gold['initial_state']=torch.zeros_like(data['initial_state']);args.pop('initial_state')
   if not use_a:gold['initial_A_state']=torch.zeros_like(data['initial_A_state']);args.pop('initial_A_state')
   args.pop('log_atk_scale');gold['log_atk_scale']=torch.full((3,),-.2)
   got=call(to_npu(args));expected=ref.reference(gold);naive=oracle.oracle(gold)
   compare(got,expected,contract);compare(got,naive,contract)
   rows.append(dict(kind='optional_states',S=use_s,A=use_a,vs_fla_naive=oracle.metrics(got,naive),passed=True))
 full=call(to_npu(data));expected=oracle.oracle(data)
 for split in (1,63,64,65,129):
  left={n:t[:,:split].contiguous() if n in ('q','k','v','g','g_atk','beta_atk','beta') else t for n,t in data.items()}
  right={n:t[:,split:].contiguous() if n in ('q','k','v','g','g_atk','beta_atk','beta') else t for n,t in data.items()}
  first=call(to_npu(left));right.update(initial_state=first['final_state'],initial_A_state=first['final_A_state'])
  last=call(to_npu(right));got=dict(last,o=torch.cat((first['o'],last['o']),dim=1))
  compare(got,full,contract);compare(got,expected,contract)
  rows.append(dict(kind='state_carry',split=split,vs_full=oracle.metrics(got,full),vs_fla_naive=oracle.metrics(got,expected),passed=True))
 no_state=api.chunk_precond_kda(**to_npu(data),output_final_state=False,block_dim=bd);torch.npu.synchronize()
 assert no_state[1:] == (None,None)
 assert torch.equal(no_state[0].cpu(),full['o'])
 rows.append(dict(kind='output_final_state_false',passed=True))
 return rows

def prepare_case(data,case):
 if 'adversarial_span' in case['parameters']:
  # Full decay jump then weak tail; small input margin keeps the exact input
  # inside the unchanged 155 gate on both CPU and NPU reduction trees.
  gate=data['g'];gate.fill_(-1e-5);gate[:,::64].fill_(-154.9992)
 return data


def tail_cases():
 """Exercise every possible runtime VF count and dependent i+1 bound."""
 return [dict(id=f'tail_count_{n}',seed=9600+n,block_dim=4,
              parameters=dict(B=1,T=64+n,H=3,N=2)) for n in range(1,65)]
