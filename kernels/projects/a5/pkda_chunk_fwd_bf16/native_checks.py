"""Additional BF16 native API checks; reference transformations are test-only."""
import hashlib
import json
import types
from pathlib import Path
import torch


def boundaries(api,ref,checks,bd,call,to_device,digest,Audit,allowed):
    rows=[]
    case=dict(id='states',seed=9351,parameters=dict(B=2,T=130,H=3))
    data=ref.make_inputs(case)
    for use_s in (False,True):
        for use_a in (False,True):
            gold=dict(data);args=dict(data)
            if not use_s:gold['initial_state']=torch.zeros_like(data['initial_state']);args.pop('initial_state')
            if not use_a:gold['initial_A_state']=torch.zeros_like(data['initial_A_state']);args.pop('initial_A_state')
            args.pop('log_atk_scale');gold['log_atk_scale']=torch.full((3,),-.2)
            got,ops=call(to_device(args));result=checks.compare(got,checks.references(gold))
            assert result['passed'],result
            rows.append(dict(kind='optional_states',S=use_s,A=use_a,host_operations=ops,**result))
    refs=checks.references(data)
    for split in (1,63,64,65,129):
        token=('q','k','v','g','g_atk','beta_atk','beta')
        left={n:(x[:,:split].contiguous() if n in token else x) for n,x in data.items()}
        right={n:(x[:,split:].contiguous() if n in token else x) for n,x in data.items()}
        first,_=call(to_device(left));right.update(initial_state=first['final_state'],initial_A_state=first['final_A_state'])
        last,_=call(to_device(right));got=dict(last,o=torch.cat((first['o'],last['o']),1))
        result=checks.compare(got,refs);assert result['passed'],result
        rows.append(dict(kind='continuation',split=split,**result))
    dev=to_device(data);whole,_=call(dev)
    no=api.chunk_precond_kda(**dev,output_final_state=False,block_dim=bd);torch.npu.synchronize()
    assert no[1:] == (None,None) and digest(no[0].cpu())==digest(whole['o'])
    rows.append(dict(kind='no_final_states',passed=True))
    tiny=ref.make_inputs(dict(id='boundary',seed=222,parameters=dict(B=1,T=64,H=1)))
    for label,change,expected in (
        ('q_nan',lambda d:d['q'].fill_(float('nan')),'finite'),
        ('v_inf',lambda d:d['v'].fill_(float('inf')),'finite'),
        ('g_positive',lambda d:d['g'].fill_(.01),'g must'),
        ('g_atk_positive',lambda d:d['g_atk'].fill_(.01),'g_atk'),
        ('beta_negative',lambda d:d['beta'].fill_(-.01),'beta'),
        ('beta_atk_over',lambda d:d['beta_atk'].fill_(1.01),'beta_atk'),
        ('A_negative',lambda d:d['initial_A_state'].fill_(-.01),'initial_A_state'),
        ('state_nan',lambda d:d['initial_state'].fill_(float('nan')),'finite'),
        ('center_inf',lambda d:d['log_atk_scale'].fill_(float('inf')),'finite'),
        ('span_over155',lambda d:d['g'].fill_(-155.1/64),'span'),
        ('key_over1',lambda d:d['k'].fill_(1.01),'key row'),
    ):
        bad={n:(x.clone() if isinstance(x,torch.Tensor) else x) for n,x in tiny.items()};change(bad)
        try:call(to_device(bad))
        except ValueError as error:
            assert expected in str(error),(label,str(error))
            rows.append(dict(kind='reject',case=label,error=str(error),passed=True))
        else:raise AssertionError(label+' accepted')
    # Exactly representable BF16 unit-norm key and the next BF16 value above it.
    key={n:x.clone() for n,x in tiny.items()};key['k'].zero_();key['k'][...,0]=1.
    got,_=call(to_device(key));result=checks.compare(got,checks.references(key));assert result['passed'],result
    rows.append(dict(kind='key_norm_exact1',**result))
    key['k'][...,0]=1.0078125
    try:call(to_device(key))
    except ValueError as error:
        assert 'key row' in str(error);rows.append(dict(kind='key_norm_next_bf16',passed=True,error=str(error)))
    else:raise AssertionError('next BF16 key norm accepted')
    # Execute the recorded before-wrapper and current wrapper with the same
    # unmodified FP32 kernels/inputs; compare bytes, including optional defaults.
    before_path=Path(__file__).parent/'evidence/fp32-entry-before.py'
    before=types.ModuleType('ascend_fla.ops._bf04_fp32_before')
    before.__file__=api.__file__;before.__package__='ascend_fla.ops'
    exec(compile(before_path.read_text(),str(before_path),'exec'),before.__dict__)
    before._module=api._module;before._compiled=api._compiled
    fp_ref=api._module('ref.reference')
    cases=json.loads((Path(api.__file__).resolve().parents[2]/'kernels/projects/a5/pkda_chunk_fwd/contract.json').read_text())['cases']
    for case in cases:
        inp=to_device(fp_ref.make_inputs(case))
        audit_before=Audit()
        with audit_before:old=before.chunk_precond_kda(**inp,output_final_state=True,block_dim=bd)
        audit_after=Audit()
        with audit_after:new=api.chunk_precond_kda(**inp,output_final_state=True,block_dim=bd)
        torch.npu.synchronize()
        assert not set(audit_after.operations)-allowed,audit_after.operations
        same=[digest(a.cpu())==digest(b.cpu()) for a,b in zip(old,new)]
        assert all(same),(case['id'],same)
        rows.append(dict(kind='fp32_before_after',case=case['id'],passed=True,bytes_equal=same,before_host_operations=audit_before.operations,after_host_operations=audit_after.operations,before_source_sha256=hashlib.sha256(before_path.read_bytes()).hexdigest()))
    for use_s in (False,True):
        for use_a in (False,True):
            inp=fp_ref.make_inputs(case)
            if not use_s:inp.pop('initial_state')
            if not use_a:inp.pop('initial_A_state')
            inp.pop('log_atk_scale');inp=to_device(inp)
            old=before.chunk_precond_kda(**inp,output_final_state=True,block_dim=bd)
            audit_after=Audit()
            with audit_after:new=api.chunk_precond_kda(**inp,output_final_state=True,block_dim=bd)
            torch.npu.synchronize()
            assert not set(audit_after.operations)-allowed,audit_after.operations
            assert all(digest(a.cpu())==digest(b.cpu()) for a,b in zip(old,new))
            rows.append(dict(kind='fp32_defaults_before_after',S=use_s,A=use_a,passed=True,host_operations=audit_after.operations))
    return rows


def fp32_full(api,bd,Audit,allowed,to_device,digest):
    ref=api._module('ref.reference');oracle=api._module('ref.oracles')
    case=dict(id='fp32_full_t4096',seed=8196,parameters=dict(B=1,T=4096,H=8))
    data=ref.make_inputs(case);gold=oracle.oracle(data);dev=to_device(data)
    before_path=Path(__file__).parent/'evidence/fp32-entry-before.py'
    before=types.ModuleType('ascend_fla.ops._bf04_fp32_before')
    before.__file__=api.__file__;before.__package__='ascend_fla.ops'
    exec(compile(before_path.read_text(),str(before_path),'exec'),before.__dict__)
    before._module=api._module;before._compiled=api._compiled
    old_audit=Audit()
    with old_audit:old=before.chunk_precond_kda(**dev,output_final_state=True,block_dim=bd)
    new_audit=Audit()
    with new_audit:new=api.chunk_precond_kda(**dev,output_final_state=True,block_dim=bd)
    torch.npu.synchronize()
    old={n:x.cpu() for n,x in zip(ref.OUTPUTS,old)};new={n:x.cpu() for n,x in zip(ref.OUTPUTS,new)}
    errors=oracle.metrics(new,gold);same={n:digest(old[n])==digest(new[n]) for n in old}
    unchanged={n:digest(dev[n].cpu())==digest(x) for n,x in data.items()}
    result=dict(case=case,block_dim=bd,errors=errors,bytes_equal=same,input_unchanged=unchanged,
                before_host_operations=old_audit.operations,after_host_operations=new_audit.operations,
                before_source_sha256=hashlib.sha256(before_path.read_bytes()).hexdigest(),
                passed=all(same.values()) and all(unchanged.values()) and not set(new_audit.operations)-allowed and all(m['relative_l2']<=1e-4 for m in errors.values()))
    return result
