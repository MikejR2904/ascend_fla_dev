"""Shared native acceptance setup: identity, complete vendor registration, provenance."""
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import ascriptor
import torch
import torch_npu
from ascend_fla.ops.kda import chunk,chunk_bwd,fused_recurrent,prepare,autograd
from ascend_fla.runtime.compile import compile_kernel
import native_environment
import unit
import verify_native


class Context:
    def __init__(self,output,chunk_bd,decode_bd,driver):
        assert os.environ.get('BF07_EXTERNAL_DEVICE_LOCK')=='1'
        assert platform.python_version()=='3.12.14'
        assert torch.__version__=='2.12.0+cu130' and torch_npu.__version__=='2.12.0'
        assert ascriptor.__version__=='0.1.0'
        root=Path(os.environ['BF07_NATIVE_ROOT'])
        assert Path(ascriptor.__file__).is_relative_to(root/'library')
        for name,sha in json.loads((root/'accepted-source-manifest.json').read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
        torch.set_num_threads(1)
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.environment=native_environment.collect()
        self.environment['drivers_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(driver).parent.glob('*.py')}
        self.environment['prep_sha256']={str(p.relative_to(unit.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in unit.ROOT.rglob('*.py')}
        self.environment['wrappers_sha256']={Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (chunk,chunk_bwd,autograd,fused_recurrent)}
        self.bd=chunk_bd;self.dbd=decode_bd
        runtime=chunk._prep_runtime()
        plan=[('prep_chunk',e,chunk_bd) for e in runtime.kernels('chunk').values()]
        plan += [('prep_decode',e,decode_bd) for e in runtime.kernels('decode').values()]
        plan += [('layout',e,chunk_bd) for e in chunk._layout_runtime().kernels().values()]
        plan += [('forward',e,chunk_bd) for e in chunk.kda_fwd_kernels().values()]
        plan += [('backward',e,chunk_bd) for e in chunk_bwd.kda_bwd_kernels().values()]
        plan += [('decode',fused_recurrent._native_kernel(d),decode_bd) for d in (torch.bfloat16,torch.float32)]
        assert len(plan)==50 and sum(f=='backward' for f,_,_ in plan)==9
        builds=[]
        for family,entry,dim in plan:
            print('COMPILE_START',family,entry.name,dim,flush=True);start=time.monotonic()
            op=compile_kernel(entry,device='a5',block_dim=dim,backend='cce')
            builds.append(dict(family=family,entry=entry.name,block_dim=dim,signature=op.signature,seconds=time.monotonic()-start))
            self.write('compile',dict(complete=len(builds)==50,entries=builds))
        prepare(block_dim=chunk_bd,backward=True,decode=True,decode_block_dim=decode_bd)
        torch.npu.set_device(0)
        self.check=verify_native.load('_bf07_checks',unit.ROOT.parent/'kda_layout/native_support.py')
        import native_audit
        native_audit.REPO=unit.ROOT.parents[3];self.check.Audit=native_audit.Audit
        self.old,self.old_auto,self.old_decode=verify_native.predecessors(chunk,autograd,fused_recurrent)
        self.real,self.reference=self.check.reference_modules()
        fla_path=Path(os.environ['FLA_KDA_NAIVE'])
        assert hashlib.sha256(fla_path.read_bytes()).hexdigest()=='60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016'
        self.fla=verify_native.load('_bf07_fla',fla_path)
        self.precision=unit.module('precision')
        self.budgets=json.loads((unit.ROOT/'budgets.json').read_text())
        self.write('environment',{})

    def write(self,name,value):
        def clean(x):
            if isinstance(x,float) and not math.isfinite(x):return None
            if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
            if isinstance(x,(tuple,list)):return [clean(v) for v in x]
            return x
        p=self.output/(name+'.json');p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(clean(dict(environment=self.environment,**value)),indent=2,allow_nan=False)+'\n');tmp.replace(p)
