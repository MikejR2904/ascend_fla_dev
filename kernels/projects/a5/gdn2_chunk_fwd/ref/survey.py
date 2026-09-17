"""Five-size CPU gate-range/precision survey; synthetic and checkpoint modes.

No recorded tensor datasets are used. Checkpoint mode generates legal token IDs
and tokenizes natural text locally, captures the read-only model's core boundary
in memory and compares each chunk composition with its FP32 recurrent oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import types

import torch

from chunk import gdn2_chunk_reference

SIZES = (4, 8, 16, 32, 64)
CHECKPOINT_SHA256 = '4ac729c627febc431bf4b2011a9cb7ad9c30cc1d017ff74aeca15f76efb2df6d'


def metric(got, expected):
    finite = bool(torch.isfinite(got).all())
    if not finite:
        return dict(finite=False, relative_l2=None, max_abs_diff=None, passed=False)
    delta = got.float() - expected.float()
    ratio = float(delta.norm() / expected.float().norm().clamp_min(1e-30))
    return dict(finite=True, relative_l2=ratio, max_abs_diff=float(delta.abs().max()), passed=ratio<=1e-4)


def survey(tensors, oracle, *, label, layer=None, scale=128**-0.5,
           normalize=True, eps=1e-6):
    q,k,v,g,b,w,state=tensors
    expected=oracle(*tensors, scale=scale, use_qk_l2norm=normalize, qk_norm_eps=eps)
    rows=[]
    for size in SIZES:
        actual=gdn2_chunk_reference(*tensors,scale=scale,use_qk_l2norm=normalize,
                                    qk_norm_eps=eps,chunk_size=size)
        time=g.shape[1]
        pad=(-time)%size
        gp=torch.cat((g,torch.zeros(g.shape[0],pad,*g.shape[2:])),dim=1)
        blocks=gp.reshape(g.shape[0],-1,size,*g.shape[2:])
        prefix=blocks.cumsum(2)
        row=dict(sample=label,layer=layer,shape=list(q.shape),chunk_size=size,
                 max_token_decay=float((-g).max()),
                 max_chunk_decay=float((-blocks.sum(2)).max()),
                 max_prefix_span=float((prefix.amax(2)-prefix.amin(2)).max()),
                 output=metric(actual[0],expected[0]),state=metric(actual[1],expected[1]))
        rows.append(row)
        print(json.dumps(row,allow_nan=False),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic',action='store_true')
    source.add_argument('--checkpoint',type=Path)
    parser.add_argument('--tokenizer',type=Path)
    parser.add_argument('--text',default='The capital of France is Paris. Linear attention processes a sequence by updating a recurrent state. ')
    parser.add_argument('--length',type=int,default=4096)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.length<=4096:
        parser.error('length must be 1..4096')
    # Only checkpoint mode needs the repository model. Synthetic mode can run
    # with this standalone unit alone, using its independent recurrence.
    torch.set_num_threads(1)
    record=dict(source='synthetic' if args.synthetic else 'real-95B-checkpoint',
                device='cpu',torch=str(torch.__version__),seed=20260914,
                chunk_sizes=list(SIZES),rows=[],passed=False,
                historical_replay_complete=False)
    try:
        with torch.inference_mode():
            if args.synthetic:
                from reference import reference as unit_reference
                def oracle(q,k,v,g,b,w,state,**unused):
                    got=unit_reference(dict(q=q,k=k,v=v,g=g,erase_gate=b,w=w,initial_state=state))
                    return got['o'],got['final_state']
                generator=torch.Generator().manual_seed(20260914)
                shape=(1,args.length,16,128)
                q,k,v=(torch.randn(shape,generator=generator) for _ in range(3))
                b=torch.rand(shape,generator=generator)*2
                w=torch.rand(shape,generator=generator)
                state=torch.randn(1,16,128,128,generator=generator)*0.1
                # Calibrate each complete 64-token block to the historical
                # magnitude; this is explicitly NOT a real-checkpoint replay.
                g=-torch.rand(shape,generator=generator)
                for start in range(0,args.length,64):
                    window=g[:,start:start+64]
                    window.mul_(1461.214 / (-window.sum(1,keepdim=True)))
                record['rows']=survey((q,k,v,g,b,w,state),oracle,label='synthetic-calibrated-1461.214')
            else:
                if args.tokenizer is None:
                    parser.error('checkpoint mode requires a local tokenizer')
                # Find the source checkout only for real-model collection.
                root=Path(__file__).resolve().parents[5]
                sys.path.insert(0,str(root))
                from ascend_fla.models import GDN2ForCausalLM
                from ascend_fla.reference.gdn2 import gdn2_recurrent_reference
                from transformers import AutoTokenizer
                digest=hashlib.sha256()
                with args.checkpoint.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024*1024),b''):
                        digest.update(block)
                record['checkpoint_sha256']=digest.hexdigest()
                if record['checkpoint_sha256']!=CHECKPOINT_SHA256:
                    raise ValueError('checkpoint digest differs from the recorded 95B object')
                model=GDN2ForCausalLM.from_checkpoint(args.checkpoint,device='cpu',dtype=torch.bfloat16,
                                                     core_backend='torch',projection_layout='canonical')
                tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
                record['tokenizer_vocab_size']=tokenizer.vocab_size
                if tokenizer.vocab_size!=32000:
                    raise ValueError('expected the recorded 32000-token tokenizer')
                record['natural_text']=args.text
                prompt=tokenizer(args.text,add_special_tokens=True)['input_ids']
                if not prompt:
                    raise ValueError('natural text produced no tokens')
                natural=torch.tensor((prompt*math.ceil(args.length/len(prompt)))[:args.length]).unsqueeze(0)
                rng=torch.Generator().manual_seed(20260914)
                random=torch.randint(0,model.config.vocab_size,(1,args.length),generator=rng)
                sample={'label':None}
                originals=[]
                for index, block in enumerate(model.transformer['h']):
                    mixer=block.attn
                    original=mixer._run_core
                    originals.append((mixer,original))
                    def capture(self,q,k,v,g,b,w,state,_original=original,_index=index):
                        tensors=tuple(x.float().contiguous() for x in (q,k,v,g,b,w))
                        if state is None:
                            state=torch.zeros(q.shape[0],q.shape[2],q.shape[3],v.shape[3])
                        record['rows'].extend(survey((*tensors,state.float()),gdn2_recurrent_reference,
                            label=sample['label'],layer=_index,scale=self.attention_scale,
                            normalize=self.use_qk_l2norm,eps=self.qk_norm_eps))
                        return _original(q,k,v,g,b,w,state)
                    mixer._run_core=types.MethodType(capture,mixer)
                try:
                    for label,tokens in (('legal-random-token-ids',random),('natural-text-repeated',natural)):
                        sample['label']=label
                        model(tokens)
                finally:
                    for mixer,original in originals:
                        mixer._run_core=original
                # This records that both actual model runs completed; comparison
                # with the historical maximum must still be reported explicitly.
                record['historical_replay_complete']=args.length==4096
                record['observed_random_max_64']=max(row['max_chunk_decay'] for row in record['rows']
                    if row['sample']=='legal-random-token-ids' and row['chunk_size']==64)
                record['historical_max_64']=1461.214
                record['historical_delta']=record['observed_random_max_64']-1461.214
            record['passed']=all(row['output']['passed'] and row['state']['passed'] for row in record['rows'])
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    if not record['passed']:
        raise RuntimeError('gate-range survey exceeded the fixed FP32 budget')


if __name__=='__main__':
    main()
