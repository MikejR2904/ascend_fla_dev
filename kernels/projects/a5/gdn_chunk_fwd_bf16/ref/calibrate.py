"""Pre-kernel A/B calibration and fixed per-case BF16 rounding floors."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from . import oracle
from .reference import cases,make_inputs,reference,metric,floor,budget,validate_reference


def run(output):
    torch.set_num_threads(1)
    report=dict(stage='cpu_pre_kernel_calibration',python=sys.version,torch=torch.__version__,
                fla_pin=oracle.PIN,oracle_sha256=oracle.SHA256,
                rule='per case/output/oracle BF16 budget=min(1e-2,3*F); F=relL2(BF16(FP32_reference),FP32_reference); zero F requires exact output; final_state remains FP32',
                source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},cases=[])
    for case in cases():
        for dtype in ('bfloat16','float32'):
            started=time.time();inp=make_inputs(dict(case,dtype=dtype))
            a=oracle.reference(inp);b=reference(inp)
            validate_reference(inp,a);validate_reference(inp,b)
            numbers={n:metric(b[n],a[n]) for n in a}
            assert all(x['finite'] and x['relative_l2']<=1e-4 for x in numbers.values()),(case,numbers)
            row=dict(case=case,dtype=dtype,A_B=numbers,
                     floors={label:{n:floor(x) for n,x in values.items()} for label,values in [('A',a),('B',b)]},
                     budgets={label:{n:budget(x,inp['q'].dtype) for n,x in values.items()} for label,values in [('A',a),('B',b)]},
                     seconds=time.time()-started)
            report['cases'].append(row);output.write_text(json.dumps(report,indent=2)+'\n')
            print('CALIBRATION',case['id'],dtype,json.dumps(numbers),flush=True)
    report['passed']=True;output.write_text(json.dumps(report,indent=2)+'\n')
    print('CALIBRATION_PASS',len(report['cases']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
