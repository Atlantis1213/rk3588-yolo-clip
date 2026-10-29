"""Causal diagnostic: change ONLY the saved fused-SDPA scale in an isolated IR copy."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import onnx
from rknn.api import RKNN
from record_ir_provenance import digest

parser=argparse.ArgumentParser()
parser.add_argument('--scale',type=float,required=True)
parser.add_argument('--label',choices=['scale0','scale1','scale0015625'],required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
source=root/'artifacts/prefix_core_control_20261010'
output=root/'artifacts'/('sdpa_ir_'+args.label+'_20261010')
output.mkdir(exist_ok=False)
os.chdir(output)
model=onnx.load(source/'prefix.model')
sdpa=[n for n in model.graph.node if n.op_type=='exSDPAttention']
if len(sdpa)!=1: raise RuntimeError('expected exactly one SDPA node')
scale=next(a for a in sdpa[0].attribute if a.name=='scale')
original=float(scale.f);scale.f=args.scale
modified=output/'scale.model'
onnx.save(model,modified)
status={'phase':'step2','original_scale':original,'modified_scale':args.scale,
    'original_ir_sha':digest(source/'prefix.model'),'modified_ir_sha':digest(modified),
    'scope':'deliberately altered diagnostic, NOT equivalent deployment candidate'}
def update(**values):
    status.update(values);(output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
def check(code,op):
    if code!=0: raise RuntimeError(op+': '+str(code))
rknn=RKNN(verbose=False)
try:
    update()
    check(rknn.hybrid_quantization_step2(model_input=str(modified),data_input=str(source/'prefix.data'),
        model_quantization_cfg=str(source/'all_fp16.cfg')),'step2')
    check(rknn.export_rknn(str(output/'control.rknn')),'export')
    check(rknn.init_runtime(),'simulator init')
    values=[]
    for name in ['cardboard','glass','metal','paper','plastic','trash']:
        result=rknn.inference(inputs=[np.load(root/'artifacts/diagnosis_20261010/inputs'/(name+'_020.png.npy'))],data_format=['nchw'])
        if result is None or not np.isfinite(result[0]).all(): raise RuntimeError('invalid simulator output')
        values.append(result[0])
    np.save(output/'simulator.npy',np.asarray(values))
    update(phase='completed')
except Exception as error:
    update(phase='failed',error=repr(error));raise
finally:
    rknn.release()
