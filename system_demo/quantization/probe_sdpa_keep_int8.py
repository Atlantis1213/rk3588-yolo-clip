"""Change only the fused attention output precision policy in the pure SDPA reproducer."""
from pathlib import Path
import json
import os
import numpy as np
import onnx
from rknn.api import RKNN
import yaml

root=Path(__file__).resolve().parent
source=root/'artifacts/standalone_sdpa_20261010'
output=root/'artifacts/standalone_sdpa_keep_int8_20261010'
output.mkdir(exist_ok=False);os.chdir(output)
cfg=yaml.safe_load((source/'all_fp16.cfg').read_text())
ir=onnx.load(source/'sdpa.model')
removed=[]
for node in ir.graph.node:
    if node.op_type=='exSDPAttention':
        for name in node.output:
            cfg['custom_quantize_layers'].pop(name,None);removed.append(name)
if len(removed)!=1: raise RuntimeError('unexpected attention output count')
policy=output/'policy.cfg';policy.write_text(yaml.safe_dump(cfg,sort_keys=False))
model=RKNN(verbose=False)
def check(code,op):
    if code!=0: raise RuntimeError(op+': '+str(code))
try:
    check(model.hybrid_quantization_step2(model_input=str(source/'sdpa.model'),data_input=str(source/'sdpa.data'),
        model_quantization_cfg=str(policy)),'step2')
    check(model.export_rknn(str(output/'control.rknn')),'export')
    check(model.init_runtime(),'simulator init')
    values=[]
    for name in ['cardboard','glass','metal','paper','plastic','trash']:
        result=model.inference(inputs=[np.load(source/(name+'_'+key+'.npy')) for key in ['q','k','v']],data_format=['nchw']*3)
        if result is None or not np.isfinite(result[0]).all(): raise RuntimeError('invalid simulator output')
        values.append(result[0])
    np.save(output/'simulator.npy',np.asarray(values))
    (output/'status.json').write_text(json.dumps({'phase':'completed','removed_fp16_sdpa_outputs':removed},indent=2))
except Exception as error:
    (output/'status.json').write_text(json.dumps({'phase':'failed','error':repr(error)},indent=2));raise
finally:
    model.release()
