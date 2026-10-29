"""Test FP16 protection of fused SDPA's named internal quantization operands."""
import json
import os
from pathlib import Path
import numpy as np
import onnx
from rknn.api import RKNN
import yaml

root=Path(__file__).resolve().parent
source=root/'artifacts/prefix_attn0_control_20261010'
output=root/'artifacts/prefix_attn0_aux_control_20261010'
output.mkdir(exist_ok=False)
os.chdir(output)
cfg=yaml.safe_load((source/'all_fp16.cfg').read_text())
ir=onnx.load(source/'prefix.model')
aux=[]
for node in ir.graph.node:
    if node.op_type=='exSDPAttention':
        for attribute in node.attribute:
            if attribute.name in ['q_matmul','q_softmax']:
                name=attribute.s.decode('utf-8')
                if name not in cfg['quantize_parameters']:
                    raise RuntimeError('missing auxiliary cfg entry '+name)
                aux.append(name)
if len(aux)!=2: raise RuntimeError('expected two first-block auxiliary names')
for name in aux: cfg['custom_quantize_layers'][name]='float16'
policy=output/'policy.cfg'
policy.write_text(yaml.safe_dump(cfg,sort_keys=False))
status={'phase':'step2','added_auxiliary_names':aux,'all_protected_count':len(cfg['custom_quantize_layers']),
        'note':'Presence of INT8 cfg defaults is not proof of actual compiled arithmetic dtype'}
def update(**values):
    status.update(values); (output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
def check(code,op):
    if code!=0: raise RuntimeError(op+': '+str(code))
model=RKNN(verbose=False)
try:
    update()
    check(model.hybrid_quantization_step2(model_input=str(source/'prefix.model'),data_input=str(source/'prefix.data'),
          model_quantization_cfg=str(policy)),'step2')
    check(model.export_rknn(str(output/'hybrid_all_fp16.rknn')),'export')
    check(model.init_runtime(),'simulator init')
    names=[name+'_020.png' for name in ['cardboard','glass','metal','paper','plastic','trash']]
    values=[]
    for name in names:
        value=model.inference(inputs=[np.load(root/'artifacts/diagnosis_20261010/inputs'/(name+'.npy'))],data_format=['nchw'])
        if value is None or not np.isfinite(value[0]).all(): raise RuntimeError('invalid simulator output')
        values.append(value[0])
    np.save(output/'hybrid_simulator.npy',np.asarray(values))
    update(phase='completed')
except Exception as error:
    update(phase='failed',error=repr(error));raise
finally:
    model.release()
