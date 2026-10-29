"""Minimal real CLIP prefix, plain vs all-FP16 hybrid route; no production changes."""
import argparse
import collections
import gc
import json
import os
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,TensorProto
import onnxruntime as ort
from rknn.api import RKNN
import yaml
from record_ir_provenance import digest
from prefix_specs import SPECS,directory_name

parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['extract','plain','step1','hybrid'],required=True)
parser.add_argument('--segment',choices=list(SPECS),default='stem')
parser.add_argument('--cpu-attention',action='store_true',help='diagnostic-only exSDPAttention CPU route')
parser.add_argument('--optimization-level',type=int,choices=[0,3],default=3)
args=parser.parse_args()
root=Path(__file__).resolve().parent
directory=directory_name(args.segment,args.cpu_attention,args.optimization_level)
output=root/'artifacts'/directory
output.mkdir(exist_ok=True)
os.chdir(output)
source=root/'artifacts/clip_visual_reference_fp32.onnx'
if args.segment in ['qkv','q','k','v','core','projection']:
    source=root/'artifacts/prefix_attn0_control_20261010/prefix.onnx'
endpoint,shape=SPECS[args.segment]
names=[r['image'] for r in json.loads((root/'artifacts/diagnosis_20261010/reference_comparison.json').read_text())['rows']]
inputs=[root/'artifacts/diagnosis_20261010/inputs'/(name+'.npy') for name in names]

if args.phase=='extract':
    if (output/'prefix.onnx').exists(): raise SystemExit('prefix already exists; do not overwrite')
    model=onnx.load(source)
    producer={name:i for i,node in enumerate(model.graph.node) for name in node.output}
    needed={endpoint}
    chosen=set()
    pending=[endpoint]
    while pending:
        name=pending.pop()
        if name not in producer: continue
        index=producer[name]
        if index in chosen: continue
        chosen.add(index)
        for value in model.graph.node[index].input:
            if value not in needed:
                needed.add(value); pending.append(value)
    nodes=[n for i,n in enumerate(model.graph.node) if i in chosen]
    initializers=[v for v in model.graph.initializer if v.name in needed]
    graph=helper.make_graph(nodes,'clip_prefix',[helper.make_tensor_value_info('pixel_values',TensorProto.FLOAT,[1,3,224,224])],
        [helper.make_tensor_value_info(endpoint,TensorProto.FLOAT,shape)],initializers)
    prefix=helper.make_model(graph,opset_imports=list(model.opset_import))
    prefix.ir_version=model.ir_version
    onnx.checker.check_model(prefix)
    onnx.save(prefix,output/'prefix.onnx')
    metadata={'source_sha256':digest(source),'endpoint':endpoint,'output_shape':shape,
        'nodes':len(nodes),'operators':dict(collections.Counter(n.op_type for n in nodes)),
        'scope':'diagnostic subgraph, NOT smaller deployment model','segment':args.segment,
        'cpu_attention_control':args.cpu_attention,
        'optimization_level':args.optimization_level,
        'prefix_sha256':digest(output/'prefix.onnx')}
    (output/'provenance.json').write_text(json.dumps(metadata,indent=2))
    del model,prefix,graph,nodes,initializers
    gc.collect()
    session=ort.InferenceSession(str(output/'prefix.onnx'),providers=['CPUExecutionProvider'])
    values=[session.run(None,{'pixel_values':np.load(p)})[0] for p in inputs]
    np.save(output/'onnx_reference.npy',np.asarray(values))
    print(json.dumps(metadata),flush=True)
    raise SystemExit(0)

model=RKNN(verbose=False)
def check(code,operation):
    if code!=0: raise RuntimeError(operation+': '+str(code))
    print('DONE '+operation,flush=True)
try:
    if args.phase in ['plain','step1']:
        check(model.config(target_platform='rk3588',quantized_dtype='w8a8',quantized_method='channel',
            quantized_algorithm='normal',mean_values=[[0,0,0]],std_values=[[1,1,1]],optimization_level=args.optimization_level,
            auto_hybrid_cos_thresh=.995,op_target={'exSDPAttention':'cpu'} if args.cpu_attention else None),'config')
        check(model.load_onnx(model=str(output/'prefix.onnx')),'load')
        if args.phase=='plain':
            check(model.build(do_quantization=False),'build')
            check(model.export_rknn(str(output/'plain_fp16.rknn')),'export')
        else:
            dataset=output/'calibration_six.txt'
            # This diagnoses a full-float export route; not a production quantization calibration.
            dataset.write_text('\n'.join(map(str,inputs))+'\n')
            check(model.hybrid_quantization_step1(dataset=str(dataset),proposal=False),'step1')
    else:
        config_path=output/'prefix.quantization.cfg'
        cfg=yaml.safe_load(config_path.read_text())
        ir=onnx.load(output/'prefix.model')
        actual_outputs={name for node in ir.graph.node for name in node.output}
        selected=sorted(actual_outputs & set(cfg['quantize_parameters']))
        cfg['custom_quantize_layers']={name:'float16' for name in selected}
        policy=output/'all_fp16.cfg'
        policy.write_text(yaml.safe_dump(cfg,sort_keys=False))
        (output/'policy_summary.json').write_text(json.dumps({'protected_count':len(selected),'names':selected},indent=2))
        # Follow SDK step2 contract: restore embedded step1 config, do not reconfigure.
        check(model.hybrid_quantization_step2(model_input=str(output/'prefix.model'),
              data_input=str(output/'prefix.data'),model_quantization_cfg=str(policy)),'step2')
        check(model.export_rknn(str(output/'hybrid_all_fp16.rknn')),'export')
    if args.phase in ['plain','hybrid']:
        check(model.init_runtime(),'simulator init')
        values=[]
        for p in inputs:
            result=model.inference(inputs=[np.load(p)],data_format=['nchw'])
            if result is None or len(result)!=1 or not np.isfinite(result[0]).all():
                raise RuntimeError('invalid simulator output')
            values.append(result[0])
        np.save(output/(args.phase+'_simulator.npy'),np.asarray(values))
finally:
    model.release()
