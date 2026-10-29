"""Weightless pure SDPA reproducer: distinguish kernel/export defect from CLIP QKV packing."""
from pathlib import Path
import argparse
import json
import os
import numpy as np
import onnx
from onnx import helper,TensorProto
from rknn.api import RKNN
import yaml

parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['prepare','plain','step1','hybrid'],required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
output=root/'artifacts/standalone_sdpa_20261010'
output.mkdir(exist_ok=True)
os.chdir(output)
shape=[1,16,257,64]
names=['cardboard','glass','metal','paper','plastic','trash']
if args.phase=='prepare':
    if (output/'sdpa.onnx').exists(): raise RuntimeError('existing standalone source')
    inputs=[helper.make_tensor_value_info(n,TensorProto.FLOAT,shape) for n in ['Q','K','V']]
    nodes=[helper.make_node('Mul',['Q','sqrt_scale'],['sq']),
           helper.make_node('Transpose',['K'],['kt'],perm=[0,1,3,2]),
           helper.make_node('Mul',['kt','sqrt_scale'],['sk']),
           helper.make_node('MatMul',['sq','sk'],['score']),
           helper.make_node('Softmax',['score'],['probability'],axis=-1),
           helper.make_node('MatMul',['probability','V'],['result'])]
    scale=helper.make_tensor('sqrt_scale',TensorProto.FLOAT,[1],[float(np.sqrt(.125))])
    graph=helper.make_graph(nodes,'pure_sdpa',inputs,[helper.make_tensor_value_info('result',TensorProto.FLOAT,shape)],[scale])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)])
    model.ir_version=8
    onnx.checker.check_model(model);onnx.save(model,output/'sdpa.onnx')
    lines=[]
    for name in names:
        paths=[]
        for key in ['q','k','v']:
            data=np.fromfile(root/'artifacts'/f'prefix_{key}_control_20261010/board/plain_fp16'/(name+'_020.png.bin.f32'),np.float32).reshape(shape)
            data=data.astype(np.float16).astype(np.float32)
            path=output/(name+'_'+key+'.npy');np.save(path,data);paths.append(str(path))
            data.transpose(0,2,3,1).astype(np.float16).tofile(output/(name+'_'+key+'.fp16'))
        lines.append(' '.join(paths))
    (output/'dataset.txt').write_text('\n'.join(lines)+'\n')
    (output/'provenance.json').write_text(json.dumps({'scope':'pure SDPA diagnostic; actual board Q/K/V half values; no CLIP weights',
        'shape':shape,'scale':.125,'inputs_layout_on_disk':'fp16 NHWC for hardware; float32 NCHW NPY for simulator'},indent=2))
    raise SystemExit(0)

model=RKNN(verbose=False)
def check(code,op):
    if code!=0: raise RuntimeError(op+': '+str(code))
    print('DONE '+op,flush=True)
try:
    if args.phase in ['plain','step1']:
        check(model.config(target_platform='rk3588',quantized_dtype='w8a8',quantized_method='channel',
              quantized_algorithm='normal',optimization_level=3),'config')
        check(model.load_onnx(model=str(output/'sdpa.onnx')),'load')
        if args.phase=='plain':
            check(model.build(do_quantization=False),'build');check(model.export_rknn(str(output/'plain.rknn')),'export')
        else:
            check(model.hybrid_quantization_step1(dataset=str(output/'dataset.txt'),proposal=False),'step1')
    else:
        cfg=yaml.safe_load((output/'sdpa.quantization.cfg').read_text())
        ir=onnx.load(output/'sdpa.model')
        actual={name for n in ir.graph.node for name in n.output}
        selected=sorted(actual & set(cfg['quantize_parameters']))
        cfg['custom_quantize_layers']={name:'float16' for name in selected}
        (output/'all_fp16.cfg').write_text(yaml.safe_dump(cfg,sort_keys=False))
        check(model.hybrid_quantization_step2(model_input=str(output/'sdpa.model'),data_input=str(output/'sdpa.data'),
              model_quantization_cfg=str(output/'all_fp16.cfg')),'step2')
        check(model.export_rknn(str(output/'hybrid.rknn')),'export')
    if args.phase in ['plain','hybrid']:
        check(model.init_runtime(),'simulator init')
        result=[]
        for name in names:
            value=model.inference(inputs=[np.load(output/(name+'_'+key+'.npy')) for key in ['q','k','v']],data_format=['nchw']*3)
            if value is None or not np.isfinite(value[0]).all(): raise RuntimeError('invalid simulator output')
            result.append(value[0])
        np.save(output/(args.phase+'_simulator.npy'),np.asarray(result))
finally:
    model.release()
