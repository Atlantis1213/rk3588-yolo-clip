"""Full same-weight CLIP variants: do not select fused SDPA for hybrid FP16."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from partial_mlp_policy import PARTIAL_KL_DOWN, select_down_tensors

parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=['fp16_except_sdpa','sensitive_except_sdpa','sensitive_stem_head_except_sdpa',
    'protect_qkv_except_sdpa','protect_attn_proj_except_sdpa','protect_mlp_except_sdpa',
    'protect_mlp_up_except_sdpa','protect_mlp_down_except_sdpa','sensitive_kl_except_sdpa']+list(PARTIAL_KL_DOWN),required=True)
parser.add_argument('--worker',action='store_true')
args=parser.parse_args()
root=Path(__file__).resolve().parent
output=root/'artifacts/sdpa_safe_full_20261010'/args.variant

if not args.worker:
    output.mkdir(parents=True,exist_ok=False)
    with (output/'build.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'--variant',args.variant,'--worker'],stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic()
        try:
            while process.poll() is None:
                time.sleep(5)
                elapsed=time.monotonic()-started
                if int(elapsed)//30!=int(elapsed-5)//30:
                    print(f'FULL {args.variant}: {elapsed:.0f}s (native log retained)',flush=True)
                if elapsed>240:
                    process.terminate();process.wait(timeout=20)
                    raise TimeoutError('bounded full-model build exceeded 240 seconds')
        except BaseException:
            if process.poll() is None:
                process.terminate();process.wait(timeout=20)
            raise
    print('FULL '+args.variant+' exit '+str(process.returncode),flush=True)
    if process.returncode:
        print((output/'build.log').read_text(errors='replace')[-3500:])
    raise SystemExit(process.returncode)

import numpy as np
import yaml
from rknn.api import RKNN
from record_ir_provenance import fingerprint,digest
from calibration_policy import reapply_precision_policy

algorithm='kl_divergence' if args.variant=='sensitive_kl_except_sdpa' or args.variant in PARTIAL_KL_DOWN else 'normal'
ir=(root/'artifacts/calibration_algorithms_20261010/kl_divergence' if algorithm=='kl_divergence'
    else root/'artifacts/manual_hybrid_20261010')
base='all_fp16_io' if args.variant=='fp16_except_sdpa' else 'sensitive_gelu_io'
base_policy=root/'artifacts/precision_controls_20261010'/base/'policy.cfg'
os.chdir(output)
metadata={'variant':args.variant,'phase':'verify','completed':False,'pid':os.getpid(),
          'scope':'same official-reference weights and original 120-image calibration, not QAT weights',
          'base_policy':base,'sdpa_int8_count':0}
started=time.monotonic()
def status(phase,**details):
    metadata.update(phase=phase,elapsed_seconds=time.monotonic()-started,**details)
    (output/'status.json').write_text(json.dumps(metadata,indent=2));print('PHASE '+phase,flush=True)
def check(code,operation):
    if code!=0: raise RuntimeError(operation+': '+str(code))
model=RKNN(verbose=False)
try:
    status('verify')
    provenance=json.loads((ir/'ir_provenance.json').read_text())
    if fingerprint(root,algorithm=algorithm)!=provenance:
        raise RuntimeError('saved source/calibration/SDK fingerprint differs')
    cfg=yaml.safe_load(base_policy.read_text())
    layers=json.loads((root/'artifacts/diagnosis_20261010/layer_summary.json').read_text())['layers']
    names=[row['tensor'] for row in layers if row['op']=='exSDPAttention']
    if len(names)!=24 or len(set(names))!=24:
        raise RuntimeError('expected one fused SDPA output in each of 24 blocks')
    for name in names:
        if name not in cfg['custom_quantize_layers'] or cfg['quantize_parameters'][name]['dtype']!='int8':
            raise RuntimeError('unexpected base policy for '+name)
        del cfg['custom_quantize_layers'][name]
    if algorithm=='kl_divergence':
        ir_status=json.loads((ir/'status.json').read_text())
        if not ir_status.get('completed') or ir_status['algorithm']!=algorithm:
            raise RuntimeError('fresh KL step1 must complete before building')
        for entry in ir_status['step1_files'].values():
            if digest(ir/entry['name'])!=entry['sha256']:
                raise RuntimeError('fresh step1 artifact SHA mismatch: '+entry['name'])
        fresh=yaml.safe_load((ir/'clip_visual_reference_fp32.quantization.cfg').read_text())
        cfg=reapply_precision_policy(fresh,cfg)
    extra=[]
    if args.variant=='sensitive_stem_head_except_sdpa':
        extra=['pixel_values_conv','/conv1/Conv_output_0','image_embedding_mm']
        for name in extra:
            if name not in cfg['quantize_parameters']:
                raise RuntimeError('stem/head tensor missing in config: '+name)
            cfg['custom_quantize_layers'][name]='float16'
    group=None
    expected=0
    selected_blocks=None
    if args.variant=='protect_qkv_except_sdpa':
        group='qkv';expected=24
        extra=[row['tensor'] for row in layers if row['op']=='Conv' and row['tensor'].endswith('/attn/MatMul_output_0_rs_sw_gather')]
    elif args.variant=='protect_attn_proj_except_sdpa':
        group='attn_proj';expected=24
        extra=[row['tensor'] for row in layers if row['op']=='Conv' and row['tensor'].endswith('/attn/Gemm_output_0_mm')]
    elif args.variant=='protect_mlp_except_sdpa':
        group='mlp';expected=48
        extra=[row['tensor'] for row in layers if row['op']=='Conv' and '/mlp/' in row['tensor']]
    elif args.variant=='protect_mlp_up_except_sdpa':
        group='mlp_up';expected=24
        extra=[row['tensor'] for row in layers if row['op']=='Conv' and '/mlp/c_fc/' in row['tensor']]
    elif args.variant=='protect_mlp_down_except_sdpa':
        group='mlp_down';expected=24
        extra=[row['tensor'] for row in layers if row['op']=='Conv' and '/mlp/c_proj/' in row['tensor']]
    elif args.variant in PARTIAL_KL_DOWN:
        group='mlp_down';expected=8
        selected_blocks=PARTIAL_KL_DOWN[args.variant]
        extra=select_down_tensors(layers,selected_blocks)
    if group:
        if len(extra)!=expected or len(set(extra))!=expected:raise RuntimeError('unexpected group tensor count')
        for name in extra:
            if name not in cfg['quantize_parameters']:raise RuntimeError('group tensor missing: '+name)
            cfg['custom_quantize_layers'][name]='float16'
    if any('sdpa' in name.lower() for name in cfg['custom_quantize_layers']):
        raise RuntimeError('an attention-internal operand is still forced to FP16')
    policy=output/'policy.cfg';policy.write_text(yaml.safe_dump(cfg,sort_keys=False))
    status('step2',sdpa_int8_count=24,removed_fp16_sdpa=names,
           extra_fp16_outputs=extra,
           protected_matrix_group=group,protected_matrix_count=expected,
           protected_matrix_blocks=selected_blocks,
           calibration_algorithm=algorithm,
           reference_precision_variant='sensitive_except_sdpa' if args.variant=='sensitive_kl_except_sdpa' else None,
           protected_tensor_count=len(cfg['custom_quantize_layers']),
           base_policy_sha=digest(base_policy),policy_sha=digest(policy),source_sha=provenance['source_sha256'])
    # Recover the configuration embedded by the original step1, not a new config.
    check(model.hybrid_quantization_step2(model_input=str(ir/'clip_visual_reference_fp32.model'),
        data_input=str(ir/'clip_visual_reference_fp32.data'),model_quantization_cfg=str(policy)),'step2')
    artifact=output/'clip.rknn';check(model.export_rknn(str(artifact)),'export')
    status('simulate_six',model_bytes=artifact.stat().st_size,model_sha=digest(artifact))
    check(model.init_runtime(),'simulator init')
    diagnosis=root/'artifacts/diagnosis_20261010'
    selected=[r['image'] for r in json.loads((diagnosis/'reference_comparison.json').read_text())['rows']]
    values=[]
    for name in selected:
        value=model.inference(inputs=[np.load(diagnosis/'inputs'/(name+'.npy'))],data_format=['nchw'])
        if value is None or len(value)!=1 or not np.isfinite(value[0]).all():
            raise RuntimeError('invalid simulator output')
        values.append(value[0].reshape(-1));print('SIMULATOR '+name,flush=True)
    np.save(output/'simulator.npy',np.asarray(values))
    status('completed',completed=True)
except Exception as error:
    status('failed',error=repr(error));raise
finally:
    model.release()
