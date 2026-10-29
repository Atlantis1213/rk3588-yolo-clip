"""Serial CLIP-only validation and optional timing; no default model replacement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'quantization'))
from partial_mlp_policy import PARTIAL_KL_DOWN, verify_partial_down_inventory

parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['validate','speed'],required=True)
parser.add_argument('--variants',nargs='+',choices=['fp16_except_sdpa','sensitive_except_sdpa','sensitive_stem_head_except_sdpa',
    'protect_qkv_except_sdpa','protect_attn_proj_except_sdpa','protect_mlp_except_sdpa',
    'protect_mlp_up_except_sdpa','protect_mlp_down_except_sdpa','sensitive_kl_except_sdpa']+list(PARTIAL_KL_DOWN),
                    default=['fp16_except_sdpa','sensitive_except_sdpa'])
parser.add_argument('--speed-label',choices=['initial','stem_head','group_qkv','group_attn_proj','group_mlp','mlp_up','mlp_down','calibration_kl',
    'kl_down_early8','kl_down_mid8','kl_down_late8'],default='initial')
parser.add_argument('--inventory-suffix',choices=['','convadd_retry'],default='')
parser.add_argument('--include-kl-reference',action='store_true')
args=parser.parse_args()
output=root/'results/sdpa_safe_full_20261010'
output.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,LD_LIBRARY_PATH=str(root/'quant_runtime232'))
input_file=root/'results/quant_public_fp16_ieee/inputs_fp16/cardboard_020.png.bin'
models={'fp16_except_sdpa':'clip_fp16_except_sdpa_232.rknn',
        'sensitive_except_sdpa':'clip_sensitive_except_sdpa_232.rknn',
        'sensitive_stem_head_except_sdpa':'clip_sensitive_stem_head_except_sdpa_232.rknn'}
for variant in ['protect_qkv_except_sdpa','protect_attn_proj_except_sdpa','protect_mlp_except_sdpa',
                'protect_mlp_up_except_sdpa','protect_mlp_down_except_sdpa','sensitive_kl_except_sdpa']+list(PARTIAL_KL_DOWN):
    models[variant]='quantization/artifacts/group_ptq_20261010/active/clip.rknn'

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def run(command,log,timeout=300):
    result=subprocess.run(list(map(str,command)),cwd=root,env=env,capture_output=True,text=True,timeout=timeout)
    log.write_text(result.stdout+'\n'+result.stderr)
    result.check_returncode()
    return result

if args.phase=='validate':
    for variant in args.variants:
        model=root/models[variant]
        print('START check '+variant,flush=True)
        local_status=json.loads((root/'quantization/artifacts/sdpa_safe_full_20261010'/variant/'status.json').read_text())
        if not local_status.get('completed'):
            raise RuntimeError('source build/simulator must complete before board validation')
        if sha(model)!=local_status['model_sha']:
            raise RuntimeError('deployed model SHA mismatch for '+variant)
        probe=output/('inventory_'+variant+('_'+args.inventory_suffix if args.inventory_suffix else ''))
        if probe.exists():raise RuntimeError('existing inventory output; do not overwrite')
        run([root/'build/dump_prefix_outputs',model,probe,'--perf',input_file],output/(variant+'_inventory.log'),90)
        detail=(probe/'perf_detail.txt').read_text()
        sdpa_lines=[line for line in detail.splitlines() if re.match(r'^\s*\d+\s+exSDPAttention\s',line)]
        if len(sdpa_lines)!=24 or not all(re.search(r'exSDPAttention\s+INT8\s+NPU',line) for line in sdpa_lines):
            raise RuntimeError('not all 24 actual SDPA operators are INT8 NPU')
        reference_variant=local_status.get('reference_precision_variant')
        precision_equal=None
        if reference_variant:
            from collections import Counter
            reference_detail=(output/('inventory_'+reference_variant)/'perf_detail.txt').read_text()
            def signature(report):
                return Counter((parts[1],parts[2],parts[3],parts[-1])
                    for line in report.splitlines() if re.match(r'^\s*\d+\s+\w+\s+\w+\s+(CPU|NPU|GPU)\s',line)
                    for parts in [line.split()])
            actual,reference=signature(detail),signature(reference_detail)
            if not actual or actual!=reference:
                raise RuntimeError('compiled operator/precision/target graph changed: '
                    +str({'added':list((actual-reference).items())[:6],
                          'removed':list((reference-actual).items())[:6]}))
            precision_equal=True
            print('PASS compiled precision graph equals '+reference_variant,flush=True)
        group=local_status.get('protected_matrix_group')
        actual_group_count=None
        partial_precision=None
        protected_blocks=local_status.get('protected_matrix_blocks')
        if protected_blocks is not None:
            if args.variants.count(variant)!=1 or variant not in PARTIAL_KL_DOWN or tuple(protected_blocks)!=PARTIAL_KL_DOWN[variant]:
                raise RuntimeError('partial block metadata differs from predefined variant')
            partial_precision=verify_partial_down_inventory(detail,protected_blocks)
            actual_group_count=partial_precision['selected_down_fp16_npu']
            if actual_group_count!=local_status['protected_matrix_count']:
                raise RuntimeError('partial protected matrix count differs')
            print('PASS exact partial matrices '+str(partial_precision),flush=True)
        elif group:
            patterns={'qkv':r'/attn/MatMul(?:#|_output)', 'attn_proj':r'/attn/Gemm', 'mlp':r'/mlp/',
                      'mlp_up':r'/mlp/c_fc/', 'mlp_down':r'/mlp/c_proj/'}
            candidates=[line for line in detail.splitlines() if re.match(r'^\s*\d+\s+Conv\w*\s',line) and re.search(patterns[group],line)]
            if len(candidates)!=local_status['protected_matrix_count'] or not all(re.search(r'Conv\w*\s+FLOAT16\s+NPU',line) for line in candidates):
                raise RuntimeError('actual matrix precision differs from requested group: '+group)
            actual_group_count=len(candidates)
        destination=root/'results'/('quant_sdpa_safe_'+variant+'_ieee')
        if destination.exists():raise RuntimeError('existing accuracy dump; do not overwrite')
        print('START 120-image '+variant,flush=True)
        run([root/'build/dump_clip_embeddings',model,root/'eval_png',destination,'host'],
            output/(variant+'_validation.log'),360)
        metadata=json.loads((destination/'manifest.json').read_text())
        if len(metadata['samples'])!=120 or metadata['embedding_dim']!=768:
            raise RuntimeError('incomplete validation dump')
        (output/(variant+'_validation_status.json')).write_text(json.dumps({'phase':'completed',
            'model_sha':local_status['model_sha'],'count':120,'sdpa_int8_npu_count':24,
            'matrix_group_fp16_npu_count':actual_group_count,
            'protected_matrix_blocks':protected_blocks,
            'partial_matrix_precision':partial_precision,
            'compiled_precision_equal_to_reference':precision_equal,
            'reference_precision_variant':reference_variant,
            'result_directory':str(destination)},indent=2))
        print('DONE validation '+variant,flush=True)
else:
    timing_output=output/('speed_'+args.speed_label)
    timing_output.mkdir(exist_ok=False)
    # Run only after validation and human-visible metric inspection in the calling task.
    configs=[('fp16_all','clip_reference_fp16_232.rknn',1,'all'),
             ('fp16_3spread','clip_reference_fp16_232.rknn',3,'spread')]
    if args.include_kl_reference:
        reference_model='quantization/artifacts/partial_kl_20261010/reference/clip.rknn'
        reference_status=json.loads((output/'sensitive_kl_except_sdpa_validation_status.json').read_text())
        reference_build=json.loads((root/'quantization/artifacts/sdpa_safe_full_20261010/sensitive_kl_except_sdpa/status.json').read_text())
        if (reference_status.get('phase')!='completed' or not reference_build.get('completed')
            or sha(root/reference_model)!=reference_status['model_sha']
            or reference_status['model_sha']!=reference_build['model_sha']):
            raise RuntimeError('temporary KL reference is not the validated baseline')
        configs += [('kl_reference_all',reference_model,1,'all'),('kl_reference_3spread',reference_model,3,'spread')]
    for variant in args.variants:
        status=output/(variant+'_validation_status.json')
        if not status.exists() or json.loads(status.read_text())['phase']!='completed':
            raise RuntimeError('validation must complete before independent timing')
        if sha(root/models[variant])!=json.loads(status.read_text())['model_sha']:
            raise RuntimeError('timing model no longer matches the validation SHA')
        configs += [(variant+'_all',models[variant],1,'all'),(variant+'_3spread',models[variant],3,'spread')]
    rows=[]
    for repeat in range(2):
        for label,model,workers,mode in (configs if repeat==0 else list(reversed(configs))):
            print('START speed '+label,flush=True)
            result=run([root/'build/bench_clip_concurrency',root/model,input_file,workers,mode,8,'none','host'],
                timing_output/f'{repeat}_{label}.log',90)
            data=json.loads(result.stdout[result.stdout.find('{'):])
            rows.append({'label':label,'repeat':repeat,'model':model,'report':data})
            print(f'DONE speed {label}: {data["fps"]:.3f}FPS',flush=True)
    (timing_output/'speed_summary.json').write_text(json.dumps(rows,indent=2))
