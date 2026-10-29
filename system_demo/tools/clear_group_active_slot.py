"""Delete ONLY this run's finished temporary model; canonical local model is retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1].resolve()
sys.path.insert(0,str(root/'quantization'))
from partial_mlp_policy import PARTIAL_KL_DOWN

parser=argparse.ArgumentParser()
parser.add_argument('--variant',choices=['protect_qkv_except_sdpa','protect_attn_proj_except_sdpa','protect_mlp_except_sdpa',
    'protect_mlp_up_except_sdpa','protect_mlp_down_except_sdpa','sensitive_kl_except_sdpa']+list(PARTIAL_KL_DOWN),required=True)
parser.add_argument('--reference',action='store_true',help='clear only the temporary validated KL baseline copy')
args=parser.parse_args()
if args.reference and args.variant!='sensitive_kl_except_sdpa':
    raise ValueError('reference cleanup requires the validated KL baseline variant')
relative_parent=('quantization/artifacts/partial_kl_20261010/reference' if args.reference
    else 'quantization/artifacts/group_ptq_20261010/active')
slot=(root/relative_parent/'clip.rknn').resolve()
expected_parent=(root/relative_parent).resolve()
if slot.parent!=expected_parent or slot.name!='clip.rknn' or slot.is_symlink():raise RuntimeError('unsafe active-slot path')
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
    try:
        command=(entry/'cmdline').read_bytes().split(b'\0')
    except (OSError,PermissionError):
        continue
    if (any(b'run_partial_kl_final_test_board.py' in arg for arg in command)
        or str(slot).encode() in command or (any(b'run_sdpa_safe_full_board.py' in arg for arg in command)
        and (args.variant.encode() in command or (args.reference and b'--include-kl-reference' in command)))):
        raise RuntimeError('temporary model is still used by an active experiment; preserve it')
build=json.loads((root/'quantization/artifacts/sdpa_safe_full_20261010'/args.variant/'status.json').read_text())
validation=json.loads((root/'results/sdpa_safe_full_20261010'/(args.variant+'_validation_status.json')).read_text())
if not build.get('completed') or validation.get('phase')!='completed':raise RuntimeError('unfinished experiment; preserve model')
digest=hashlib.sha256()
with slot.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
if digest.hexdigest()!=build['model_sha'] or digest.hexdigest()!=validation['model_sha']:
    raise RuntimeError('active file is not this finished variant')
slot.unlink()
print(json.dumps({'deleted_temporary_file':str(slot),'variant':args.variant,'model_sha':digest.hexdigest(),
                  'canonical_model_retained_on_local_workspace':True}))
