"""Shape-checked comparison for fused attention internal-operand protection."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parent/'artifacts'
base=root/'prefix_attn0_control_20261010'
candidate=root/'prefix_attn0_aux_control_20261010'
golden=np.load(base/'onnx_reference.npy')
simulator=np.load(candidate/'hybrid_simulator.npy')
manifest=json.loads((candidate/'board/manifest.json').read_text())
if manifest['dims']!=list(golden.shape[1:]): raise RuntimeError('unexpected output layout')
def metric(a,b):
    a,b=np.asarray(a,np.float64).ravel(),np.asarray(b,np.float64).ravel()
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all(): raise RuntimeError('invalid embedding')
    return {'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            'rmse':float(np.sqrt(np.mean((a-b)**2))),'max_abs_error':float(np.abs(a-b).max())}
rows=[]
for i,row in enumerate(manifest['rows']):
    value=np.fromfile(candidate/'board'/row['file'],np.float32)
    plain=np.fromfile(base/'board/plain_fp16'/row['file'],np.float32)
    previous=np.fromfile(base/'board/hybrid_all_fp16'/row['file'],np.float32)
    rows.append({'image':row['image'],'to_onnx':metric(value,golden[i]),'to_plain':metric(value,plain),
                 'to_previous_hybrid':metric(value,previous),'sim_to_onnx':metric(simulator[i],golden[i])})
keys=['to_onnx','to_plain','to_previous_hybrid','sim_to_onnx']
report={'samples':rows,'mean_cosine':{key:float(np.mean([r[key]['cosine'] for r in rows])) for key in keys}}
(candidate/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['mean_cosine'],indent=2))
