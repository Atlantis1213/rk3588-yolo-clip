"""Verify changed attention dtype and restored query dependence, not full-model accuracy."""
from pathlib import Path
import json
import re
import numpy as np
import torch

torch.set_num_threads(4)
root=Path(__file__).resolve().parent/'artifacts'
source=root/'standalone_sdpa_20261010'
output=root/'standalone_sdpa_keep_int8_20261010'
if not re.search(r'exSDPAttention\s+INT8\s+NPU',(output/'board/perf_detail.txt').read_text()):
    raise RuntimeError('attention dtype did not actually become INT8')
def metric(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    return {'cosine':float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))),
            'rmse':float(torch.sqrt(torch.mean((a-b)**2))),'max_abs_error':float(torch.max(torch.abs(a-b)))}
rows=[]
sim=torch.from_numpy(np.load(output/'simulator.npy'))
for i,name in enumerate(['cardboard','glass','metal','paper','plastic','trash']):
    q,k,v=[torch.from_numpy(np.load(source/(name+'_'+key+'.npy'))) for key in ['q','k','v']]
    correct=torch.softmax(q@k.transpose(-2,-1)*.125,dim=-1)@v
    actual=torch.from_numpy(np.fromfile(output/'board'/(name+'.f32'),np.float32).reshape(1,16,257,64))
    rows.append({'image':name,'to_correct':metric(actual,correct),'sim_to_correct':metric(sim[i],correct),
                 'query_token_variation':float(torch.max(torch.abs(actual-actual[:,:,:1,:])))})
report={'actual_operator_dtype':'INT8','actual_operator_target':'NPU',
        'mean_cosine_to_correct':float(np.mean([r['to_correct']['cosine'] for r in rows])),
        'scope':'attention-only diagnostic, not complete-model recovery','samples':rows}
(output/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
