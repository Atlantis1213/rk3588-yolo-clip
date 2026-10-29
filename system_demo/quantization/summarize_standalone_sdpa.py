"""Numerical independent-input attention reproducer, baseline acts as IO/layout control."""
from pathlib import Path
import json
import numpy as np
import torch

torch.set_num_threads(4)
root=Path(__file__).resolve().parent/'artifacts/standalone_sdpa_20261010'
names=['cardboard','glass','metal','paper','plastic','trash']
for name in ['plain','hybrid']:
    attrs=json.loads((root/('board_'+name)/'manifest.json').read_text())
    if attrs['dims']!=[1,16,257,64]: raise RuntimeError('unknown output layout')
def metric(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    return {'cosine':float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))),
            'rmse':float(torch.sqrt(torch.mean((a-b)**2))),'max_abs_error':float(torch.max(torch.abs(a-b)))}
rows=[]
plain_sim=torch.from_numpy(np.load(root/'plain_simulator.npy'))
hybrid_sim=torch.from_numpy(np.load(root/'hybrid_simulator.npy'))
for i,name in enumerate(names):
    q,k,v=[torch.from_numpy(np.load(root/(name+'_'+key+'.npy'))) for key in ['q','k','v']]
    golden=torch.softmax((q@k.transpose(-2,-1))*.125,dim=-1)@v
    uniform=v.mean(dim=-2,keepdim=True).expand_as(golden)
    plain,hybrid=[torch.from_numpy(np.fromfile(root/('board_'+kind)/(name+'.f32'),np.float32).reshape(1,16,257,64)) for kind in ['plain','hybrid']]
    rows.append({'image':name,'plain_to_correct':metric(plain,golden),'hybrid_to_correct':metric(hybrid,golden),
        'hybrid_to_uniform':metric(hybrid,uniform),'hybrid_query_token_variation':float(torch.max(torch.abs(hybrid-hybrid[:,:,:1,:]))),
        'plain_sim_to_correct':metric(plain_sim[i],golden),'hybrid_sim_to_correct':metric(hybrid_sim[i],golden)})
keys=['plain_to_correct','hybrid_to_correct','hybrid_to_uniform','plain_sim_to_correct','hybrid_sim_to_correct']
report={'mean_cosine':{key:float(np.mean([r[key]['cosine'] for r in rows])) for key in keys},'samples':rows}
(root/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
