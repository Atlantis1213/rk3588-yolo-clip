"""Falsifiable SDPA scale, hidden-QDQ and QKV packing predictions on fixed hardware data."""
from pathlib import Path
import json
import math
import numpy as np
import torch
import yaml

torch.set_num_threads(4)
root=Path(__file__).resolve().parent/'artifacts'
names=[name+'_020.png.bin.f32' for name in ['cardboard','glass','metal','paper','plastic','trash']]
cfg=yaml.safe_load((root/'prefix_core_control_20261010/prefix.quantization.cfg').read_text())['quantize_parameters']
qk_cfg=next(v for k,v in cfg.items() if k.endswith('_q_matmul'))
sm_cfg=next(v for k,v in cfg.items() if k.endswith('_q_softmax'))

def qdq(x,p):
    return (torch.clamp(torch.round(x/p['scale'][0])+p['zero_point'][0],-128,127)-p['zero_point'][0])*p['scale'][0]

def metric(a,b):
    a,b=a.double().flatten(),b.double().flatten()
    return {'cosine':float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))),
            'rmse':float(torch.sqrt(torch.mean((a-b)**2))), 'max_abs_error':float(torch.max(torch.abs(a-b)))}

rows=[]
for i,name in enumerate(names):
    q,k,v=[torch.from_numpy(np.fromfile(root/f'prefix_{key}_control_20261010/board/plain_fp16'/name,np.float32).reshape(1,16,257,64)) for key in ['q','k','v']]
    qkv=torch.from_numpy(np.fromfile(root/'prefix_qkv_control_20261010/board/plain_fp16'/name,np.float32).reshape(257,3072))
    plain=torch.from_numpy(np.fromfile(root/'prefix_core_control_20261010/board/plain_fp16'/name,np.float32).reshape(1,16,257,64))
    hybrid=torch.from_numpy(np.fromfile(root/'prefix_core_control_20261010/board/hybrid_all_fp16'/name,np.float32).reshape(1,16,257,64))
    dot=q@k.transpose(-2,-1)
    candidates={}
    for scale in [0.,1/512,1/256,1/128,1/(64*math.sqrt(2)),1/64,1/(32*math.sqrt(2)),1/32,1/16,1/8,1/4,1/math.sqrt(8),1/2,1.,2.,4.,8.]:
        candidates['scale_'+str(scale)]=torch.softmax(dot*scale,dim=-1)@v
    for before in [False,True]:
        for softmax_q in [False,True]:
            scores=qdq(dot,qk_cfg)*.125 if before else qdq(dot*.125,qk_cfg)
            probabilities=torch.softmax(scores,dim=-1)
            if softmax_q: probabilities=qdq(probabilities,sm_cfg)
            candidates[f'qdq_qk_before_scale_{before}_softmax_{softmax_q}']=probabilities@v
    # Wrong reshape grouping: [head,QKV,channel] instead of [QKV,head,channel].
    mixed=qkv.reshape(257,16,3,64).permute(2,1,0,3).unsqueeze(1)
    mq,mk,mv=mixed[0],mixed[1],mixed[2]
    candidates['wrong_qkv_head_interleaving']=torch.softmax((mq@mk.transpose(-2,-1))*.125,dim=-1)@mv
    candidates['swapped_qk']=torch.softmax((k@q.transpose(-2,-1))*.125,dim=-1)@v
    candidates['softmax_wrong_axis']=torch.softmax(dot*.125,dim=-2)@v
    for label,value in candidates.items():
        rows.append({'image':name,'hypothesis':label,'to_plain':metric(value,plain),'to_hybrid':metric(value,hybrid)})
labels=list(candidates)
summary=[{'hypothesis':label,
          'cosine_to_hybrid':float(np.mean([r['to_hybrid']['cosine'] for r in rows if r['hypothesis']==label])),
          'rmse_to_hybrid':float(np.mean([r['to_hybrid']['rmse'] for r in rows if r['hypothesis']==label])),
          'cosine_to_plain':float(np.mean([r['to_plain']['cosine'] for r in rows if r['hypothesis']==label]))} for label in labels]
summary.sort(key=lambda r:r['cosine_to_hybrid'],reverse=True)
report={'note':'Numerical predictions, not alone proof of compiler implementation',
        'qk_cfg':qk_cfg,'softmax_cfg':sm_cfg,'summary':summary,'samples':rows}
(root/'core_numerical_hypotheses_20261010.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary[:10],indent=2))
