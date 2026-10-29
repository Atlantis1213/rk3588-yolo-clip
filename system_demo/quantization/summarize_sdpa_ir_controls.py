"""Check whether modified SDPA scale affects simulator and real hardware consistently."""
from pathlib import Path
import json
import numpy as np

root=Path(__file__).resolve().parent/'artifacts'
golden=np.load(root/'prefix_core_control_20261010/onnx_reference.npy')
names=[name+'_020.png.bin.f32' for name in ['cardboard','glass','metal','paper','plastic','trash']]
def metric(a,b):
    a,b=np.asarray(a,np.float64).ravel(),np.asarray(b,np.float64).ravel()
    return {'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            'max_abs_error':float(np.abs(a-b).max()),'rmse':float(np.sqrt(np.mean((a-b)**2)))}
rows=[]
for label in ['scale0','scale1','scale0015625']:
    directory=root/('sdpa_ir_'+label+'_20261010')
    sim=np.load(directory/'simulator.npy')
    for i,name in enumerate(names):
        actual=np.fromfile(directory/'board'/name,np.float32)
        prior=np.fromfile(root/'prefix_core_control_20261010/board/hybrid_all_fp16'/name,np.float32)
        rows.append({'label':label,'image':name,'board_to_original_hybrid':metric(actual,prior),
                     'board_to_onnx':metric(actual,golden[i]),'simulator_to_onnx':metric(sim[i],golden[i]),
                     'board_to_own_simulator':metric(actual,sim[i])})
summary=[{'label':label,**{key:float(np.mean([r[key]['cosine'] for r in rows if r['label']==label]))
    for key in ['board_to_original_hybrid','board_to_onnx','simulator_to_onnx','board_to_own_simulator']},
    'max_abs_error_to_original':max(r['board_to_original_hybrid']['max_abs_error'] for r in rows if r['label']==label)}
    for label in ['scale0','scale1','scale0015625']]
report={'summary':summary,'samples':rows}
(root/'sdpa_ir_scale_comparison_20261010.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2))
