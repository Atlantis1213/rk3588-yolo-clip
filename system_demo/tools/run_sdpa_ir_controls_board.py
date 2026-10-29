"""Hardware causal probes on deliberately altered scale IR, not deployment models."""
from pathlib import Path
import os
import subprocess

root=Path(__file__).resolve().parents[1]
inputs=[root/'results/quant_public_fp16_ieee/inputs_fp16'/(name+'_020.png.bin')
        for name in ['cardboard','glass','metal','paper','plastic','trash']]
env=dict(os.environ,LD_LIBRARY_PATH=str(root/'quant_runtime232'))
for label in ['scale0','scale1','scale0015625']:
    directory='sdpa_ir_'+label+'_20261010'
    output=root/'results'/directory
    if output.exists(): raise RuntimeError('existing result directory; do not overwrite')
    subprocess.run([str(root/'build/dump_prefix_outputs'),
        str(root/'quantization/artifacts'/directory/'control.rknn'),str(output),*[str(p) for p in inputs]],
        cwd=root,env=env,timeout=60,check=True)
