"""Read-only comparison of source QKV row ordering with saved hybrid IR."""
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

root=Path(__file__).resolve().parent/'artifacts'
source=onnx.load(root/'prefix_core_control_20261010/prefix.onnx')
ir=onnx.load(root/'prefix_core_control_20261010/prefix.model')
initial={v.name:v for v in source.graph.initializer}
ir_initial={v.name:v for v in ir.graph.initializer}
node=next(n for n in source.graph.node if n.name.endswith('/attn/MatMul'))
weight=numpy_helper.to_array(initial[node.input[1]]).T.copy()
conv=next(n for n in ir.graph.node if n.op_type=='Conv' and any('attn/MatMul_output' in o for o in n.output))
converted=numpy_helper.to_array(ir_initial[conv.input[1]]).reshape(3072,1024)
hash_row=lambda row:hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
lookup={hash_row(row):i for i,row in enumerate(weight)}
mapping=[lookup.get(hash_row(row),-1) for row in converted]
expected=np.arange(3072).reshape(3,16,64).transpose(1,0,2).reshape(-1)
report={'source_weight_shape':list(weight.shape),'ir_weight_shape':list(converted.shape),
        'all_rows_found':-1 not in mapping,'ir_matches_original':np.array_equal(converted,weight),
        'ir_matches_head_interleaving':np.array_equal(converted,weight[expected]),
        'first_200_row_source_indices':mapping[:200]}
(root/'core_weight_order_audit_20261010.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='first_200_row_source_indices'},indent=2))
