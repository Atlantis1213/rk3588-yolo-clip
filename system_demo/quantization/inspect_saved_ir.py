"""Read-only node/tensor routing for the SDK's saved editable graph."""
from pathlib import Path
import onnx

root=Path(__file__).resolve().parent/'artifacts/manual_hybrid_20261010'
model=onnx.load(str(root/'clip_visual_reference_fp32.model'))
print('REAL_INPUT',[x.name for x in model.graph.input if x.name=='pixel_values'])
print('OUTPUTS',[x.name for x in model.graph.output])
print('FIRST_NODES',[(node.op_type,list(node.input),list(node.output)) for node in model.graph.node[:3]])
print('LAST_NODES',[(node.op_type,list(node.input),list(node.output)) for node in model.graph.node[-8:]])
