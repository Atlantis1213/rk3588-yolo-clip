"""Assert the captured defect and causal control; not a claim of production repair."""
from pathlib import Path
import json
import re

root=Path(__file__).resolve().parent/'artifacts'
for segment in ['qkv','q','k','v']:
    report=json.loads((root/f'prefix_{segment}_control_20261010/comparison.json').read_text())
    assert len(report['samples'])==6
    assert all(r['board_hybrid_to_plain']['max_abs_error']==0 for r in report['samples'])
    assert report['mean_cosine']['board_plain_to_onnx']>.9999
reproducer=root/'standalone_sdpa_20261010'
report=json.loads((reproducer/'comparison.json').read_text())
assert len(report['samples'])==6
assert report['mean_cosine']['plain_to_correct']>.999
assert report['mean_cosine']['hybrid_to_correct']<.7
assert report['mean_cosine']['hybrid_to_uniform']>.999999
assert all(r['hybrid_query_token_variation']==0 for r in report['samples'])
for mode in ['plain','hybrid']:
    assert re.search(r'exSDPAttention\s+FLOAT16\s+NPU',(reproducer/('board_'+mode)/'perf_detail.txt').read_text())
control=json.loads((root/'standalone_sdpa_keep_int8_20261010/comparison.json').read_text())
assert control['mean_cosine_to_correct']>.9
assert all(r['query_token_variation']>1 for r in control['samples'])
print('PASS: QKV matches; standalone hybrid-FP16 attention becomes uniform; INT8 restores query dependence')
