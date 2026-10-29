"""Summarize only supported NHWC submissions; keep malformed-layout negative control."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent/'artifacts/diagnosis_20261010'
data = root/'layout_core_v2_20261010'
references = np.load(root/'torch.f32.npy')
names = [row['image']+'.bin' for row in json.loads((root/'reference_comparison.json').read_text())['rows']]
variants = {}
for directory in sorted(data.iterdir()):
    if not directory.is_dir():
        continue
    manifest = json.loads((directory/'manifest.json').read_text())
    variants[directory.name] = { (r['image'],r['mode']):np.fromfile(directory/r['file'],np.float32)
                                for r in manifest['rows'] }


def metric(a,b):
    a,b = np.asarray(a,np.float64),np.asarray(b,np.float64)
    if a.shape!=(768,) or b.shape!=(768,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('invalid embedding output')
    return {'cosine':float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            'max_abs_error':float(np.abs(a-b).max())}


summary = []
samples = []
for model in ['baseline','all_fp16_io','sensitive_gelu_io']:
    for core in ['all','0']:
        value = variants[model+'_'+core]
        records = []
        for index,name in enumerate(names):
            canonical = value[(name,'hwc_nhwc')]
            other_core = variants[model+('_0' if core=='all' else '_all')][(name,'hwc_nhwc')]
            baseline = variants['baseline_'+core][(name,'hwc_nhwc')]
            wrong_baseline = variants['baseline_'+core][(name,'chw_nhwc_wrong')]
            row = {'variant':model+'_'+core,'image':name,
                'to_torch':metric(canonical,references[index]),
                'to_baseline':metric(canonical,baseline),
                'to_other_core':metric(canonical,other_core),
                'to_host_fp16':metric(canonical,value[(name,'fp16_nhwc')]),
                'to_native_passthrough':metric(canonical,value[(name,'native_passthrough')]),
                'to_wrong_layout_baseline':metric(canonical,wrong_baseline)}
            records.append(row)
        samples.extend(records)
        summary.append({'variant':model+'_'+core,**{key:float(np.mean([r[key]['cosine'] for r in records]))
            for key in ['to_torch','to_baseline','to_other_core','to_host_fp16',
                        'to_native_passthrough','to_wrong_layout_baseline']}})
runtime_rows = []
runtime_directory = root/'layout_runtime230_20261010'
if runtime_directory.exists():
    for model in ['baseline','all_fp16_io','sensitive_gelu_io']:
        directory = runtime_directory/(model+'_all')
        manifest = json.loads((directory/'manifest.json').read_text())
        value = {(r['image'],r['mode']):np.fromfile(directory/r['file'],np.float32) for r in manifest['rows']}
        rows = [metric(value[(name,'hwc_nhwc')],variants[model+'_all'][(name,'hwc_nhwc')]) for name in names]
        runtime_rows.append({'variant':model,'system_api_version':manifest['api_version'],
            'driver_version':manifest['driver_version'],
            'cosine_mean_system_to232':float(np.mean([r['cosine'] for r in rows])),
            'max_abs_error_system_to232':max(r['max_abs_error'] for r in rows)})
report = {'protocol':'six fixed validation tensors; no training; isolated runtime232 and system-runtime subprocesses; serial runs',
          'excluded':'Initial v1 NCHW-normalize rejected modes; no results used',
          'summary':summary,'samples':samples,'runtime_comparison':runtime_rows}
(root/'layout_core_comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2))
print(json.dumps(runtime_rows,indent=2))
