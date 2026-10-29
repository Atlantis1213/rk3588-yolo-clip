"""Package only small code/config assets; the remote downloads weights/data itself."""
import json
from pathlib import Path
import zipfile
import yaml

root = Path(__file__).resolve().parent
cfg = yaml.safe_load((root/'artifacts/manual_hybrid_20261010/clip_visual_reference_fp32.quantization.cfg').read_text())
parameters = {name:{'scale':value.get('scale',[]),'zero_point':value.get('zero_point',[])}
              for name,value in cfg['quantize_parameters'].items()
              if len(value.get('scale',[]))==1 and value.get('dtype')=='int8'}
destination = root/'artifacts/qat_pilot_bundle.zip'
with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in ['qat_visual_core.py','test_qat_visual_core.py','test_distillation_loss.py','train_qat.py',
                 'remote_download_assets.py','run_qat_pipeline.py']:
        archive.write(root/name,name)
    archive.write(root/'assets/clip_model_official.py','clip_model_official.py')
    for name in ['dataset_provenance.json','eval_labels.json','final_test_labels.json','public_text_features.npy']:
        archive.write(root/'artifacts'/name,name)
    archive.writestr('activation_parameters.json',json.dumps(parameters))
print(json.dumps({'bundle':str(destination),'bytes':destination.stat().st_size,
                  'int8_per_layer_entries':len(parameters)}))
