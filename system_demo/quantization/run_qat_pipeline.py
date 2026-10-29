"""Download verified assets, test CUDA seam, then run bounded pilot independently of SSH."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import argparse

root = Path(__file__).resolve().parent
status_file = root/'pipeline_status.json'


def status(phase, **details):
    value = {'phase':phase,'pid':os.getpid(),'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**details}
    temporary = status_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary,status_file)
    print(json.dumps(value),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['pilot','ranking-ablation','projection-ablation'],default='pilot')
    args=parser.parse_args()
    # WMI launches this process outside the SSH session job; retain logs locally.
    sys.stdout = (root/'pipeline_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr = (root/'pipeline_stderr.log').open('a',encoding='utf-8',buffering=1)
    try:
        stages=[
            ('download',['remote_download_assets.py']),
            ('cuda_regression',['test_qat_visual_core.py','--device','cuda']),
            ('loss_regression',['test_distillation_loss.py'])]
        if args.experiment=='pilot':
            stages += [('qat',['train_qat.py','--steps','300','--eval-every','50'])]
        elif args.experiment=='ranking-ablation':
            stages += [
                ('qat_kl5',['train_qat.py','--run','qat_ln_kl5_20261010','--kl-weight','5','--steps','300']),
                ('qat_kl5_rank1',['train_qat.py','--run','qat_ln_kl5_rank1_20261010',
                                 '--kl-weight','5','--rank-weight','1','--steps','300'])]
        else:
            stages += [('qat_ln_projection',['train_qat.py','--run','qat_ln_projection_20261010',
                         '--train-projection','--steps','300'])]
        for phase,arguments in stages:
            status(phase)
            subprocess.run([sys.executable,'-u',*arguments],cwd=root,check=True,
                           stdout=sys.stdout,stderr=sys.stderr)
        status('completed')
    except Exception as error:
        status('failed',error=repr(error),traceback=traceback.format_exc())
        raise
