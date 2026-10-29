"""Isolated subprocess logs for the weightless reproducer."""
from pathlib import Path
import subprocess
import sys
import json

root=Path(__file__).resolve().parent
output=root/'artifacts/standalone_sdpa_20261010'
output.mkdir(exist_ok=False)
for phase in ['prepare','plain','step1','hybrid']:
    print('STANDALONE '+phase,flush=True)
    with (output/(phase+'.log')).open('w') as log:
        try:
            subprocess.run([sys.executable,str(root/'build_standalone_sdpa.py'),'--phase',phase],
                stdout=log,stderr=subprocess.STDOUT,timeout=90,check=True)
        except Exception as error:
            (output/'status.json').write_text(json.dumps({'phase':'failed','stage':phase,'error':repr(error)}));raise
(output/'status.json').write_text(json.dumps({'phase':'completed'}))
