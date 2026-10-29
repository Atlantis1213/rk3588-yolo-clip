"""Fixed group ablations, serialized SDK workers with separate output directories."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for variant in ['protect_qkv_except_sdpa','protect_attn_proj_except_sdpa','protect_mlp_except_sdpa']:
    subprocess.run([sys.executable,str(root/'build_sdpa_safe_full.py'),'--variant',variant],check=True)
