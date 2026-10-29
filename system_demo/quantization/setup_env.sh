#!/usr/bin/env bash
set -euo pipefail

TASK_QUANT_ENV="${TASK_QUANT_ENV:-/home/atlantis/yolo-clip-quant-venv}"
TASK_QUANT_ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$TASK_QUANT_ROOT/assets"
if [ ! -x "$TASK_QUANT_ENV/bin/pip" ]; then
  curl --fail --location --retry 2 --output "$TASK_QUANT_ROOT/assets/virtualenv.pyz" \
    https://bootstrap.pypa.io/virtualenv.pyz
  python3 "$TASK_QUANT_ROOT/assets/virtualenv.pyz" "$TASK_QUANT_ENV"
fi
"$TASK_QUANT_ENV/bin/python" -m pip install --upgrade pip
# CPU wheels avoid installing CUDA dependencies on the WSL conversion host.
"$TASK_QUANT_ENV/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.2.0+cpu' 'torchvision==0.17.0+cpu'
"$TASK_QUANT_ENV/bin/python" -m pip install 'setuptools==69.5.1' 'numpy==1.26.4' \
  'onnx==1.16.1' 'onnxruntime==1.18.1' 'opencv-python==4.10.0.84' 'scipy==1.12.0' \
  'rknn-toolkit2==2.3.2' pillow pyyaml
"$TASK_QUANT_ENV/bin/python" -m pip install \
  'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
"$TASK_QUANT_ENV/bin/python" -m pip freeze
