#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data2/lxj/projects/CervixAgent
BASE_PYTHON="$PROJECT_ROOT/.envs/rag/bin/python"
ENV_ROOT="$PROJECT_ROOT/.envs/vl_rag"
PIP_CACHE="$PROJECT_ROOT/tmp/pip-cache-vl-rag"
REPORT_DIR="$PROJECT_ROOT/configs/environments"

test -x "$BASE_PYTHON"
mkdir -p "$PIP_CACHE" "$REPORT_DIR"

if [ ! -x "$ENV_ROOT/bin/python" ]; then
  "$BASE_PYTHON" -m venv "$ENV_ROOT"
fi

PYTHON="$ENV_ROOT/bin/python"
PIP="$PYTHON -m pip"

PIP_CACHE_DIR="$PIP_CACHE" $PIP install --upgrade pip

# The Qwen3-VL embedding/reranker model cards specify the CUDA 12.8-compatible
# PyTorch 2.8 stack. This environment is isolated from the document parser.
PIP_CACHE_DIR="$PIP_CACHE" $PIP install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

PIP_CACHE_DIR="$PIP_CACHE" $PIP install \
  'transformers>=4.57.0' \
  'qwen-vl-utils>=0.0.14' \
  'sentence-transformers>=5.0.0' \
  'accelerate>=1.0.0' \
  'qdrant-client>=1.14.0' \
  'pillow>=11.0.0' \
  'pydantic>=2.0.0' \
  'psutil>=6.0.0'

"$PYTHON" - <<'PY'
import json
import platform
from importlib.metadata import version

import torch

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "transformers": version("transformers"),
    "sentence_transformers": version("sentence-transformers"),
    "qwen_vl_utils": version("qwen-vl-utils"),
    "qdrant_client": version("qdrant-client"),
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

"$PYTHON" -m pip freeze | sort > "$REPORT_DIR/vl_rag_pip_freeze_20260725.txt"
