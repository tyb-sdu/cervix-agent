#!/usr/bin/env bash
set -euo pipefail

project_root="/data2/lxj/projects/CervixAgent"
core_python="${project_root}/.envs/core/bin/python"
rag_environment="${project_root}/.envs/rag"
rag_python="${rag_environment}/bin/python"
record_root="${project_root}/configs/environments"
pip_cache="${project_root}/tmp/pip-cache"

if [[ ! -x "${core_python}" ]]; then
    printf 'Missing project-local core Python: %s\n' "${core_python}" >&2
    exit 2
fi
if [[ -e "${rag_environment}" ]]; then
    printf 'RAG environment already exists: %s\n' "${rag_environment}" >&2
    exit 2
fi

mkdir -p "${record_root}" "${pip_cache}"
export PIP_CACHE_DIR="${pip_cache}"
export HF_HOME="${project_root}/models/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${project_root}/models/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

"${core_python}" -m venv --copies "${rag_environment}"
"${rag_python}" -m pip install \
    --disable-pip-version-check \
    --no-input \
    --upgrade \
    pip setuptools wheel

# Official PyTorch CUDA 12.8 wheels. The installed NVIDIA driver reports CUDA
# compatibility 13.2 and is backward-compatible with this runtime.
"${rag_python}" -m pip install \
    --disable-pip-version-check \
    --no-input \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0" \
    "torchvision==0.26.0"

"${rag_python}" -m pip install \
    --disable-pip-version-check \
    --no-input \
    docling \
    sentence-transformers \
    qdrant-client \
    lxml \
    beautifulsoup4 \
    pypdf \
    pymupdf \
    pandas \
    pyarrow \
    rank-bm25 \
    rapidfuzz \
    pytest

"${rag_python}" - <<'PY'
import json
import torch

result = {
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
}
print(json.dumps(result, ensure_ascii=False, indent=2))
assert result["cuda_available"] is True
assert result["gpu_count"] == 2
assert all(name == "NVIDIA GeForce RTX 4090" for name in result["gpu_names"])

tensor = torch.rand((1024, 1024), device="cuda:0")
value = (tensor @ tensor.T).mean().item()
print(f"cuda_smoke_value={value:.8f}")
PY

"${rag_python}" - <<'PY'
import docling
import lxml
import pandas
import pyarrow
import pymupdf
import pypdf
import qdrant_client
import sentence_transformers

print("rag_imports=passed")
PY

"${rag_python}" -m pip freeze \
    > "${record_root}/rag_pilot_pip_freeze_20260724.txt"
sha256sum "${record_root}/rag_pilot_pip_freeze_20260724.txt" \
    > "${record_root}/rag_pilot_pip_freeze_20260724.txt.sha256"

"${rag_python}" - <<'PY' \
    > "${record_root}/rag_gpu_validation_20260724.json"
import json
import torch

print(json.dumps({
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
    ],
}, ensure_ascii=False, indent=2))
PY
sha256sum "${record_root}/rag_gpu_validation_20260724.json" \
    > "${record_root}/rag_gpu_validation_20260724.json.sha256"

printf 'RAG_ENVIRONMENT=%s\n' "${rag_environment}"
printf 'PYTHON=%s\n' "$("${rag_python}" --version 2>&1)"
printf 'PYTORCH=%s\n' "$("${rag_python}" -c 'import torch; print(torch.__version__)')"
printf 'DOCLING=%s\n' "$("${rag_python}" -c 'from importlib.metadata import version; print(version("docling"))')"
printf 'SENTENCE_TRANSFORMERS=%s\n' "$("${rag_python}" -c 'import sentence_transformers; print(sentence_transformers.__version__)')"
printf 'QDRANT_CLIENT=%s\n' "$("${rag_python}" -c 'from importlib.metadata import version; print(version("qdrant-client"))')"
printf 'STATUS=passed\n'
