#!/usr/bin/env bash
# Stage 1 only. No training or CARLA server is started here.
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR=/ehsan/m.sadegh/simlingo_lidar
ENV_DIR=/data/ghazaleh/miniconda3/envs/simlingo-lidar
CONDA_BIN=/data/ghazaleh/miniconda3/bin/conda
export CUDA_VISIBLE_DEVICES=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$REPO_DIR"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$ARTIFACT_DIR/cache/pip"
export TMPDIR=/data/ghazaleh/.cache/simlingo-lidar/tmp
export DS_BUILD_OPS=0
export MAX_JOBS=4
if [[ -f "$REPO_DIR/tools/lidar_stage1/locks/constraints.txt" ]]; then
    export PIP_CONSTRAINT="$REPO_DIR/tools/lidar_stage1/locks/constraints.txt"
fi
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$ARTIFACT_DIR/reports"
rm -f "$ARTIFACT_DIR/reports/bootstrap.complete"
trap 'printf "Stage 1 environment failed at line %s\n" "$LINENO" > "$ARTIFACT_DIR/reports/bootstrap.failed"' ERR
df --output=avail -B1 "$ARTIFACT_DIR" | tail -1 | awk '$1 < 161061273600 {exit 1}'
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$CONDA_BIN" create -y -p "$ENV_DIR" python=3.10 pip setuptools=72.1.0 wheel
fi
PYTHON="$ENV_DIR/bin/python"
export PATH="$ENV_DIR/bin:$PATH"
export CUDA_HOME="$ENV_DIR"
# DeepSpeed's metadata check needs nvcc even with all extension builds disabled.
if [[ ! -x "$ENV_DIR/bin/nvcc" ]]; then
    "$CONDA_BIN" install -y -p "$ENV_DIR" -c nvidia/label/cuda-12.4.1 cuda-nvcc=12.4.131 cuda-cudart-dev=12.4.127
fi
"$PYTHON" -m pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
# imgaug declares opencv-python; install one provider of cv2, not overlapping wheels.
if "$PYTHON" -m pip show opencv-python-headless >/dev/null 2>&1; then
    "$PYTHON" -m pip uninstall -y opencv-python-headless
fi
"$PYTHON" -m pip install -r "$REPO_DIR/tools/lidar_stage1/requirements.txt"
"$PYTHON" -m pip install spconv-cu124==2.3.8
"$PYTHON" -m pip install 'torch-scatter==2.1.2+pt25cu124' --only-binary=:all: -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
FLASH_WHEEL=flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
if [[ ! -f "$ARTIFACT_DIR/cache/$FLASH_WHEEL" ]]; then
    curl --fail --location --retry 3 --continue-at - \
      'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2%2Bcu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl' \
      --output "$ARTIFACT_DIR/cache/$FLASH_WHEEL.part"
    mv "$ARTIFACT_DIR/cache/$FLASH_WHEEL.part" "$ARTIFACT_DIR/cache/$FLASH_WHEEL"
fi
"$PYTHON" -m pip install "$ARTIFACT_DIR/cache/$FLASH_WHEEL"
"$PYTHON" -m pip install deepspeed==0.16.2 --no-build-isolation
# Setuptools copies file modes during builds, which this SMB mount rejects.
# Build from a small local copy; retain the pinned archive/source on /ehsan.
LOCAL_SOURCE="$TMPDIR/Utonia-de1f9deef4ee65e508d74347f8d0eb848d878abb"
"$PYTHON" - "$ARTIFACT_DIR/source/Utonia-de1f9deef4ee65e508d74347f8d0eb848d878abb" "$LOCAL_SOURCE" <<'PY'
import shutil, sys
shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True, copy_function=shutil.copyfile,
                ignore=shutil.ignore_patterns('build', '*.egg-info', '__pycache__'))
PY
"$PYTHON" -m pip install --no-deps --no-build-isolation "$LOCAL_SOURCE"
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$ARTIFACT_DIR/reports/pip-freeze.txt"
"$CONDA_BIN" list -p "$ENV_DIR" --explicit > "$ARTIFACT_DIR/reports/conda-explicit.txt"
rm -f "$ARTIFACT_DIR/reports/bootstrap.failed"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$ARTIFACT_DIR/reports/bootstrap.complete"
