#!/usr/bin/env bash
# Isolate the Python 3.10 stack from the login shell's legacy CARLA egg.
set -Eeuo pipefail
ulimit -c 0
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR=/data/ghazaleh/miniconda3/envs/simlingo-lidar
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_DIR"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME="$ENV_DIR"
export OMP_NUM_THREADS=4
export MAX_JOBS=4
export TMPDIR=/data/ghazaleh/.cache/simlingo-lidar/tmp
exec "$ENV_DIR/bin/python" -X faulthandler "$@"
