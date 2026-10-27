#!/usr/bin/env bash
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$REPO_DIR/tools/lidar_stage1/run.sh" "$REPO_DIR/tools/lidar_training/manage.py" "$@"
