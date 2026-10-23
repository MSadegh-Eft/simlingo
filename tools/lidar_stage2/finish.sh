#!/usr/bin/env bash
# Complete only the bounded Stage 2 suite after pilot preparation has passed.
set -Eeuo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR=/ehsan/m.sadegh/simlingo_lidar/stage2
cd "$REPO_DIR"
trap 'result=$?; if (( result != 0 )); then echo "failed: $result" > "$ARTIFACT_DIR/logs/finish-status.txt"; fi' EXIT
echo 'waiting for pilot audit' > "$ARTIFACT_DIR/logs/finish-status.txt"
while [[ ! -f "$ARTIFACT_DIR/pilot/ready.json" ]]; do
    if ! tmux has-session -t simlingo-lidar-stage2-pilot 2>/dev/null; then
        echo 'Pilot preparation exited without a readiness report' >&2
        exit 1
    fi
    sleep 15
done
echo 'unit tests' > "$ARTIFACT_DIR/logs/finish-status.txt"
bash tools/lidar_stage1/run.sh -m unittest discover -s tests -v > "$ARTIFACT_DIR/logs/unit-tests.log" 2>&1
echo 'real data and worker checks' > "$ARTIFACT_DIR/logs/finish-status.txt"
bash tools/lidar_stage1/run.sh tools/lidar_stage2/data_checks.py --manifest "$ARTIFACT_DIR/pilot/manifest.json" > "$ARTIFACT_DIR/logs/data-checks.log" 2>&1
echo 'combined model and tiny learning' > "$ARTIFACT_DIR/logs/finish-status.txt"
bash tools/lidar_stage1/run.sh tools/lidar_stage2/smoke.py --manifest "$ARTIFACT_DIR/pilot/manifest.json" > "$ARTIFACT_DIR/logs/smoke.log" 2>&1
echo 'joint optimizer step' > "$ARTIFACT_DIR/logs/finish-status.txt"
bash tools/lidar_stage1/run.sh tools/lidar_stage2/joint_step.py --manifest "$ARTIFACT_DIR/pilot/manifest.json" > "$ARTIFACT_DIR/logs/joint-step.log" 2>&1
echo 'released baseline regression' > "$ARTIFACT_DIR/logs/finish-status.txt"
bash tools/lidar_stage1/run.sh tools/lidar_stage1/baseline_probe.py --name stage2-regression --reference /ehsan/m.sadegh/simlingo_lidar/reports/baseline-original.npz > "$ARTIFACT_DIR/logs/baseline-regression.log" 2>&1
bash tools/lidar_stage1/run.sh tools/lidar_stage2/report.py > "$ARTIFACT_DIR/logs/report.log" 2>&1
echo 'complete' > "$ARTIFACT_DIR/logs/finish-status.txt"
