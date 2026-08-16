conda activate simlingo
export CARLA_ROOT=/data/ghazaleh/carla
export WORK_DIR=/data/ghazaleh/simlingo
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/Bench2Drive/leaderboard
export PYTHONPATH=${WORK_DIR}:${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}