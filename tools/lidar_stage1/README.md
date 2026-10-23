# LiDAR training: Stage 1

This directory prepares and checks the environment and one dataset archive. It
does not modify the driving model, launch training, or run Bench2Drive routes.
The agreed design is Utonia Base, connector alignment with original components
frozen, followed by joint training of the RGB branch, existing trained LoRA,
driving/navigation components, connector, and the last two LiDAR encoder stages.
All weights start from the released SimLingo and pinned Utonia checkpoints.

## Locations and isolation

- New environment: `/data/ghazaleh/miniconda3/envs/simlingo-lidar`.
- Data, assets, logs and detailed reports: `/ehsan/m.sadegh/simlingo_lidar/`.
- Original dataset archives remain in `/ehsan/m.sadegh/datasets/simlingo/archives/`.
- Original SimLingo environment and checkpoint are preserved.
- `run.sh` exposes only GPU 2 and replaces the inherited `PYTHONPATH`. The login
  shell includes an incompatible CARLA Python 3.7 egg; simply activating the new
  Python 3.10 environment is insufficient. Preflight verifies the GPU UUID and
  CARLA import location before running GPU kernels.
- `/ehsan` is SMB and rejects symbolic links and some build-time permission
  operations. Environment, compiler and temporary build files therefore reside
  on `/data`; large downloads and reports reside on `/ehsan`.

## Reproduce the preparation

Run from the repository root. Execute the long commands in tmux and retain their
exit status and logs. The fetch command uses the existing base Python (3.9+).

```bash
/data/ghazaleh/miniconda3/bin/python -B tools/lidar_stage1/fetch_assets.py
bash tools/lidar_stage1/bootstrap.sh
bash tools/lidar_stage1/run.sh -B tools/lidar_stage1/audit_archive.py
bash tools/lidar_stage1/run.sh -B tools/lidar_stage1/audit_archive.py --coordinate-only
bash tools/lidar_stage1/run.sh -u -B tools/lidar_stage1/preflight.py
```

The bootstrap uses prebuilt wheels for FlashAttention and torch-scatter matched
to Torch 2.5 / CUDA 12.4. CUDA nvcc 12.4 is included because DeepSpeed's package
metadata checks require it even with extension compilation disabled. Upstream
Utonia's setuptools build uses `pkg_resources`; build isolation is disabled and
the build runs from a small local copy using the environment's setuptools 72.1.
Utonia source and weight hashes are recorded and checked; preflight also compares
installed Python sources to the pinned source archive.

`requirements.txt` records direct dependency choices. `locks/` records the tested
complete environment; these are snapshots, not instructions to upgrade existing
environments. Utonia's local source URL in pip-freeze is reconstructed by bootstrap.
Bootstrap applies `locks/constraints.txt` to keep tested transitive package versions;
local Utonia and FlashAttention assets are pinned separately by source/hash or wheel.

## Baseline compatibility fixture

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 \
  /data/ghazaleh/miniconda3/envs/simlingo/bin/python -B \
  tools/lidar_stage1/baseline_probe.py --name original
bash tools/lidar_stage1/run.sh -B tools/lidar_stage1/baseline_probe.py \
  --name new --reference /ehsan/m.sadegh/simlingo_lidar/reports/baseline-original.npz
```

The fixture uses strict loading, the already cached InternVL snapshot, eager
FP32 attention on CPU, one real front image, fixed navigation inputs, and the
original image/LLM/driving components. It compares deterministic component
outputs using `atol=rtol=1e-4`. It does not exercise the agent's `run_step`, full
context assembly, generated commentary, or the new multimodal model.

## Verified results and review boundary

See `stage1-results.json` for the recorded checks and detailed artifact paths.
The sensor archive contains 26 routes and 2,470 synchronized frame sets. Existing
quality filters retain 23 routes and 1,727 windows under current indexing.
Language coverage counts in the detailed audit are raw matching frames; Stage 2
must intersect them with eligible windows and task-specific masks.

Preflight checks CUDA kernels and Utonia inference on a real scan. Its measured
memory is **not a training memory estimate**. No full-model gradient accumulation,
new task sampler, multimodal gradients, or runtime throughput has been validated.

Stage 1 ends at this review point. Stage 2 implements manifests, preprocessing,
the spatial connector, explicit task/loss masks, training correctness fixes, and
tiny-dataset learning. The long training run waits until the later complete
pipeline profile and presentation/time-budget review. Bench2Drive evaluation and
merging remain outside this stage.
