# Stage 2 checks

Run from the repository root with the isolated Stage 1 launcher. It exposes only
physical GPU 2 and avoids the incompatible CARLA egg in the login environment.
All persistent data and logs live under `/ehsan/m.sadegh/simlingo_lidar/stage2`.

```bash
bash tools/lidar_stage1/run.sh -m unittest discover -s tests -v
bash tools/lidar_stage1/run.sh tools/lidar_stage2/data_checks.py
bash tools/lidar_stage1/run.sh tools/lidar_stage2/smoke.py
bash tools/lidar_stage1/run.sh tools/lidar_stage2/joint_step.py
bash tools/lidar_stage1/run.sh tools/lidar_stage1/baseline_probe.py \
  --name stage2-regression \
  --reference /ehsan/m.sadegh/simlingo_lidar/reports/baseline-original.npz
```

The smoke check verifies strict pretrained loading; sensor/target separation;
128-token LiDAR prefixes; direct-backbone parity with the PEFT language model;
cached/uncached greedy generation; separate driving/language gradients; 16 tiny
alignment updates; frozen weights; full model/optimizer/CUDA RNG recovery; and
backward passes for all task families. A pilot geometry-QA example failed cached
generation parity; caching remains disabled and the suite uses uncached generation.
Its temporary checkpoint is removed after
verification. The joint check performs one separate disposable optimizer update
and verifies every parameter group, unchanged frozen weights and FP32 Adam state.
Neither check writes over pretrained checkpoints or produces a selected model.

Pilot preparation is restartable at completed shard boundaries. Use tmux:

```bash
tmux new-session -d -s simlingo-lidar-stage2-pilot \
  'cd /data/ghazaleh/simlingo && bash tools/lidar_stage1/run.sh \
  -m simlingo_training.lidar.storage --include-validation --minimum-train 50000 \
  --output /ehsan/m.sadegh/simlingo_lidar/stage2/pilot \
  > /ehsan/m.sadegh/simlingo_lidar/stage2/logs/pilot-prepare.log 2>&1'
```

Source archives are SHA-verified and left intact. RGB, LAZ and gzip JSON payloads
remain compressed inside indexed uncompressed tar shards. Image pose augmentation
is excluded; the original photometric operations have explicit per-request seeds.
Annotation reads use a per-worker cache capped at 256 documents / 8 MiB of decoded
JSON. Geometry masks use contiguous arrays; workers do not inherit all parsed
annotations or thousands of small shared tensor allocations.
The preparer enforces the 100 GiB preparation cap and 150 GiB free-space reserve.
It selects official training partitions and Town13 validation, groups repetitions,
screens recorded trajectories against the local 220-route XML, and excludes a
whole source group on an identified overlap. The manifest records the screen's
thresholds and limits; this does not prove that scenarios are independent or undo
the released checkpoint's prior exposure.

After preparation, every eligible scan is decoded, voxelized and checked against
vehicle boxes to persist supported geometry labels and task eligibility. Invalid
or absent objectives are masked, with exclusions recorded. `validation.json`
contains fixed route-balanced requests and monitoring subsets for all task
families; the new reader loads validation language annotations explicitly.
`ready.json` is published last and binds the manifest, protocol and audit hashes.
The full CPU audit uses four processes and saves a manifest-bound JSONL recovery
cache. It can resume with `python -m simlingo_training.lidar.validation MANIFEST`
through the same launcher. Completed sensor shards can be verified and reused
with `--resume-sensors` when resuming the label-preparation command.

The engine uses loss counts across the full accumulation window and performs one
clip/optimizer/scheduler step per window. A failed window aborts; it must never
save the advanced sampler cursor. Resume from the last completed boundary. Use
`checkpoint_metadata` to bind checkpoints to code, recipe, manifest, validation,
environment and experiment settings. Resume requires the same stage and metadata;
joint training starts with aligned weights and a fresh optimizer/scheduler.

Stage 3 still needs the full task mixture, 100-token generation, accumulation 64,
optimizer state, validation, checkpoint I/O, memory and time measurements. The
small checks here do not replace that profile or authorize longer training.
