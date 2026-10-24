# Bounded Stage 3 profile

Use the Stage 1 isolated launcher. It exposes only physical GPU 2 and avoids the
login environment's incompatible CARLA egg. The profile checks the physical UUID,
branch, Stage 2 readiness/source hashes, RAM/free space and CUDA reserve. It has
a 45-minute alarm and finite update counts; it has no main-training mode.

```bash
bash tools/lidar_stage1/run.sh -m unittest discover -s tests -v
bash tools/lidar_stage1/run.sh tools/lidar_stage3/profile_pipeline.py --help
tmux new-session -d -s simlingo-lidar-stage3 \
  'cd /data/ghazaleh/simlingo && bash tools/lidar_stage1/run.sh \
  tools/lidar_stage3/profile_pipeline.py \
  --output /ehsan/m.sadegh/simlingo_lidar/stage3/CHOOSE_FRESH_RUN_DIRECTORY \
  > /ehsan/m.sadegh/simlingo_lidar/stage3/CHOOSE_FRESH_RUN_DIRECTORY/profile.log 2>&1'
```

Create the fresh output directory before launching the command. Existing
`progress.json` prevents accidental reuse. Inspect `progress.json`, `profile.log`
and the finite per-presentation JSONL records. A failed profile stops without
dropping/reassigning a sample. Temporary learned checkpoints are deleted by the
exit handler; an abrupt server/power failure can leave disposable `.pt.part`
files, which must not be selected as trained weights. tmux survives SSH logout,
not server reboot. No pretrained weights or source archives are overwritten.

This executes one alignment update and four joint accumulation-64 updates,
including the repeated next-update recovery check. The Adam/OneCycle settings,
task requests, photometric augmentation, uncached 100-token generation cap and
learning pass all use the real Stage 2 pipeline. Per-presentation CUDA-event
component intervals are nested inside forward wall times; do not add them twice.
The small joint sample is reweighted to the exact agreed task/replay proportions
when estimating the longer budget. It is not a selected alignment/joint run.

Recovery checks every loaded model and Adam-state tensor for exact equality to
the saved boundary. The repeated next update is compared tensor by tensor with
fixed model atol=1e-6/rtol=1e-5 and Adam atol=1e-8/rtol=1e-3. Both bitwise and
numerical results are recorded. Numerical tolerances are never raised on failure;
CUDA next-update bitwise reproducibility is not assumed. Scheduler, sampler and
augmented samples must match exactly.

Actual training must create the model with
`simlingo_training.lidar.numerics.make_training_model()`. This installs
deterministic FlashAttention backward in the pinned Qwen, image and Utonia
implementations, strict PyTorch/CUBLAS settings, FP32 cumulative waypoint outputs,
and a native-grid identity for vision position embeddings. No pretrained
parameters are changed. A different vision patch grid raises an error until its
interpolation policy is reviewed. Bind `model.lidar_numerics_policy` into
`checkpoint_metadata` experiment settings, as this profiler does.

The initial profiles failed next-update recovery checks. The focused
`diagnose_recovery.py` probe compares repeated forward values, generated IDs,
RNG states and all clipped gradients before/after the numerical corrections; it
performs no optimizer updates and has a ten-minute limit. Its final artifact is
`/ehsan/m.sadegh/simlingo_lidar/stage3/recovery-diagnostic-final.json`.

Four fixed requests per family are evaluated in paired and eligible RGB modes
before any update. Generated answers drive the commentary/Dreamer predictions;
references are scoring targets only. Metrics are route-group balanced. Unknown
structured-QA targets are excluded with coverage, while unparseable predictions
on known targets count as wrong. Missing coverage cannot pass retention gates.
The complete released RGB baseline and complete fixed validation are still needed
for actual checkpoint selection; small profiling subsets never establish retention.

After a **complete** profile, produce the budget report:

```bash
bash tools/lidar_stage1/run.sh tools/lidar_stage3/rescore.py \
  /ehsan/m.sadegh/simlingo_lidar/stage3/CHOOSE_FRESH_RUN_DIRECTORY
bash tools/lidar_stage1/run.sh tools/lidar_stage3/summarize.py \
  /ehsan/m.sadegh/simlingo_lidar/stage3/CHOOSE_FRESH_RUN_DIRECTORY
```

The rescoring command normalizes decoded chat role headers while retaining the
original strings and generations. It writes separate corrected artifacts and
binds the original/final scoring source hashes. It performs no model inference.

The report includes full-mixture rates, exact recipe/resource settings, fixed
validation extrapolation, actual full-state checkpoint save/load and next-update
recovery, worker I/O/RAM, maximum-cloud/long-context/100-token stress, and
48k/240k/epoch budgets. The planning range is an explicit heuristic for shared GPU
and SMB variability. The original dataset's full eligible frame count is unknown;
the larger 100k epoch is a conditional scenario. Parking-lane coverage is still a
review item. Main training requires a separate budget review and user notification
before launch on GPU 2; neither script starts it.
