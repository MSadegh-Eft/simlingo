# Agreed LiDAR training plan

Implement in stages on `feat/simlingo-lidar`, based on clean commit
`4b143b74b235293d5b28737bcc496c3ec687fc92`. Each stage ends with a report and user
review. Training duration is decided after the complete pipeline is profiled.
Only physical GPU 2 is authorized. Benchmark evaluation and merging are deferred.

Execution records: [Stage 1](../tools/lidar_stage1/stage1-results.json),
[Stage 2](lidar-stage2-report.md), and the corrected single-GPU
[Stage 3 profile](lidar-stage3-report.md) are verified. Stage 3 found and corrected
reference scoring and backward repeatability issues before completing its full
profile. No main training has started. Completed
[pre-training readiness checks](lidar-readiness-report.md) add audited
parking coverage in `stage2/pilot-parking-v2` (74,285 training / 13,434 validation
frames), verify exact fresh-process recovery in both stages and establish the full
7,680-request released RGB baseline. All 37 tests pass. Review the actual budget
before Stage 4: the proposed 750-update alignment is about 30.7 hours including
validation/checkpoints, with a 24.5–53.7-hour planning band under shared load.
LLM KV caching remains disabled.
Notify the user before any main training on GPU 2; the provisional budgets below
do not authorize an automatic launch.

## Fixed model choices

Use pretrained **Utonia Base**. No encoder/freezing/fine-tuning comparison grid is
requested. Preserve the released SimLingo checkpoint, including its trained
image encoder/projector, LoRA, driving heads and navigation embeddings. Do not
restart from the original Qwen weights or replace the trained adapters.

| Component | Alignment | Joint training |
|---|---|---|
| RGB encoder/projector | Frozen | Trainable at low LR |
| Base LLM | Frozen | Frozen |
| Existing trained LoRA | Frozen | Trainable |
| Driving heads/navigation embeddings | Frozen | Trainable |
| Utonia Base | Frozen | Last two encoder stages trainable |
| New connector/position embeddings/geometry head | Trainable | Trainable |

Read Utonia architecture from its saved checkpoint configuration. Pin source
`de1f9deef4ee65e508d74347f8d0eb848d878abb` and weights revision
`9457f025e36c7001ee962a542f0d27a5b600e1e7`. Stage 1 verifies source and weight hashes.

Project first/middle/final encoder features with their metric positions and level
embeddings to width 512. Use two spatial-resampler blocks, 8 heads and FFN width
2048, with 128 queries (32 per front/right/rear/left 90-degree sector). Sector
cross-attention and query interaction produce tokens projected to Qwen width 896.
Handle empty sectors without fully masked attention. Prefix both language and
driving paths; omit the prefix completely for RGB-only replay.

Retain 10 future speed waypoints at 0.25 s intervals and 20 route points.

## Data, supervision and correctness

Keep archives intact. Prepare indexed pilot shards with up to 100k eligible
training frames (minimum 50k), capped at 100 GiB initially. Maintain 150 GiB shared
free space. Group repeated route geometry/weather variants into fixed manifests;
use official training partitions and Town13 validation. Apply splits to all task
families. Exclude identified benchmark-route overlaps from new training; exposure
of the released checkpoint cannot be undone.

Use packed XYZ, sample offsets, timestamps/history offsets, validity, task identity
and loss masks. Keep RGB and LiDAR history independent. Original saved XYZ is
already ego-relative; read LAZ scaled coordinates and do not apply the mount twice.
Intensity is zero. Keep the initial-frame exclusion, complete future targets and
route boundaries. Saved cadence is 4 Hz, inferred from the collector.

Use coordinate scale 0.2 / grid 0.01 (5 cm physical voxels), preserve the ego origin
and separate metre positions, and provide zero color/normal channels. Validation
voxel representatives, serialization and pooling order are deterministic. Freeze
parameters and module modes explicitly. Disable geometric/camera-pose augmentation
until it can transform all modalities/targets consistently; retain compatible RGB
photometric augmentation.

Keep the complete surrounding point cloud. Vehicle annotations cover 64 m, while
the sensor range is 85 m. Geometry labels cover only the supported region, using
horizontal distance bins [0,10), [10,20), [20,40), [40,64). Recount support after
voxelization; require >=3 retained points for positives and mask ambiguous labels.
"No observed vehicle" does not mean a lane is clear. No single-sweep velocity,
color, traffic-light appearance or flashing-light targets.

Original VQA remains scoped to the front view; audit commentary and Dreamer safety
claims for unsupported global assertions. Boxes, future states and answers are
supervision only. Driving must not consume ground-truth explanatory answers.

Fix zero-count NaN gradients and normalize each loss component over the whole
accumulation window. Avoid framework double scaling. Verify that language and
driving gradients reach the connector through frozen modules. Check each trainable
parameter occurs exactly once in optimizer groups.

## Training defaults and budget review

Stage A freezes original components and Utonia while training the new connector,
positions and geometry head. Preserve autograd through the frozen LLM. Proposed
budget: 48k presentations / 750 optimizer updates.

Stage B uses the trainability table above and a fresh optimizer/scheduler from
aligned weights. Review one sampled epoch first: 74,304 presentations / 1,161
updates for the expanded pilot. The older 240k / 3,750-update proposal remains a
ceiling, not an automatic initial budget. Decide this after the alignment report.

| Task family | Stage B share |
|---|---:|
| Dreamer/instruction following | 40% |
| Ordinary driving | 13 1/3% |
| Original VQA | 13 1/3% |
| Commentary | 13 1/3% |
| Additional rear/side driving | 10% |
| New observed-geometry QA | 10% |

Within each original family, 25% uses RGB-only replay (20% overall). Disable
LiDAR-specific losses on replay. Stage A omits replay and renormalizes paired
tasks. Explicit task sampling replaces adaptive fallback probabilities and bucket
size bias; log realized proportions and missing-label exclusions.

Use teacher-forced CE for language targets, but generated text for associated
commentary/Dreamer driving. Generate without gradients, then recompute the sensor
and driving pass with gradients and the same augmentation. No detached generation
cache in the learning pass. Greedy generation cap is 100 tokens; enable KV caching
only after parity checks.

Seed 42; microbatch 1 / accumulation 64. Microbatch 2 / accumulation 32 only after
validation. AdamW: weight decay 0.1, beta2 .999, epsilon 1e-8, global-norm clipping
0.3. OneCycle cosine: 5% warmup, div_factor 25, final_div_factor 10000, beta1 cycling
.95 -> .85 -> .95. Geometry loss starts at weight .2 alongside existing losses.

| Parameter group | Maximum LR |
|---|---:|
| Connector/geometry in alignment | 2e-4 |
| Connector/geometry in joint stage | 1e-4 |
| Upper two Utonia stages | 1e-5 |
| Existing LoRA | 1e-5 |
| Driving/navigation components | 1e-5 |
| RGB encoder and projector | 1e-6 |

Use BF16 mixed precision with explicit FP32 sparse boundaries and FP32 optimizer
state. Avoid global BF16 module construction. Use activation checkpointing where
needed. Target training-process peak below 40 GiB and bound CPU worker/prefetch RAM.
Never silently discard difficult/OOM samples.

Create actual training models with
`simlingo_training.lidar.numerics.make_training_model()`. Stage 3's full recovery
check required deterministic FlashAttention backward in Qwen, vision and Utonia,
strict PyTorch/CUBLAS settings, FP32 cumulative waypoint arithmetic, and native-grid
vision position identity instead of redundant bicubic interpolation. This keeps
pretrained parameters and architecture intact; other patch grids fail until their
resize policy is reviewed. Bind this policy into checkpoint metadata. The report
includes its measured throughput cost and the initial failed checks.

Fresh-process verification additionally requires numerical-policy schema 2:
checkpoint spconv's sparse kernel tuning tables through its Python dispatcher,
retain FP32 and split-K=1, and prime the stage's forward/backward kernels before
the initial checkpoint. Priming restores buffers/RNG/gradients and performs no
optimizer update. Initialize imgaug's lazy global generator before setting the
training seeds. These changes address an actual fresh-process failure that the
same-process profile did not expose. This kernel state is separate from LLM KV
caching, which remains disabled.

Before any long run, profile the full task mixture, generation, forward/backward,
I/O, validation and checkpoint saving. Report estimated hours for alignment, pilot
and a larger-data epoch; agree the actual budget with the user. 288k total
presentations is provisional, not permission to automatically launch the run.

## Validation, stages and recovery

Persist route-balanced validation: up to 4096 driving frames, 2048 original QA,
512 commentary and 512 Dreamer examples, plus supported geometry labels. Use the
dedicated language loader and fixed questions, not the ordinary loader that skips
validation language labels. Use smaller fixed monitoring subsets; full validation
at stage completion and fixed joint-training intervals.

Select among retention-passing checkpoints by route-balanced generated-condition
speed/route ADE, normalized by the corrected released baseline. Report target
scenarios and rear geometry separately. RGB-only retention defaults: <=2% relative
degradation for each driving error, <=2 percentage points for structured QA and
Dreamer decisions, <=2 points for fixed commentary ROUGE-L; retain generated
examples for inspection. These are engineering training gates, not safety claims.

1. Branch/environment/baseline verification/small-archive audit: Stage 1.
2. Manifests, preprocessing, connector, masks, correctness fixes, tiny-data learning.
3. Complete GPU-2 memory/throughput profile and time-budget review.
4. Agreed alignment run and report.
5. Agreed joint pilot and report.
6. Broader training only after review: at most three sampled epochs initially,
   where one epoch is ceil(eligible_frames/64)*64 presentations. Fresh continuation
   schedule from the selected pilot weights; validate twice per epoch and stop
   after three consecutive full validations without improvement.

Test coordinates/timing/visibility, split leakage, zero-label losses, accumulation,
prefix/masks/positions/output slicing, separate gradient paths, frozen weights,
optimizer groups, RGB-only mode, strict reload/resume and generation caching.

Use tmux, atomic checkpoints at optimizer boundaries every 100 updates or about
30 minutes, an initial checkpoint at update zero, and bounded last/best/one-recovery
retention. Include optimizer, scheduler, RNG, sampler position, augmentation seeds,
sparse kernel choices and config hashes. Test resume;
tmux alone does not survive server reboot.

Stage 3 verified all loaded model/Adam tensors and the complete next update in the
same GPU process. Before a long run, the actual training entry point still needs
a fresh-process resume smoke check and the complete released RGB validation
baseline; Stage 3's small validation timing subsets do not establish retention.

Temporal LiDAR is a separate reviewed extension: -0.5/-0.25/0 s histories, shared
encoder, timestamps and 128 output tokens. Saved half-sweeps use privileged pose
alignment and lack the raw sensor stream needed to reproduce live odometry.
Require sensor-recording calibration and a documented discrepancy policy before
temporal training; do not assume exact parity. Runtime sensor faults, anti-stuck
handling, CARLA/model combined memory, GPU pinning of evaluation helpers and final
benchmark protocol are deferred to the later deployment/evaluation plan.
