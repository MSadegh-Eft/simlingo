# Pre-training readiness

**Historical report for the previous runner.** The [monitored training update](lidar-monitored-training.md)
adds stopping, diagnostic monitoring, transparency and permission guards. Its new
GPU proofs are pending; this report does not establish readiness of the changed
code. Old baseline/recovery artifacts are preserved without relabelling them.

Technical checks before Stage 4 are complete on `feat/simlingo-lidar`. The actual training
budget still needs user review. No alignment or main joint run was launched.
Only disposable bounded checks used optimizer updates. The proposals remain
`budget_reviewed: false`; the training CLI rejects an unreviewed configuration.

The versioned pilot contains **74,285 training frames** in
762 source groups and **13,434 Town13
validation frames** in 59 groups. All original pilot rows
remain. Parking adds Town12 training and Town13 validation source groups without
changing the old artifacts. The original grouping conservatively merges repeated
source IDs; 32 selected parking training groups add 31 previously unseen group IDs.
Parking bucket counts are 1,145 driving,
1,145 QA and
918 commentary. No requested
sampling bucket is missing. Benchmark containment still excludes the same whole
Town04 source group; this does not establish complete benchmark independence or
undo exposure of the released checkpoint.

Verification passed:

- All prepared shard hashes and pinned source relationships; pretrained release,
  Utonia weights/source/installed code, and Hugging Face snapshot assets.
- 37 tests, including failure before an optimizer boundary, update-zero
  checkpoint restoration, interrupted evaluation journals and budget guards.
- Independent uncached/cached metadata and complete sensor/target fingerprints on
  parking examples in both splits, covering every family and RGB replay.
- Exhaustive finite text-variant audit and runtime 2,048-token guards that reserve
  the 128-token prefix, up to 100 generated tokens and 30 driving queries.
- Full **7,680-request released RGB baseline** on the new immutable protocol,
  with generated text, route-group metrics, scenario buckets and parse coverage.
  No subset substitutes for the baseline. All 8,192 paired requests remain fixed
  for the later trained checkpoint validations.
- Fresh-process alignment and joint recovery using the production construction,
  update and boundary code: every model/Adam tensor, sampler, scheduler, input and
  target fingerprint, generated token, loss and Python/NumPy/Torch/CUDA RNG matches
  exactly. This simulates a restart without rebooting the shared server.
- Maximum training-cloud and all-split text stress with resident joint Adam state, retained teacher
  graph, exactly 100 uncached generated tokens and driving backward. This combined
  extreme is a resource test, not a training example or learning result.

The released RGB baseline, averaged within source groups and then across groups:

| Metric | Result | Scored requests |
|---|---:|---:|
| Route ADE | 0.06343 m | 5,632 |
| Speed-waypoint ADE | 0.77627 m | 5,632 |
| QA exact match | 82.09% | 2,048 |
| Structured QA accuracy | 92.35% | 485 |
| Dreamer decision accuracy | 97.36% | 501 |
| Commentary ROUGE-L | 80.75 / 100 | 512 |

Both driving metrics measure waypoint position errors in metres. Structured QA
covers 485/2,048 targets (23.68%), and Dreamer covers 501/512 (97.85%); both span
all 59 groups. The independent annotation-only audit matches these exact scoring
populations. Unknown targets are excluded explicitly, while an unparseable
prediction for a known target is wrong. Parking-maneuver results include 155
driving requests across 16 groups; its commentary/Dreamer subgroups are small and
are descriptive only. Exact counts and generated examples remain in the artifacts.
These are engineering gates, not evidence of driving safety or improved rear
awareness. Complete paired evaluation and retention of a trained candidate are
performed after alignment.

The accepted GPU checks peak at 30.74 GiB reserved,
below the 40 GiB process limit. Exhaustive text maxima are 643 teacher tokens and
596 question tokens; reserving prefix, generation and driving queries needs at
most 854 tokens against the 2,048-token limit. The synthetic backward stress uses
the largest training cloud (31,361 voxels), resident joint Adam state and 100
uncached generated tokens. Validation's largest cloud is 31,516 voxels and uses
no-grad evaluation; all eligible scans were audited on CPU.

| Proposed next run | Presentations / updates | Central estimate | Planning range |
|---|---:|---:|---:|
| Alignment | 48,000 / 750 | 30.7 h | 24.5–53.7 h |
| One sampled joint epoch | 74,304 / 1,161 | 58.3 h | 46.7–102.1 h |
| Older joint ceiling | 240,000 / 3,750 | 176.4 h | 141.1–308.7 h |
| Conditional later 100k-frame epoch | 100,032 / 1,563 | 75.5 h | 60.4–132.0 h |

These include initial/periodic checkpoints, full paired+RGB validations and fixed
monitors. The initial released baseline is already complete and is not charged
again. Training uses the Stage 3 full-mixture rates, conservatively adjusted with
new-pilot bounded windows; actual full RGB validation took 1.54 h.
Paired timing is still extrapolated. The range is a 0.8–1.75× planning band,
not a statistical interval or guaranteed completion time under shared load.
The 240k ceiling is 3.23 sampled epochs of this pilot.
The 100k-frame scenario requires preparing and auditing more eligible data first;
it is not the size of the entire original dataset, which remains unestablished.

Recommend reviewing the 750-update alignment run first, then its full report
before deciding the joint run. The concrete configurations are in
[`tools/lidar_readiness/configs`](../tools/lidar_readiness/configs/).
GPU 2 is shared; check current free memory before launch and notify the user.
KV caching stays disabled. Keep microbatch 1 / accumulation 64, preserve the
trained LoRA and image pipeline, and use the corrected numerical policy below.

The production CLI writes an initial checkpoint before any update, retains only
last/recovery/best plus an atomic write, and commits sampler position only after
successful optimizer windows. A failed window must reload its last completed
boundary. Resume binds data, code, pretrained assets, numerical policy, package
versions and the complete reviewed configuration. Validation journals bind their
exact checkpoint and request set. Joint initialization binds the selected aligned
weights and starts a fresh optimizer/schedule. tmux survives logout; checkpoints
and journals provide reboot recovery.

The first fresh-process attempt failed despite identical inputs and loaded
weights. Its logs and comparisons remain. spconv's process-local tuning choices
were not in the old checkpoint; lazy imgaug initialization also consumed NumPy
RNG only in the restarted process. The current policy exposes the same sparse
kernels through Python dispatch, primes stage-specific forward/backward kernels
before checkpoint zero, saves/restores their tuning tables, uses FP32 and
split-K=1, and initializes augmentation's global generator before training seeds
are set. Priming performs no optimizer update and restores buffers/RNG/gradients.
The accepted checks are fresh runs with this policy, without relaxed tolerances.

Earlier baseline startup/partial artifacts were superseded while finalizing the
training entry point; they remain for audit and are not used in these results.
All disposable recovery weights were removed after successful comparisons.
CARLA integration, sensor-fault behavior, temporal LiDAR, closed-loop benchmarking,
broader training and merging remain separate later stages.

Artifacts: `/ehsan/m.sadegh/simlingo_lidar/readiness`. Reproduction:
[`readiness README`](../tools/lidar_readiness/README.md). Exact metrics, coverage,
hashes, recovery comparisons, resource bounds and budget inputs:
[`readiness-results.json`](../tools/lidar_readiness/readiness-results.json).
