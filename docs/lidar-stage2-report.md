# Stage 2 verification

Stage 1 passed its repeated pinned-source/environment, GPU-2 kernel and released
checkpoint checks. The corrected RGB components still match the saved baseline
within 1e-4. Work remains on `feat/simlingo-lidar`; Stage 1 is commit `93d2080`.

Stage 2 is complete. The pretrained Utonia Base encoder supplies first/middle/final
features and metric positions to the two-block, width-512 spatial connector. Its
128 sector queries produce width-896 tokens for both language and driving. The
released RGB/projector, trained LoRA, navigation and driving weights are preserved.
There are 9,641,861 trainable parameters in alignment
and 467,028,821 in joint training. The last two Utonia stages account for
129,855,312 of these.

The pilot contains **61,962 training frames** across
731 source groups and **6,748 Town13
validation frames** across 43 groups. Prepared shards occupy
24.86 GiB including the small correctness dataset. Archives remain
intact. Every eligible RGB was decoded; LAZ scans were voxelized; boxes, navigation
and complete future measurements were checked. Geometry QA eligibility uses
retained-point support and ambiguous-label masks. Annotation memory is bounded,
and geometry masks use contiguous arrays for worker spawning.

The manifest excludes 1 source route group(s) identified
by the recorded benchmark containment screen. This screen is conservative and
does not establish independent scenario design or undo prior checkpoint exposure.
Its thresholds and matches are recorded in the audit. Fixed route-balanced
validation requests are persisted for driving, original QA, commentary, Dreamer,
rear/side driving and observed-geometry QA. Counts: `{"commentary": 512, "dreamer": 512, "driving": 4096, "geometry_qa": 512, "qa": 2048, "rear_driving": 512}`.
The pilot has no `parkinglane` bucket; its missing-bucket exclusions and weight
renormalization are explicit. Parking coverage needs review before longer training.

Verification passed:

- 15 contract tests: coordinates/sectors, visibility/range masks,
  empty sectors, zero-count gradients, accumulation normalization, task/replay
  quotas, bucket weighting, source groups, future boundaries, indexed payloads,
  photometric seeds, legacy inference masks and checkpoint recovery.
- Real samples match original waypoint/route construction. Ten presentations
  covering all tasks and RGB replay match with zero and two spawned loader workers.
- Combined loading is strict outside the new LiDAR modules. Prefix placement,
  no-target-leakage checks, direct-backbone parity and repeated uncached generation
  pass on the fixtures. A pilot geometry-QA example failed cached/uncached token parity, so KV
  caching remains disabled. The final suite uses uncached generation throughout.
- Language and driving gradients separately reach the connector through frozen
  modules. All intended joint groups receive finite nonzero gradients. A separate
  joint optimizer update changes each group, preserves frozen weights and keeps
  Adam moments FP32.
- The final disposable tiny alignment check ran 16 updates on
  32 presentations; loss changed from 7.390
  to 1.419. Frozen parameters remained byte-identical. Full model,
  optimizer, sampler and CUDA random state recovered from an atomic checkpoint.

Development checks found and corrected an unsupported cuDNN attention path,
non-recursive imgaug seeding, unbounded annotation retention, the legacy RGB
projector freeze, zero-count loss division, inference padding/commentary-off
handling, agent evaluation mode and the overwritten overfit setting.

These are bounded correctness checks. They do not show improved driving, a
retention-passing checkpoint, full-mixture memory/throughput, or live-agent sensor
parity. Tiny-check weights were discarded. Longer training, Stage 3 profiling,
benchmark evaluation, temporal LiDAR and merging have not started.

Detailed artifacts: `/ehsan/m.sadegh/simlingo_lidar/stage2/pilot/{manifest,validation,audit,ready}.json` and
`/ehsan/m.sadegh/simlingo_lidar/stage2/reports/`. Reproduction commands are in
[`tools/lidar_stage2/README.md`](../tools/lidar_stage2/README.md). The machine-readable
summary is [`stage2-results.json`](../tools/lidar_stage2/stage2-results.json).
