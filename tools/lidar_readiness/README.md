# Checks before Stage 4

Use `bash tools/lidar_stage1/run.sh ...` from the repository root. It isolates
Python and exposes only physical GPU 2. No command below launches main training.
Artifacts live under `/ehsan/m.sadegh/simlingo_lidar/readiness`.

`prepare_parking.py` creates `stage2/pilot-parking-v2`, preserving the original
pilot and all archives. It selects 32 training and 16 validation source groups
using a fixed SHA ordering, retains their repetitions, recognizes the official
parking collector layout, checks source/shard hashes, and audits every newly
eligible RGB/LAZ/box/future window. Existing audited rows and shards are reused.
All parent rows must remain; the same benchmark containment exclusion applies.
Town12 training and Town13 validation remain separated. The 100 GiB preparation
cap and 150 GiB shared free-space reserve still apply.

After its `ready.json` is published:

```bash
bash tools/lidar_stage1/run.sh tools/lidar_readiness/check_integrity.py \
  --manifest /ehsan/m.sadegh/simlingo_lidar/stage2/pilot-parking-v2/manifest.json \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness/integrity.json
bash tools/lidar_stage1/run.sh tools/lidar_readiness/audit_contexts.py \
  --manifest /ehsan/m.sadegh/simlingo_lidar/stage2/pilot-parking-v2/manifest.json \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness/context-audit.json
bash tools/lidar_stage1/run.sh tools/lidar_readiness/check_metadata.py \
  --manifest /ehsan/m.sadegh/simlingo_lidar/stage2/pilot-parking-v2/manifest.json \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness/metadata.json
bash tools/lidar_stage1/run.sh tools/lidar_readiness/full_baseline.py \
  --manifest /ehsan/m.sadegh/simlingo_lidar/stage2/pilot-parking-v2/manifest.json \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness/baseline
```

Run long checks in tmux. The baseline saves manifest/code/policy-bound generated
records in a durable journal and resumes completed requests after logout/reboot.
It requires all 7,680 applicable fixed requests. The full paired protocol contains
8,192 requests; geometry QA has no RGB-only counterpart. Narrow structured-target
coverage must include at least 100 samples, ten route groups and 10% of the
corresponding family. Unknown targets are reported, not counted as correct.

Run fresh-process checks **sequentially**, never concurrently on GPU 2:

```bash
bash tools/lidar_stage1/run.sh tools/lidar_readiness/fresh_resume.py \
  --manifest /ehsan/m.sadegh/simlingo_lidar/stage2/pilot-parking-v2/manifest.json \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness/resume-align \
  --stage align --phase reference
bash tools/lidar_stage1/run.sh tools/lidar_readiness/fresh_resume.py \
  --manifest /ehsan/m.sadegh/simlingo_lidar/stage2/pilot-parking-v2/manifest.json \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness/resume-align \
  --stage align --phase resume
```

Repeat both commands with `joint` instead of `align`. For the new one-epoch
proposal, pass `--total-updates 1161` to both invocations (the older ceiling is
3750). Add
`--context-audit /ehsan/m.sadegh/simlingo_lidar/readiness/context-audit.json`
to the joint resume invocation to include the final resource stress check.

Each stage executes two unique accumulation-64 windows and repeats the second
in a new interpreter/model/optimizer process. Every model/Adam tensor, scheduler,
sampler, augmentation, sensor/target fingerprint, generated token and RNG state
must match exactly. This exercises `train.create_session` and `TrainingSession`,
the production construction/update/checkpoint path. It does not reboot the shared
server. Temporary full checkpoints are removed only after successful verification;
diagnostic weights are never selected for training. The combined maximum-cloud,
maximum-teacher and maximum-question resource case is synthetic and has no
learning interpretation.

For a disposable alignment-to-joint handoff, pass `--handoff PATH` to the
successful alignment resume, then `--initial-weights PATH` to both joint
invocations. This verifies new joint optimizer construction from aligned weights;
the handoff is diagnostic and must be deleted after the joint check succeeds.

The first fresh-process check exposed spconv's process-local kernel tuning state
and lazy imgaug initialization. The corrected policy uses the same sparse kernels
through spconv's Python dispatcher, FP32/split-K=1, primes forward/backward choices
before checkpoint zero, and saves/restores the tuning tables. Priming restores
buffers, RNG and gradients and performs no optimizer update. imgaug's global
generator is initialized before training seeds are set. Numerical-policy schema 2
and source hashes reject checkpoints from the earlier policy.

`simlingo_training.lidar.train` is the actual entry point for the later reviewed
run. Its configuration requires an explicitly reviewed budget, completed readiness
report and the matching full baseline. It uses the same tested session, commits
the sampler only after complete optimizer windows, writes atomic last/recovery/
best checkpoints, resumes incomplete validation at a saved boundary, enforces full
RGB retention before selecting paired driving performance, and ends without
launching the next stage. Joint initialization requires the selected alignment
checkpoint hash and a new optimizer/schedule. Broader training and live-agent
evaluation remain separate reviewed stages.

Proposed configurations remain `budget_reviewed: false` until the user reviews the
actual duration. Notify the user before any main training on GPU 2. tmux survives
logout; atomic completed-boundary checkpoints provide reboot recovery.
