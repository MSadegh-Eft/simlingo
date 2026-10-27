# Monitored training: settings, operation and readiness

Updated 27 October 2026. Work is on `feat/simlingo-lidar`.

**Main alignment and joint training have not started.** Both new configurations
require budget review and explicit confirmation of supervisor permission. The
previous readiness report describes the older runner. Changed code needs new
GPU evidence before this runner can be declared ready.

CPU verification completed: **56 tests passed**, all **47 prepared shards** were
rehashed, pinned pretrained assets matched, and both closed launch guards were
tested without creating stage outputs. The new machine-readable record is
`/ehsan/m.sadegh/simlingo_lidar/readiness-monitored-v1/result.json`; its status is
`verification_pending`, with no failed checks. GPU 2 was still occupied when this
record was prepared.

## Initial settings and their meaning

| Setting | Alignment | Joint fine-tuning |
|---|---:|---:|
| Eligible prepared training frames | 74,285 | 74,285 |
| Maximum presentations | 48,000 | 74,304 |
| Maximum optimizer updates | 750 | 1,161 |
| Sampled epochs at ceiling | 0.646 | 1.0003 |
| Microbatch / accumulation | 1 / 64 | 1 / 64 |
| RGB-only replay | 0 | Approximately 20% |
| Planned unique frames, seed 42, if ceiling reached | 33,996 | 44,627 |
| Minimum updates before plateau stopping | 500 | 581 |

The 13,434 Town13 validation frames stay outside training. Task/scenario sampling
permits repetition: a sampled epoch does not visit every frame once. The complete
CPU sampler audit reports every task's reachable pool, realized task quotas,
replay counts and scenario exposure. All eligible task pools remain reachable.
Different scenario bucket counts overlap and must not be added as distinct frames.

These are bounded pilot budgets. Neither 48k presentations nor one epoch has been
shown sufficient for satisfactory results. Use the audited existing pool to establish
learning and retention before paying to expand preparation. The older 2.25% number
was relative to the published total dataset count, not an audited training-only
denominator. The 48k choice is a heuristic presentation budget, not a published
SimLingo LiDAR recipe or a convergence result.

The released model's learned image pipeline, trained LoRA and driving components
are retained. Alignment trains only the connector/positions/geometry head; joint
training also opens the upper two LiDAR stages, original LoRA, image pipeline and
driving/navigation components as specified in the original staged plan. Keep the
tested optimizer, numerical policy and disabled LLM KV caching. Maximum learning
rates are 2e-4 for alignment; joint uses 1e-4 connector, 1e-5 LiDAR/LoRA/driving/
navigation and 1e-6 RGB. No learning rates, sampling weights or losses change
automatically inside a run.

## Monitoring and decisions

The immutable monitoring protocol contains 512 paired and 448 RGB requests,
including 100 scored structured QA targets and 100 Dreamer targets, each spanning
59 groups. An additional 192 training diagnostic requests (32 per family) use no
augmentation. Additions are selected by annotation eligibility and group coverage,
without consulting prediction correctness. Full validation requests are unchanged.

Monitor at update zero, every 125 updates (8k presentations) and the stopping
boundary. Each request journals generated answers, reference answers, generated-
condition driving metrics, teacher CE and geometry diagnostics. Teacher answers
never condition driving. Reuse the generated driving predictions for loss
diagnostics rather than generating them again. Preserve module modes, all training
RNGs and sparse kernel choices around evaluation.

The stopping score averages route and speed ADE normalized by the corresponding
released RGB monitor populations. Meaningful improvement is 0.5% in that score,
or 1% in token-normalized language CE or the mean presence/distance geometry loss.
Any of these resets patience. Three successive checks without improvement can stop
training only after the minimum updates above and when every active group's
learning rate has descended below 25% of its configured maximum. The configured
ceiling always stops the run. Missing or invalid metrics fail the check instead
of advancing patience. Thresholds are engineering defaults; they are not proof of
globally optimal convergence.

Monitor candidates prioritize provisional RGB retention and then normalized driving
error; update zero is ineligible. Small-subset retention is a screen. At completion,
run the unchanged full offline protocol (8,192 paired / 7,680 RGB requests) on the
selected candidate. Only if it fails retention, evaluate the final checkpoint once
as a fallback, when different. At most two full candidate evaluations occur.
Keep the full retention limits: 2% relative degradation per RGB driving metric,
two percentage points structured QA/Dreamer and two ROUGE-L points. Failure leaves
no selected checkpoint and requires review. A passing retention result alone does
not prove LiDAR helped driving.

The report includes route-balanced metrics, task/scenario/sector support, generated
examples, curves and recommendations. Training/validation language-gap comparisons
use equal weights across the same task families to avoid confounding different
mixtures or answer lengths. Advice includes another 16k alignment presentations
or one joint epoch when still improving, more independent routes when overfitting,
an initial 100k-frame pool for coverage gaps, or review of learning rates/replay
when RGB retention worsens. Flat performance or little paired-versus-RGB benefit
triggers inspection of gradients, labels and modality use. Advice never edits
settings or starts another stage. Continuation needs a new reviewed configuration
and explicit schedule; changing an existing OneCycle horizon invalidates resume.

These are offline dataset evaluations. The full Bench2Drive run remains separate.

## Launch, follow and resume

New proposed configurations are under `tools/lidar_training/configs`. Older
schema-1 proposals are historical and are rejected by this runner. Keep both
`budget_reviewed` and `supervisor_permission_confirmed` false until the actual
budget and permission have been confirmed. The permission note must identify the
confirmation. For joint training, bind the selected aligned checkpoint hash.

The following are examples for **after** readiness and permission, using a separate
approved configuration. Running them with the current proposals is blocked.

```bash
bash tools/lidar_training/tmux.sh start --config /absolute/path/align-approved.json
tmux attach -t simlingo-lidar
```

Detach with Ctrl-b, then d. The pane remains after exit or failure. The launcher
does not queue jobs waiting for a GPU, grant permission, start a subsequent stage
or restart a failed run. Direct CLI launches and tmux launches share a process
lock. It does not lock out other lab users; available resources are checked again
before model construction. Only physical GPU 2 with the pinned UUID is accepted.

```bash
bash tools/lidar_training/tmux.sh status --config /absolute/path/align-approved.json
bash tools/lidar_training/tmux.sh resume --config /absolute/path/align-approved.json \
  --checkpoint /ehsan/m.sadegh/simlingo_lidar/stage4/align-monitored-v1/last.pt
```

If an interrupted atomic save left no `last.pt`, explicitly choose `recovery.pt`.
An older boundary preceding a committed monitoring decision is rejected; inspect
the latest boundary instead. The configuration and bound sources must stay
unchanged for resume. A completed run requires a reviewed continuation rather than
resuming it as though it had crashed.

The live terminal shows optimizer/accumulation/validation progress. Status includes
sampled epochs, unique frames, component losses, rates, gradient norms, GPU memory,
throughput, training-only ETA, next monitoring point, best candidate and patience.
Separate phase labels show loading, training, validation, saving and failure.
`training.log`, `events.jsonl` and atomic `status.json` persist outside tmux. Each
attempt has an ID; resume logs explicitly identify rolled-back updates. Checkpoint
counters, not summed log rows, are authoritative.

Checkpoints precede monitoring and are also saved periodically (15 alignment or
12 joint updates, or 30 minutes), at completed optimizer boundaries. Keep last,
one recovery, monitoring candidate and full-validated best, plus an atomic write.
Track sampler/optimizer/scheduler/RNG/sparse state, unique-frame counters and
controller history. Decisions and evaluation journals bind checkpoint hashes;
interrupted monitoring is completed before another optimizer update. tmux survives
SSH disconnection; checkpoints provide explicit recovery after a process/server
failure.

Reports: `report.md`, `report.html`, `report.json`, `learning-curves.svg`,
`learning-curves.csv`, per-check journals and `selection.json`. Completion is
`awaiting_review`, with an explicit stop reason and retention-passing selection
flag. No automatic stage progression occurs.

## Verification and remaining GPU work

Current artifacts live at
`/ehsan/m.sadegh/simlingo_lidar/readiness-monitored-v1`.
The runner requires this new report, current source hashes, verified artifact
hashes, a fresh full RGB baseline and confirmation that the fresh baseline agrees
with the fixed monitor references. Original baseline/recovery evidence remains
untouched. A mismatch requires a new monitoring protocol version; it is not
silently accepted.

CPU checks prepare the expanded protocol, audit the full planned sampler,
rehash 47 shards and pinned assets, and test stopping, transactional recovery,
logging, reports and harmless tmux attach/detach. To record their exact source
snapshot and publish readiness (which remains pending without GPU evidence):

```bash
bash tools/lidar_stage1/run.sh tools/lidar_readiness/verify_cpu.py \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness-monitored-v1
bash tools/lidar_stage1/run.sh tools/lidar_readiness/monitored_readiness.py \
  --output /ehsan/m.sadegh/simlingo_lidar/readiness-monitored-v1
```

**Notify the user before GPU verification.** Run the following sequentially only
when physical GPU 2 has at least 40 GiB free. Launch also requires 16 GiB available
host RAM and 200 GiB shared free disk (150 GiB reserve plus 50 GiB checkpoint
capacity). These are disposable checks, not main training:

1. Run `full_baseline.py` on the existing manifest, writing to the new evidence
   directory's `baseline` subdirectory. It performs zero optimizer updates.
2. Run `fresh_resume.py` reference and resume phases for alignment (750-update
   schedule) and joint (1,161-update schedule), writing to the new `resume-align`
   and `resume-joint` directories. Use its disposable alignment handoff for joint
   initialization, and pass the existing exhaustive context audit on joint resume
   to repeat the combined resource stress. Remove the diagnostic handoff after
   successful checks; never select diagnostic weights for training.
3. Run `probe_monitoring.py` separately for `align` and `joint`, passing the
   prepared manifest and `monitor-protocol.json`, writing to new `monitor-align`
   and `monitor-joint` directories. Each compares the exact next update with and
   without the full monitoring sequence, measures all three monitoring passes and
   atomic copy cost, and removes disposable weights only on success.
4. Republish `monitored_readiness.py`. It requires all new proofs, current code
   and matching baseline/monitor provenance before setting readiness true. Then
   review the refreshed runtime estimate. No approval flag is changed by this tool.

Historical training-only estimates were 25.2 h alignment and 47.8 h joint. The old
30.7/58.3 h totals do not include the new monitoring policy. New totals remain
pending measurement; the report will include monitor/checkpoint cost, one full
validation, a separately priced fallback and a heuristic 0.8–1.75x planning band.
Paired full-validation time is still extrapolated until measured.

**Training remains blocked until new verification, budget review and supervisor
permission are all complete.**
