# Stage 3 single-GPU profile

Completed on `feat/simlingo-lidar`, using only physical GPU 2, an NVIDIA RTX 6000
Ada with 46,068 MiB reported memory. Stage 2's hashes and saved checks were rechecked;
the 15 original tests plus eleven evaluation/numerics/recovery tests pass. No main training, benchmark
evaluation or merge was started. Profiling weights and temporary checkpoints were
discarded. The run was bounded to 45 minutes in tmux.

One alignment update processed 64 paired presentations. Joint profiling executed
four accumulation-64 updates: 192 unique presentations and a repeated 64-sample
next-update recovery check. All six task families and original-family RGB replay
were measured with 100-token greedy caps and KV caching disabled. Joint rates
below are reweighted to the agreed 960-presentation task/replay cycle; the shorter
profile itself is not an exact 960-sample cycle.

| Measurement | Result |
|---|---:|
| Peak allocated GPU memory, including stress | 29.02 GiB |
| Peak reserved GPU memory | 29.63 GiB |
| Alignment, including measured I/O | 1.823 s/presentation |
| Joint, weighted mixture including I/O/step | 2.234 s/presentation |
| Full model/Adam checkpoint size | 7.59 GiB |
| Atomic checkpoint save / load | 47.9 / 18.6 s |

The full training/validation metadata took
7.2 minutes
to read from shared storage. Loader measurements cover the same 64 requests:

| Spawn workers / prefetch | Wall time including startup | Median read wait | Summed process RSS peak |
|---|---:|---:|---:|
| 0 | 8.4 s | 0.081 s | 5.6 GiB |
| 2 / 1 per worker | 19.6 s | 0.037 s | 8.8 GiB |

RGB/prompt tensors matched across these requests. The full-update timings use
unoverlapped CPU reads with zero workers; benefits from two workers are not
credited in the budget. Host memory available during monitoring stayed above
26.0 GiB, including the
temporary full-state recovery comparisons. Keep microbatch 1 / accumulation 64;
microbatch 2 has not been validated.

The profile used the original released RGB/projector, trained LoRA, navigation and
driving weights. The base LLM remained frozen. Alignment preserved every frozen
parameter. Joint Adam moments stayed FP32. Reloading a completed joint checkpoint
restored every model and Adam-state tensor exactly and reproduced all augmented
samples, scheduler and sampler. The full next accumulation-64 update passed the
predeclared numerical comparisons: model maximum absolute difference
0; Adam
maximum absolute difference
0.
Tolerances were model atol=1e-6/rtol=1e-5 and Adam atol=1e-8/rtol=1e-3, checked
against every tensor. Bitwise next-update equality was
`True` for model tensors and
`True` for Adam tensors.
This was a same-process reload. A fresh-process/reboot trajectory comparison was
not performed; the actual training entry point needs its own restart smoke check
before a long run.
Two earlier profiles stopped at failed recovery assertions; logs remain in
`run-20261024a` and `run-20261024b`. The second failed the declared Adam tolerance;
it was not accepted. A focused diagnostic found equal inputs, forward losses,
generated text and RNG states but differing backward gradients. The final policy
enables deterministic FlashAttention backward in Qwen, vision and Utonia, uses an
FP32 triangular scan for waypoint accumulation, and skips vision position
interpolation when the pretrained/input grids are identical. Other vision grid
sizes fail explicitly. The paired-gradient diagnostic then matched bit for bit,
and this fresh full run measures that corrected policy. Pretrained parameters
and architecture are unchanged; cumulative waypoint arithmetic now stays FP32.
Source/config/data hashes, numerical policy and RNG states are bound to
checkpoints. Checkpoint I/O was measured on `/ehsan`.

The first two corrected joint updates averaged 133.4 seconds versus
106.2 seconds earlier (about 26% slower).
This is an observed comparison under shared load, not a controlled ablation. The
budget below uses the corrected pipeline exclusively.

| Proposed budget (not launched) | Training only | Including proposed checks/I/O | Planning range, total |
|---|---:|---:|---:|
| Alignment: 48k / 750 updates | 24.3 h | 29.9 h | 23.9–52.4 h |
| Joint pilot: 240k / 3750 updates | 148.9 h | 171.4 h | 137.1–300.0 h |
| One sampled pilot epoch: 62,016 presentations | 38.5 h | 48.8 h | 39.0–85.4 h |
| Larger epoch scenario: 100,032 presentations | 62.1 h | 73.3 h | 58.7–128.3 h |

The total estimates include dataset/model startup, checkpoints at the earlier of
100 updates or about 30 minutes, and explicit validation counts. The proposal costs one final
full paired+RGB validation and two monitors for alignment; four full checks and
four monitors for the joint pilot. Full validation was extrapolated from four
fixed requests per task in each applicable mode (44
examples total), not executed over all 8,192 paired and 7,680 RGB requests.
Estimated full paired validation is 169.9 minutes;
full RGB validation is 87.0 minutes.
A complete released RGB baseline adds another 87.0
minutes once, before alignment. Generation lengths can change after training.
The 240k joint ceiling corresponds to approximately 3.9 sampled
epochs of this pilot; it remains a budget to review, not a required initial run.
The 0.8–1.75× planning band is a heuristic; another process was using GPU 2 and
SMB throughput varies. It is not a statistical confidence interval.

The evaluator computes generated-condition speed/route ADE, word-token ROUGE-L,
conservatively parsed QA/Dreamer decisions with explicit coverage, and per-sector
geometry accuracy. It averages within source route groups before averaging groups,
stores generated examples, and requires identical metric populations for retention
comparisons. The released RGB measurements here are a bounded baseline fixture;
they cannot establish retention or improved driving. Complete fixed validation
and sufficient parseable-target coverage remain required for checkpoint selection.
Real-sample checks exposed the decoded `assistant` role header in references.
Reference normalization was corrected before the final profile. Its saved
generations were re-scored as a CPU-only verification; the metrics were unchanged.
Original timing/scoring artifacts are preserved. `scoring-correction.json` records
the scoring-source hashes and outputs; earlier incorrect scoring from
`run-20261024a` is not used in this report. All learning/model/data source files
remained unchanged during the final profile.

Resource stress included the maximum audited training cloud
(31,271 voxels), candidate-based long
contexts, paired/replay tasks, and 100 uncached generated tokens with the teacher
graph retained before driving/backward. Synthetic combined-cloud/text cases are
only resource checks. Global maximum text length and a complete eligible count
for the original dataset have not been established; the 100k epoch is conditional.
The pilot still lacks parking-lane coverage, and KV caching remains disabled.
Do not silently truncate contexts, drop OOM samples, or treat these checks as
live-agent parity or safety evidence.

Measurements and generated examples: `/ehsan/m.sadegh/simlingo_lidar/stage3/run-20261024c`.
Machine-readable task rates, resource readings, provenance and budget formula
inputs: [`stage3-results.json`](../tools/lidar_stage3/stage3-results.json).
Reproduction: [`Stage 3 README`](../tools/lidar_stage3/README.md).

**Next:** review the actual budget and parking coverage with the user before
Stage 4. Notify the user before any main training on GPU 2. Nothing auto-launches.

Follow-up: the [completed readiness checks](lidar-readiness-report.md) add parking,
full released RGB validation and exact fresh-process recovery. That recovery
exposed sparse-kernel/RNG state missing from the same-process test above; the
follow-up report records its correction and the revised budget. Stage 4 still
requires budget review and notification before launch.
