# SimLingo on Bench2Drive — Thesis Evaluation Fork

Personal fork of [SimLingo](https://github.com/RenzKa/simlingo) (CVPR'25 Highlight) used for my thesis work:
**evaluating the released SimLingo model on all 220 Bench2Drive routes, twice, with full instrumentation —
and analysing where the model, the sensors, and the benchmark itself each contribute to the failures.**

> **Author:** Mohammad Sadegh Eftekhar
> **Fork:** [MSadegh-Eft/simlingo](https://github.com/MSadegh-Eft/simlingo) ← upstream [RenzKa/simlingo](https://github.com/RenzKa/simlingo)
> **Period:** August – September 2026
> The original upstream README is preserved at [`README_original.md`](README_original.md).

---

## 1. What this fork does

Upstream SimLingo ships the model, training pipeline and a CARLA Leaderboard evaluation stack.
This fork adds a **complete Bench2Drive evaluation laboratory** around the released checkpoint:

- an unattended **220-route batch evaluation runner** with automatic retries, crash recovery and live monitoring
- a **resource monitor** (GPU/CPU/RAM sampling) running alongside every evaluation
- an **official-metrics pipeline** that reproduces Bench2Drive's own post-processing (ability benchmark, efficiency/smoothness)
- **chase-cam video generation** for every route, plus per-route detail exports
- **two full 220-route evaluations** of the identical checkpoint (independent runs), and a
- **220-route manual video review** with a written analysis report (per-route + thematic + code-level root causes)

Main deliverable: **`simlingo_bench2drive_report.docx`** —
the full report (12 sections: results, ability breakdown, per-scenario and per-route tables, manual review of
all 220 routes, code-verified root-cause map, ranked improvement plan, threats to validity).
The report itself is kept **outside the public repository** (see Section 9); this README is its companion summary.

## 2. Server configuration

All work ran on a single Linux server:

| Component | Specification |
|---|---|
| GPU | 3× NVIDIA RTX 6000 Ada Generation (48 GB each) |
| GPU driver / CUDA | 550.135 / CUDA 12.4 |
| CPU | 2× Intel Xeon Platinum 8480+ (224 threads) |
| System RAM | 64 GB |
| OS | Ubuntu 24.04 LTS |
| Storage | 984 GB NVMe mounted at `/data` (repo, CARLA, results and videos all live here) |

## 3. Software stack

| Software | Version | Used for |
|---|---|---|
| CARLA | **0.9.15** (`/data/ghazaleh/carla`) | simulator |
| Bench2Drive leaderboard | bundled in this repo (fork of CARLA Leaderboard 2.0) | evaluation harness |
| scenario_runner | bundled (Bench2Drive variant) | scenario logic |
| Python (env `simlingo`) | 3.8.18 | agent + model inference |
| Python (env `b2d`) | 3.8.20 | evaluation leaderboard |
| PyTorch | 2.2.0 (+ torchvision 0.17.0) | model |
| transformers | 4.46.3 | InternVL2 VLM backbone |
| flash-attn | 2.7.0.post2 | attention kernels |
| pytorch-lightning | 2.4.0 (+ deepspeed 0.16.2, accelerate 1.0.1) | training/inference scaffolding |
| carla (Python API) | 0.9.15 | agent–simulator bridge |
| numpy / opencv / pillow | 1.23.0 / 4.2.0.34 / 10.2.0 | data handling |

Model evaluated: the **released SimLingo checkpoint** (InternVL2-1B backbone), unmodified —
the thesis evaluates the public reproduction, not a re-trained model.

## 4. Key files I wrote

Everything below is new in this fork (upstream files are only touched where noted):

| File | Lines | What it does |
|---|---|---|
| `tools/batch_runner.py` | 796 | The core runner: launches CARLA + leaderboard per route, tracks state in `batch_status.jsonl`, auto-retries failed routes, resumes interrupted batches, manages attempts and tick limits |
| `tools/build_matrix.py` | 330 | Builds the per-scenario/per-route result matrices and flip analysis between runs |
| `tools/run_official_metrics.py` | 445 | Official-metrics pipeline: merges route JSONs, runs Bench2Drive's `ability_benchmark.py`, efficiency & smoothness benchmarks; includes a fix for the official tool's broken `-p` flag |
| `tools/monitor_resources.py` | 352 | Samples GPU/CPU/RAM during evaluations → `resource_samples.csv` + peak summaries |
| `tools/export_route_details.py` | 183 | Flattens raw result JSONs + manifest into `route_details.csv` (one row per route, official success flag, all infraction counts, durations) |
| `start_eval_simlingo.py` | 304 | Single-route evaluation entry point (upstream file, adapted for batch use) |
| `tools/build_videos_220.sh` | 72 | Generates chase-cam videos for all routes from the recorded runs |
| `tools/build_videos_all_attempts.sh` | 51 | Variant covering all recording attempts (untracked helper) |
| `build_videos.sh` | 34 | Upstream-style video builder used during smoke tests |
| `tools/cleanup_failed_attempts.py`, `tools/filter_broken_files.py`, `tools/infraction_gifs.py` | — | Housekeeping helpers for the results tree |
| `activate_env.sh` | — | One-command environment activation (conda envs + CARLA + leaderboard paths) |
| `team_code/agent_simlingo.py` | modified | Upstream agent; enabled debug-viz saving, guarded encoder cleanup, chase-cam viz sensor for the recorded videos |

Evaluation outputs (not in git, on `/data`): `eval220/` (run 1), `eval220v2/` (run 2, with videos + recorder logs) —
raw per-route result JSONs, merged results, official metrics, ability breakdowns, resource samples, videos.

## 5. Timeline of work

| Date (2026) | Milestone |
|---|---|
| Aug 16 | Smoke test of the upstream eval stack on single routes; first results + example video (`f594ed5`); agent viz fixes (`d19ee77`) |
| Aug 16 | **220-route batch runner** written (`7ef1c81`) — evaluation of the full benchmark becomes unattended |
| Aug 16–22 | **Run 1** (`eval220/`): all 220 routes evaluated |
| Aug 22 | Monitoring + review tooling (`a8c0e6d`): resource sampler, result matrices, video pipeline hardening |
| Sep 3 | Per-route details exporter (`7289693`); rerun prep with chase-cam viz + CARLA recorder logs + official metrics (`51e0d2d`); analysis tools repointed (`bcc9a7f`) |
| Sep 3 | **Run 2** (`eval220v2/`): full rerun of all 220 routes with instrumentation |
| Sep 3 | Official ability/efficiency metrics reproduced with a fix to the official tool (`69f4676`) |
| Sep 3–23 | Manual video review of all 220 routes (both runs), root-cause code analysis, and the written report |

## 6. Headline results

Identical checkpoint, identical 220 routes, two independent runs:

| Metric | Run 1 | Run 2 |
|---|---|---|
| Driving Score | 88.29 | 88.76 |
| Official success rate | 69.1% (152/220) | 71.4% (157/220) |
| Route completion (mean) | 99.5% | 99.5% |

- The **+2.3 pp run-to-run swing is the measured noise floor** — single-run ablation claims smaller than ~5 routes are not evidence.
- Weakest ability categories (official breakdown): **Merging (~52%)** and **Give Way (50%)**, flat across both runs; strongest: **Emergency Brake (~88%)**.
- Under the strict official success definition the merge-family scenario types (HighwayExit, EnterActorFlow, YieldToEmergencyVehicle) are 0/5 clean **in both runs**, but 3–5/5 under destination-reached: the model always finishes these routes, never legally.
- Key root causes (all code-verified in the report): single front-facing camera with single-frame input (no rear perception, no temporal memory); chain-of-thought text causally conditions the waypoints; benchmark properties — Min Speed logged-but-unscored, zero-penalty outside-route-lanes, no collision fault attribution (0.1 m/s EPSILON filter), multi-mechanism actor de-spawns.

## 7. Reproducing

```Shell
conda activate simlingo  # or: source activate_env.sh (sets CARLA_ROOT, WORK_DIR, leaderboard paths)

# Full 220-route batch evaluation (unattended, resumable)
python tools/batch_runner.py

# Official metrics (ability benchmark, efficiency, smoothness) from raw results
python tools/run_official_metrics.py

# Per-route flat table for analysis
python tools/export_route_details.py

# Chase-cam videos for all routes
bash tools/build_videos_220.sh
```

Results land in the run directory (e.g. `eval220v2/`): `results/*.json`, `route_details.csv`,
`official_metrics_summary.txt`, `resource_samples.csv`, `videos/`, `viz/`.

## 8. Credits

- **SimLingo** — RenzKa/simlingo (CVPR'25 Highlight): model, training pipeline, agent; see [`README_original.md`](README_original.md) and its citations.
- **Bench2Drive** — Thinklab-SJTU/Bench2Drive: benchmark, leaderboard, scenario suite (bundled under `Bench2Drive/`).
- **CARLA** — 0.9.15 simulator.
- Everything in Section 4 (runner, monitors, metrics pipeline, exports, reports) was written for this thesis fork.
- Licenses from the upstream projects apply to their respective code.

## 9. Notes

- The original upstream README is kept unchanged at [`README_original.md`](README_original.md).
- The full evaluation report (`simlingo_bench2drive_report.docx` / `.pdf`) is intentionally **not** part of
  this public repository — it contains server details and per-route data tied to the lab environment;
  it lives alongside the repo checkout locally. This README is written to stand in for it.
- The Claude conversation log documenting the earlier stages of this work may be added later
  (`claude_export/`); its download manifest is gitignored either way.

