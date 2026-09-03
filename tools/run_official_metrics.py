#!/usr/bin/env python3
"""
run_official_metrics.py -- runs the OFFICIAL Bench2Drive metric pipeline
over a finished eval run directory (default: eval220v2), so the metrics
are computed the way the benchmark intends instead of via ad-hoc
re-implementations:

  1. Metric-info flattening: the agent writes metric_info.json deep under
     viz/bench2drive_XX/{save_name}/debug_viz/.../metric/, but the official
     efficiency_smoothness_benchmark.py expects
     {metric_dir}/{save_name}/metric_info.json. This step maps each result
     record's save_name (unique per attempt -- it embeds a currentTime) to
     that attempt's metric_info.json and copies it into the flat layout.
     Keying on save_name (instead of "latest mtime") is what correctly
     disambiguates retried routes, where more metric_info.json files exist
     on disk than there are routes.
  2. merge_route_json.py            -> results/merged.json
     (Driving Score, Success Rate over all records)
  3. ability_benchmark.py           -> results/merged_ability.json
     (per-ability success: Overtaking / Merging / Emergency_Brake /
     Give_Way / Traffic_Signs. NOTE: spawns its own CARLA server on the
     script's built-in default port 4000 and needs a free port + GPU;
     skipped by --skip-ability. The spawned server is killed again
     afterwards, unlike the official script which orphans it.)
  4. efficiency_smoothness_benchmark.py
     -> official Driving Efficiency (from the min_speed_infractions
     messages -- yes, that is genuinely how the official script computes
     it) + Driving Smoothness (comfort metric from metric_info.json)

Each step skips itself if its output already exists (delete the output or
pass --force to redo). --dry-run only validates the metric-info mapping
against what is on disk; no files are written and no subprocesses run.

Usage:
    python3 tools/run_official_metrics.py
    python3 tools/run_official_metrics.py --dry-run
    python3 tools/run_official_metrics.py --skip-ability
    python3 tools/run_official_metrics.py --eval-dir /data/ghazaleh/simlingo/eval220
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK_DIR = "/data/ghazaleh/simlingo"
BENCH2DRIVE_TOOLS = f"{WORK_DIR}/Bench2Drive/tools"
ACTIVATE_ENV_SCRIPT = f"{WORK_DIR}/activate_env.sh"
SIMLINGO_PYTHON = "/data/ghazaleh/miniconda3/envs/simlingo/bin/python"
# The tracked copy of the 220-route xml (ability_benchmark needs route
# ids/towns/waypoints from it).
ROUTES_XML = f"{WORK_DIR}/leaderboard/data/bench2drive220.xml"
TOTAL_ROUTES = 220

# Fields the official comfort metric reads from every step of
# metric_info.json -- verified against agent_simlingo's get_metric_info()
# (inherited from autonomous_agent.py) and the official script's
# read_from_json().
EXPECTED_METRIC_KEYS = [
    "acceleration", "angular_velocity", "forward_vector",
    "right_vector", "location", "rotation",
]

def load_valid_record(result_path: Path):
    """Returns (record, None) for a complete route result, else (None, reason).
    Same validity rule as batch_runner.py / build_matrix.py: progress
    finished and score_composed present."""
    try:
        with open(result_path) as f:
            d = json.load(f)
        checkpoint = d["_checkpoint"]
        progress = checkpoint.get("progress", [])
        if len(progress) < 2 or progress[0] < progress[1]:
            return None, "incomplete_progress"
        record = checkpoint["records"][0]
        if "score_composed" not in record.get("scores", {}):
            return None, "no_scores"
        return record, None
    except Exception as e:  # noqa: BLE001 -- report anything unreadable
        return None, f"unreadable: {e}"


def find_metric_info(route_viz_dir: Path, save_name: str):
    """Latest metric_info.json under this attempt's save_name dir."""
    base = route_viz_dir / save_name
    if not base.is_dir():
        return None
    candidates = sorted(
        base.rglob("metric_info.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def metric_file_is_wellformed(path: Path) -> bool:
    """Light sanity check: first step must carry all fields the official
    comfort metric reads."""
    try:
        with open(path) as f:
            d = json.load(f)
        first = next(iter(d.values()), None)
        return isinstance(first, dict) and all(k in first for k in EXPECTED_METRIC_KEYS)
    except Exception:  # noqa: BLE001
        return False


def run_in_env(cmd: str, log_path: Path = None):
    """Runs a command inside the simlingo env (activate_env.sh provides
    CARLA_ROOT / PYTHONPATH; the conda activate line is a no-op in
    non-interactive shells, which is why the env's python is invoked by
    absolute path -- same workaround as batch_runner.py)."""
    full = f"source {ACTIVATE_ENV_SCRIPT} && cd {WORK_DIR} && {SIMLINGO_PYTHON} -u {cmd}"
    if log_path is None:
        return subprocess.run(["bash", "-c", full])
    with open(log_path, "w") as log_f:
        return subprocess.run(["bash", "-c", full], stdout=log_f, stderr=subprocess.STDOUT)

def step_flatten(eval_dir: Path, force: bool = False, dry_run: bool = False) -> bool:
    """Maps save_name -> metric_info.json into the official flat layout.
    Returns True only if every route is valid AND has a well-formed
    metric_info.json (i.e. the official scripts can run unimpeded)."""
    results_dir = eval_dir / "results"
    viz_dir = eval_dir / "viz"
    metric_dir = eval_dir / "metric"

    invalid_results = []   # (route_idx, reason)
    missing_metric = []    # valid result but no metric_info.json found
    malformed = []         # found but missing expected fields
    ready = 0
    copied = 0

    for idx in range(TOTAL_ROUTES):
        result_path = results_dir / f"bench2drive_{idx:02d}_result.json"
        if not result_path.exists():
            invalid_results.append((idx, "no_result_file"))
            continue
        record, reason = load_valid_record(result_path)
        if record is None:
            invalid_results.append((idx, reason))
            continue
        save_name = record.get("save_name")
        src = find_metric_info(viz_dir / f"bench2drive_{idx:02d}", save_name) if save_name else None
        if src is None:
            missing_metric.append(idx)
            continue
        if not metric_file_is_wellformed(src):
            malformed.append(idx)
            continue
        ready += 1
        if dry_run:
            continue
        dst_dir = metric_dir / save_name
        dst = dst_dir / "metric_info.json"
        if force or not dst.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"[flatten] ready routes: {ready}/{TOTAL_ROUTES}")
    if invalid_results:
        print(f"[flatten] routes WITHOUT a valid result (official pipeline "
              f"will warn about != {TOTAL_ROUTES} records): {invalid_results}")
    if missing_metric:
        print(f"[flatten] valid results with NO metric_info.json: {missing_metric}")
    if malformed:
        print(f"[flatten] metric_info.json missing expected fields: {malformed}")
    if dry_run:
        print("[flatten] dry-run: nothing written")
    else:
        print(f"[flatten] copied {copied} metric_info.json file(s) into {metric_dir}")
    return ready == TOTAL_ROUTES


def step_merge(eval_dir: Path, force: bool = False) -> Path:
    """Official merge: Driving Score + Success Rate -> results/merged.json."""
    merged = eval_dir / "results" / "merged.json"
    if merged.exists() and not force:
        print(f"[merge] skipped, {merged} already exists")
        return merged
    log_path = eval_dir / "merge_route_json.log"
    print(f"[merge] running merge_route_json.py (log: {log_path})")
    result = run_in_env(
        f"{BENCH2DRIVE_TOOLS}/merge_route_json.py -f {eval_dir / 'results'}",
        log_path=log_path,
    )
    if result.returncode != 0 or not merged.exists():
        print(f"[merge] FAILED (exit {result.returncode}) -- see {log_path}")
        sys.exit(1)
    print(f"[merge] wrote {merged}")
    return merged

def step_ability(eval_dir: Path, port: int, force: bool = False) -> Path:
    """Official ability benchmark (Overtaking / Merging / Emergency_Brake /
    Give_Way / Traffic_Signs). Spawns its OWN CARLA server (port 4000, the
    script's built-in default) -- needs a genuinely free port and some GPU;
    takes a while because it loads several towns for the Traffic_Signs
    route planning.

    Deliberately does NOT pass -p/--port: the official script declares it
    with argparse nargs=1, so passing '-p 4000' makes args.port a LIST
    (['4000']), which corrupts both the CARLA spawn command string
    (-carla-rpc-port=['4000']) and carla.Client(host, ['4000']). The flag
    only works when omitted -- exactly how the official README invokes it."""
    ability_json = eval_dir / "results" / "merged_ability.json"
    merged = eval_dir / "results" / "merged.json"
    if ability_json.exists() and not force:
        print(f"[ability] skipped, {ability_json} already exists")
        return ability_json
    log_path = eval_dir / "ability_benchmark.log"
    print(f"[ability] running ability_benchmark.py with its own CARLA on "
          f"port {port} (log: {log_path}) -- this loads towns and can take minutes")
    result = run_in_env(
        # No -p: see the docstring -- the official script's nargs=1 turns it
        # into a list and breaks both the spawn command and carla.Client.
        f"{BENCH2DRIVE_TOOLS}/ability_benchmark.py "
        f"-f {ROUTES_XML} -r {merged}",
        log_path=log_path,
    )
    # The official script never kills the CARLA server it spawns -- don't
    # orphan it on this shared box. The server always runs on the script's
    # own default port 4000 (see docstring for why -p is not passed), so
    # --ability-port only parameterizes this cleanup match.
    subprocess.run(["pkill", "-9", "-f", f"carla-rpc-port={port}"], capture_output=True)
    time.sleep(2)
    if result.returncode != 0 or not ability_json.exists():
        print(f"[ability] FAILED (exit {result.returncode}) -- see {log_path}; "
              f"re-run with --skip-ability to produce everything else")
        return None
    print(f"[ability] wrote {ability_json}")
    return ability_json


def step_efficiency_smoothness(eval_dir: Path) -> dict:
    """Official Driving Efficiency + Driving Smoothness from the flattened
    metric_info.json layout."""
    merged = eval_dir / "results" / "merged.json"
    metric_dir = eval_dir / "metric"
    log_path = eval_dir / "efficiency_smoothness_benchmark.log"
    print(f"[eff+smooth] running efficiency_smoothness_benchmark.py (log: {log_path})")
    result = subprocess.run(
        ["bash", "-c",
         f"source {ACTIVATE_ENV_SCRIPT} && cd {WORK_DIR} && "
         f"{SIMLINGO_PYTHON} {BENCH2DRIVE_TOOLS}/efficiency_smoothness_benchmark.py "
         f"-f {merged} -m {metric_dir}"],
        capture_output=True, text=True,
    )
    with open(log_path, "w") as f:
        f.write(result.stdout)
        f.write(result.stderr)
    parsed = {}
    for line in (result.stdout or "").splitlines():
        if line.startswith("Driving Efficiency="):
            parsed["Driving Efficiency"] = line.split("=", 1)[1].strip()
        elif line.startswith("Driving Smoothness="):
            parsed["Driving Smoothness"] = line.split("=", 1)[1].strip()
    if result.returncode != 0 or not parsed:
        print(f"[eff+smooth] FAILED (exit {result.returncode}) -- see {log_path}")
        print((result.stderr or "")[-2000:])
    return parsed

def build_summary(eval_dir: Path, ability_json, eff_smooth: dict):
    """Aggregates the official outputs into one readable summary file."""
    summary_path = eval_dir / "official_metrics_summary.txt"
    lines = [f"Official Bench2Drive metrics -- {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"eval dir: {eval_dir}", ""]

    merged = eval_dir / "results" / "merged.json"
    if merged.exists():
        with open(merged) as f:
            m = json.load(f)
        lines += [
            "-- merge_route_json.py --",
            f"Driving Score : {m.get('driving score')}",
            f"Success Rate  : {m.get('success rate')}",
            f"Eval num      : {m.get('eval num')} (of {TOTAL_ROUTES}; "
            f"official tool warns if this differs)",
            "",
        ]
    else:
        lines += ["-- merge_route_json.py -- MISSING (merge step failed)", ""]

    if ability_json is not None and Path(ability_json).exists():
        with open(ability_json) as f:
            a = json.load(f)
        lines += ["-- ability_benchmark.py (per-ability success rate) --"]
        for key in ("Overtaking", "Merging", "Emergency_Brake", "Give_Way",
                    "Traffic_Signs", "mean"):
            if key in a:
                lines.append(f"{key:<16}: {a[key]}")
        if a.get("crashed"):
            lines.append(f"crashed/missing routes: {len(a['crashed'])}")
        lines.append("")
    else:
        lines += ["-- ability_benchmark.py -- not run", ""]

    if eff_smooth:
        lines += ["-- efficiency_smoothness_benchmark.py --"]
        lines.append(f"Driving Efficiency : {eff_smooth.get('Driving Efficiency')}")
        lines.append(f"Driving Smoothness : {eff_smooth.get('Driving Smoothness')}")
        lines.append("")
    else:
        lines += ["-- efficiency_smoothness_benchmark.py -- not run or failed", ""]

    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\n===== {summary_path} =====")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Official Bench2Drive metric pipeline wrapper")
    parser.add_argument("--eval-dir", default=f"{WORK_DIR}/eval220v2",
                        help="Eval run directory containing results/ and viz/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only validate the metric_info mapping; write nothing, run nothing")
    parser.add_argument("--skip-ability", action="store_true",
                        help="Skip ability_benchmark.py (it spawns its own CARLA server)")
    parser.add_argument("--ability-port", type=int, default=4000,
                        help="Port to pkill-match when cleaning up the CARLA "
                             "server that ability_benchmark.py spawns on its "
                             "BUILT-IN default (4000). The script's own -p "
                             "flag is broken (argparse nargs=1 turns the "
                             "value into a list), so we never pass it -- "
                             "this only parameterizes the cleanup match.")
    parser.add_argument("--force", action="store_true",
                        help="Redo steps even if their outputs already exist")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not (eval_dir / "results").is_dir():
        print(f"ERROR: {eval_dir}/results does not exist -- nothing to compute yet")
        sys.exit(1)

    print(f"=== Step 1: metric-info flattening ({eval_dir}) ===")
    complete = step_flatten(eval_dir, force=args.force, dry_run=args.dry_run)
    if not complete:
        print("[flatten] WARNING: run is not fully complete/valid -- continuing with "
              "a partial official pipeline (the official tools handle this with warnings), "
              "but numbers are only comparable at 220/220.")
    if args.dry_run:
        print("dry-run complete -- no files written, no subprocesses run")
        return

    print("\n=== Step 2: merge_route_json.py ===")
    step_merge(eval_dir, force=args.force)

    ability_json = None
    if not args.skip_ability:
        print("\n=== Step 3: ability_benchmark.py ===")
        ability_json = step_ability(eval_dir, port=args.ability_port, force=args.force)
    else:
        print("\n=== Step 3: ability_benchmark.py SKIPPED (--skip-ability) ===")

    print("\n=== Step 4: efficiency_smoothness_benchmark.py ===")
    eff_smooth = step_efficiency_smoothness(eval_dir)

    build_summary(eval_dir, ability_json, eff_smooth)


if __name__ == "__main__":
    main()


