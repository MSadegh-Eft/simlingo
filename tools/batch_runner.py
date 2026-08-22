#!/usr/bin/env python3
"""
batch_runner.py -- Hardened 220-route batch evaluation runner for
SimLingo inside Bench2Drive / CARLA.

WHAT THIS DOES
  - Iterates all 220 routes by POSITION-INDEX (0-219), matching the
    `bench2drive_XX.xml` split-file / `bench2drive_XX_result.json`
    naming convention already used by your smoke-test results
    (confirmed via cross-check: position 167 == YieldToEmergencyVehicle
    == the yield infraction seen in bench2drive_167_result.json).
  - Restarts CARLA fresh before every attempt (matches your validated
    10-route practice; trades some wall-clock time for not needing to
    trust a single CARLA process to survive 220 sequential routes
    unattended on a shared box).
  - Retries a failed route up to MAX_RETRIES times, then flags it for
    manual review and moves on -- never blocks the whole batch on one
    route.
  - Skips any route that already has a VALID result (not just an
    existing file -- see result_is_valid()), so this script is safe
    to Ctrl+C and simply re-run to resume.
  - Appends one line per attempt to batch_status.jsonl for live
    progress (tail -f it) and a full audit trail.

BEFORE YOUR FIRST REAL RUN, CHECK EVERY LINE MARKED "VERIFY" BELOW.
Everything else in the CONFIG block is taken directly from your
confirmed-working eval command template / source-code findings.

USAGE
    conda activate simlingo   # activate_env.sh is sourced per-attempt below too,
                               # but the python interpreter running THIS script
                               # doesn't need to be inside the env itself.
    python3 batch_runner.py
    # Safe to interrupt (Ctrl+C) and re-run at any time -- already-valid
    # routes are skipped, in-flight CARLA/eval processes are cleaned up.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

# ======================================================================
# CONFIG
# ======================================================================

WORK_DIR = "/data/ghazaleh/simlingo"

# NOTE: two genuinely different Bench2Drive trees are involved here --
# don't collapse them into one root again.
#   1. The NESTED copy inside the simlingo fork ({WORK_DIR}/Bench2Drive) --
#      this is where the vendored leaderboard evaluator actually lives,
#      confirmed via activate_env.sh's LEADERBOARD_ROOT export.
#   2. The STANDALONE, top-level clone -- this is where bench2drive220.xml
#      actually lives, per your own confirmed path (sibling to simlingo/,
#      NOT inside it).
BENCH2DRIVE_ROOT = f"{WORK_DIR}/Bench2Drive"  # tree #1 -- nested
LEADERBOARD_EVALUATOR = f"{BENCH2DRIVE_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py"
ACTIVATE_ENV_SCRIPT = f"{WORK_DIR}/activate_env.sh"

# Combined 220-route file -- used ONLY to read scenario_type/town metadata
# by position-index. Deliberately NOT passed to --routes (see module
# docstring: the per-route split files are the confirmed-working path).
# Tree #2 -- standalone, NOT under WORK_DIR/BENCH2DRIVE_ROOT.
ROUTES_METADATA_XML = "/data/ghazaleh/Bench2Drive/leaderboard/data/bench2drive220.xml"

# Per-route split files -- THIS is what --routes actually points at.
ROUTES_SPLIT_DIR = f"{WORK_DIR}/leaderboard/data/bench2drive_split"


def route_split_file(idx: int) -> str:
    return f"{ROUTES_SPLIT_DIR}/bench2drive_{idx:02d}.xml"


AGENT_PATH = f"{WORK_DIR}/team_code/agent_simlingo.py"
# VERIFY: confirm this checkpoint is still the one you want for the full 220-route run.
AGENT_CHECKPOINT = f"{WORK_DIR}/models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"

# VERIFY: pick where the real 220-route output should live.
OUTPUT_DIR = f"{WORK_DIR}/eval220"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
VIZ_DIR = f"{OUTPUT_DIR}/viz"
LOGS_DIR = f"{OUTPUT_DIR}/logs"
MANIFEST_PATH = f"{OUTPUT_DIR}/batch_status.jsonl"

# Confirmed against activate_env.sh's own export -- no longer a guess.
CARLA_ROOT = "/data/ghazaleh/carla"
CARLA_PORT = 2000
TRAFFIC_MANAGER_PORT = 8000

# activate_env.sh's `conda activate simlingo` (its first line) silently fails
# in a non-interactive `bash -c` subshell -- no conda init/hook sourcing
# precedes it, so the shell function conda activate needs was never loaded.
# It hasn't broken anything YET only because bash doesn't stop on one failed
# line mid-script, and the parent shell's PATH (if already simlingo-activated)
# gets inherited regardless. That's fragile -- bypassing it entirely by
# invoking the env's python directly, independent of shell/conda state.
SIMLINGO_PYTHON = "/data/ghazaleh/miniconda3/envs/simlingo/bin/python"
EVAL_TIMEOUT = 600  # confirmed --timeout value from your template
TRACK = "SENSORS"
REPETITIONS = 1

MAX_RETRIES = 3

# Redesigned twice. First pass used a flat elapsed-time cap -- wrong,
# because it can't distinguish "naturally slow" from "actually stuck"
# (route 4 hit it while genuinely progressing at 0.052x realtime). Second
# pass added stall detection but kept a 3-hour absolute backstop on top --
# also wrong: route 6 confirmed 74 real minutes for a route stuck
# indecisively behind traffic, entirely healthy the whole time (Game time
# never stopped advancing), and worse GPU contention than we've already
# observed (0.015x) would need well over 3 hours for an equally healthy
# route. Any fixed duration number is unsafe here, because contention is
# unpredictable -- there's no number that's "long enough" that isn't also
# sometimes wrong.
#
# What's actually invariant: SimLingo's own internal tick_count > 4000
# cap (scenario_manager.py, confirmed from source) fires at exactly 200s
# of SIMULATED time regardless of how long that takes in the real world --
# so any route that's genuinely still ticking terminates on its own, no
# matter how slow. The only thing that DOESN'T self-resolve is a true
# stall: zero ticks happening at all (deadlock, frozen connection, a crash
# that doesn't exit cleanly) -- tick_count just sits frozen forever in
# that case, and the internal cap never gets the chance to fire. That's
# the actual gap external supervision needs to cover, and it doesn't
# require guessing a duration at all -- only "has any progress happened
# recently", which is safe at any reasonable timeout since a healthy route
# is never stuck at the identical Game time for this long, however slow
# it is overall. Set comfortably above the internal 600s per-tick watchdog
# so that gets first chance to resolve things on its own.
STALL_TIMEOUT_SECONDS = 20 * 60  # 20 minutes with zero game-time progress = genuinely stuck


GPU_CANDIDATES = [0, 1, 2]
# Real incident on this box: memory.used alone picked GPU 0 while it sat at
# 100% utilization, and the run crawled at 0.015x realtime as a result --
# 4.7 hours projected to finish something that normally takes minutes.
# Selection now screens by utilization FIRST (avoid compute-saturated GPUs
# entirely), then breaks ties by memory.used among what's left.
GPU_UTIL_THRESHOLD = 50  # percent -- placeholder, tune from what you observe day to day
# GPU 2's real incident: 3.6GB free was clearly not enough (0.03x ratio,
# 92% memory used). This is a placeholder set comfortably above that failure
# point, not a measured requirement -- monitor_resources.py's RSS numbers
# are real, but per-process GPU memory attribution isn't available on this
# box (see its own notes), so there's no precise "CARLA+model actually need
# N GB" figure to set this from yet. Tune once you have one.
GPU_MIN_FREE_MEM_MB = 15000
GPU_WAIT_POLL_SECONDS = 120
GPU_MAX_WAIT_SECONDS = 20 * 60  # give up waiting and proceed with least-bad option after this long

TOTAL_ROUTES = 220

# ======================================================================
# Process-group tracking + graceful shutdown
#
# Motivation: your own handoff notes bench2drive_200 needed a re-run after
# being killed mid-crash-cascade. Ungraceful kills leaving orphaned CARLA
# processes behind is a known, real failure mode here -- so every subprocess
# this script launches is tracked and force-killed on exit, whether that
# exit is normal, an exception, or Ctrl+C.
# ======================================================================

_active_pgids = set()


def _track(proc):
    try:
        pgid = os.getpgid(proc.pid)
        _active_pgids.add(pgid)
    except ProcessLookupError:
        pass
    return proc


def _untrack(proc):
    try:
        pgid = os.getpgid(proc.pid)
        _active_pgids.discard(pgid)
    except ProcessLookupError:
        pass


def _kill_all_tracked():
    for pgid in list(_active_pgids):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _active_pgids.clear()
    kill_stray_carla()


def _signal_handler(signum, frame):
    print(f"\n[batch_runner] received signal {signum} -- cleaning up and exiting")
    _kill_all_tracked()
    sys.exit(1)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ======================================================================
# Manifest
# ======================================================================


def append_manifest(record: dict):
    record = {**record, "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(MANIFEST_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ======================================================================
# Route metadata (scenario_type / town, by position-index)
# ======================================================================


def load_route_metadata():
    if not os.path.exists(ROUTES_METADATA_XML):
        raise FileNotFoundError(
            f"ROUTES_METADATA_XML not found at: {ROUTES_METADATA_XML}\n"
            f"This path is hardcoded in the CONFIG block, separately from "
            f"BENCH2DRIVE_ROOT -- if it's wrong, run "
            f"`find /data/ghazaleh -maxdepth 6 -iname bench2drive220.xml 2>/dev/null` "
            f"to locate it and update the constant directly, rather than guessing again."
        )
    tree = ET.parse(ROUTES_METADATA_XML)
    routes = tree.getroot().findall("route")
    if len(routes) != TOTAL_ROUTES:
        raise RuntimeError(
            f"Expected {TOTAL_ROUTES} routes in {ROUTES_METADATA_XML}, found "
            f"{len(routes)}. Stopping rather than silently misaligning indices "
            f"against the bench2drive_XX split-file naming."
        )
    metadata = []
    for idx, r in enumerate(routes):
        scen = r.find("scenarios/scenario")
        metadata.append(
            {
                "route_idx": idx,
                "xml_route_id": r.get("id"),
                "town": r.get("town"),
                "scenario_type": scen.get("type") if scen is not None else None,
            }
        )
    return metadata


# ======================================================================
# GPU selection
# ======================================================================


def query_gpu_stats():
    """Returns {idx: {"mem_used": float, "mem_free": float, "util": float}} or {} on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
    except Exception as e:
        print(f"[WARN] nvidia-smi query failed ({e})")
        return {}

    stats = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0])
            mem_used = float(parts[1])
            mem_total = float(parts[2])
            util = float(parts[3])
        except ValueError:
            continue
        if idx in GPU_CANDIDATES:
            stats[idx] = {"mem_used": mem_used, "mem_free": mem_total - mem_used, "util": util}
    return stats


def pick_gpu() -> int:
    """Screens on BOTH utilization and free memory -- a candidate must clear
    both to be considered usable. Two real, opposite incidents on this box
    motivated this: GPU 0 had low memory but 100% util (burned ~40 min at
    0.015x realtime), and GPU 2 had low util (6%) but only ~3.6GB free out
    of 46GB (crawled at 0.03x with compute nearly idle) -- neither axis
    alone is a reliable screen. Waits and re-polls if no candidate clears
    both, rather than launching into a run likely to crawl or trip a
    watchdog timeout; gives up after GPU_MAX_WAIT_SECONDS so an unattended
    run doesn't stall forever if the shared server stays busy."""
    waited = 0
    while True:
        stats = query_gpu_stats()
        if not stats:
            print(f"[WARN] no GPU stats available; defaulting to GPU {GPU_CANDIDATES[0]}")
            return GPU_CANDIDATES[0]

        usable = {
            i: s for i, s in stats.items()
            if s["util"] < GPU_UTIL_THRESHOLD and s["mem_free"] >= GPU_MIN_FREE_MEM_MB
        }
        print(f"[gpu] current state: {stats}")

        if usable:
            chosen = min(usable, key=lambda i: usable[i]["mem_used"])
            print(f"[gpu] choosing GPU {chosen} (util={stats[chosen]['util']}%, free={stats[chosen]['mem_free']:.0f}MiB)")
            return chosen

        if waited >= GPU_MAX_WAIT_SECONDS:
            # least-bad: prefer clearing the memory floor over the util
            # threshold, since memory pressure was the harder failure to
            # diagnose after the fact -- an OOM crash is more informative
            # than a silent crawl.
            mem_ok = {i: s for i, s in stats.items() if s["mem_free"] >= GPU_MIN_FREE_MEM_MB}
            pool = mem_ok if mem_ok else stats
            chosen = min(pool, key=lambda i: pool[i]["util"])
            print(
                f"[WARN] no GPU clears both thresholds after waiting {waited}s -- "
                f"proceeding anyway with GPU {chosen} (least-bad option: {stats[chosen]})"
            )
            return chosen

        print(
            f"[gpu] all GPUs at/above {GPU_UTIL_THRESHOLD}% util -- waiting {GPU_WAIT_POLL_SECONDS}s "
            f"before rechecking ({waited}s waited so far, giving up at {GPU_MAX_WAIT_SECONDS}s)"
        )
        time.sleep(GPU_WAIT_POLL_SECONDS)
        waited += GPU_WAIT_POLL_SECONDS


# ======================================================================
# CARLA lifecycle
# ======================================================================


def kill_stray_carla():
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4-Linux"], capture_output=True)
    time.sleep(2)


def wait_for_port_free(port: int, timeout: int = 30) -> bool:
    """Actively verifies the port is bindable rather than trusting a fixed
    sleep -- prevents the exact 'bind: Address already in use' crash seen
    when a retry launches CARLA before the previous instance's socket is
    fully released by the OS. Returns False (rather than hanging forever)
    if something else entirely is squatting on the port -- that's a
    different problem than our own cleanup timing and shouldn't be
    silently retried into indefinitely."""
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", port))
            s.close()
            return True
        except OSError:
            s.close()
            time.sleep(1)
    print(f"[WARN] port {port} still not free after {timeout}s -- may be an external process, not our own cleanup")
    return False


def launch_carla(gpu_index: int, log_path: str):
    cmd = (
        f"cd {CARLA_ROOT} && "
        f"./CarlaUE4.sh -RenderOffScreen -nosound "
        f"-graphicsadapter={gpu_index} -carla-rpc-port={CARLA_PORT}"
    )
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    _track(proc)
    return proc, log_f


def teardown_carla(proc, log_f):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        pass
    _untrack(proc)
    log_f.close()
    kill_stray_carla()  # belt-and-suspenders, same spirit as clean_carla.sh


# ======================================================================
# Result validation -- "done" means genuinely complete AND actually
# successful, not just present with some populated fields.
#
# Fixed against real evidence from start_eval_simlingo.py's own
# filter_completed(): status can literally be the string "Failed",
# "Failed - Agent crashed", "Failed - Simulation crashed", or
# "Failed - Agent couldn't be set up" -- all of which are non-empty,
# truthy strings that would have passed the old check even though they
# mean the route did NOT succeed. That check also never looked at
# _checkpoint.progress, which the original authors' own code treats as
# the primary completion signal (progress[0] < progress[1] means not
# actually finished).
# ======================================================================


def result_is_valid(result_path: str) -> bool:
    if not os.path.exists(result_path):
        return False
    try:
        with open(result_path) as f:
            d = json.load(f)
        checkpoint = d["_checkpoint"]

        progress = checkpoint.get("progress", [])
        if len(progress) < 2 or progress[0] < progress[1]:
            return False

        record = checkpoint["records"][0]
        status = record.get("status") or ""
        if status.startswith("Failed"):
            return False

        has_scores = bool(record.get("scores"))
        has_infractions = "infractions" in record
        return bool(status) and has_scores and has_infractions
    except Exception:
        return False


HEARTBEAT_INTERVAL_SECONDS = 15


def last_agent_line(log_path: str) -> str:
    """Tails the last few KB of the eval log and returns the most recent
    '=== [Agent] ...' line, if any -- avoids re-reading the whole file
    every heartbeat on a long-running route."""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            chunk = f.read().decode(errors="replace")
    except Exception:
        return ""
    agent_lines = [line for line in chunk.splitlines() if line.startswith("=== [Agent]")]
    return agent_lines[-1] if agent_lines else ""


def parse_game_time(agent_line: str):
    """Extracts the Game time float from a '=== [Agent] ...' line, or None
    if the line is empty/unparseable. This is the actual progress signal --
    route 6 proved wallclock elapsed alone can't distinguish a route that's
    legitimately still working (indecisive lane-change, Game time steadily
    advancing) from one that's genuinely stuck (Game time frozen)."""
    m = re.search(r"Game time = ([\d.]+)", agent_line)
    return float(m.group(1)) if m else None


def hit_tick_limit(eval_log_path: str) -> bool:
    """Checks whether this attempt failed specifically because SimLingo's
    own internal tick_count > 4000 cap fired (scenario_manager.py,
    confirmed from source), as opposed to a crash, infra hiccup, or a
    stall we killed ourselves. A tick-limit hit means the agent/scenario
    ran its FULL simulated budget and still didn't resolve -- a model or
    scenario problem, not something a bare retry with nothing changed is
    likely to fix differently. Detected via the literal error text the
    framework raises, not guessed."""
    try:
        with open(eval_log_path, "r", errors="replace") as f:
            content = f.read()
        return "tick_count > 4000" in content or "TickRuntimeError" in content
    except Exception:
        return False


# ======================================================================
# Single attempt: run one route once CARLA is already up
# ======================================================================


def run_single_route(route_idx: int, gpu_index: int, attempt: int):
    """Returns (success: bool, eval_log_path: str)."""
    route_file = route_split_file(route_idx)
    if not os.path.exists(route_file):
        print(f"[route {route_idx}] ERROR: split file not found: {route_file}")
        return False, None

    result_path = f"{RESULTS_DIR}/bench2drive_{route_idx:02d}_result.json"
    debug_checkpoint_path = f"{RESULTS_DIR}/bench2drive_{route_idx:02d}_live.txt"
    save_path = f"{VIZ_DIR}/bench2drive_{route_idx:02d}/"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    eval_cmd = (
        f"source {ACTIVATE_ENV_SCRIPT} && "
        f"cd {WORK_DIR} && "
        f"export SAVE_PATH={save_path} && "
        f"CUDA_VISIBLE_DEVICES={gpu_index} {SIMLINGO_PYTHON} -u {LEADERBOARD_EVALUATOR} "
        f"--routes={route_file} "
        f"--repetitions={REPETITIONS} --track={TRACK} "
        f"--checkpoint={result_path} "
        f"--timeout={EVAL_TIMEOUT} --agent={AGENT_PATH} "
        f'--agent-config="{AGENT_CHECKPOINT}" --traffic-manager-seed=1 '
        f"--port={CARLA_PORT} --traffic-manager-port={TRAFFIC_MANAGER_PORT} "
        f"--debug-checkpoint={debug_checkpoint_path}"
    )

    eval_log_path = f"{LOGS_DIR}/bench2drive_{route_idx:02d}_eval_attempt{attempt}.log"
    with open(eval_log_path, "w") as log_f:
        proc = subprocess.Popen(
            ["bash", "-c", eval_cmd],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        _track(proc)
        start = time.time()
        last_game_time = None
        last_progress_at = start
        try:
            while True:
                try:
                    proc.wait(timeout=HEARTBEAT_INTERVAL_SECONDS)
                    break  # process exited
                except subprocess.TimeoutExpired:
                    pass
                now = time.time()
                elapsed = now - start

                heartbeat = last_agent_line(eval_log_path)
                game_time = parse_game_time(heartbeat)
                if game_time is not None and game_time != last_game_time:
                    last_game_time = game_time
                    last_progress_at = now

                stalled_for = now - last_progress_at
                if stalled_for >= STALL_TIMEOUT_SECONDS:
                    print(
                        f"[route {route_idx}] STALLED -- Game time hasn't advanced in "
                        f"{stalled_for:.0f}s (last value: {last_game_time}) -- killing"
                    )
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=15)
                    break

                suffix = f" | {heartbeat}" if heartbeat else " | (no agent output yet)"
                print(f"[route {route_idx}] attempt {attempt} still running ({elapsed:.0f}s elapsed){suffix}")
        finally:
            _untrack(proc)

    return result_is_valid(result_path), eval_log_path


# ======================================================================
# Per-route orchestration: skip-if-done, else retry loop with fresh CARLA
# ======================================================================


def process_route(meta: dict, forced_gpu: int = None) -> str:
    route_idx = meta["route_idx"]
    result_path = f"{RESULTS_DIR}/bench2drive_{route_idx:02d}_result.json"

    if result_is_valid(result_path):
        append_manifest({**meta, "attempt": 0, "outcome": "already_complete"})
        return "already_complete"

    for attempt in range(1, MAX_RETRIES + 1):
        gpu_index = forced_gpu if forced_gpu is not None else pick_gpu()
        print(
            f"[route {route_idx}] attempt {attempt}/{MAX_RETRIES} on GPU {gpu_index} "
            f"({meta['scenario_type']}, {meta['town']})"
        )

        kill_stray_carla()
        wait_for_port_free(CARLA_PORT)
        os.makedirs(LOGS_DIR, exist_ok=True)
        carla_log_path = f"{LOGS_DIR}/bench2drive_{route_idx:02d}_carla_attempt{attempt}.log"
        carla_proc, carla_log_f = launch_carla(gpu_index, carla_log_path)

        start = time.time()
        try:
            success, eval_log_path = run_single_route(route_idx, gpu_index, attempt)
        finally:
            teardown_carla(carla_proc, carla_log_f)
        duration = time.time() - start

        tick_limited = (not success) and eval_log_path and hit_tick_limit(eval_log_path)
        outcome = "success" if success else ("failed_tick_limit" if tick_limited else "failed")

        append_manifest(
            {
                **meta,
                "attempt": attempt,
                "gpu_index": gpu_index,
                "duration_seconds": round(duration, 1),
                "outcome": outcome,
            }
        )

        if success:
            print(f"[route {route_idx}] SUCCESS on attempt {attempt} ({duration:.0f}s)")
            return "success"

        if tick_limited:
            print(
                f"[route {route_idx}] hit SimLingo's internal tick_count>4000 cap on attempt {attempt} "
                f"({duration:.0f}s) -- not retrying, this is a model/scenario problem, not infra. "
                f"Flagging for manual review."
            )
            append_manifest({**meta, "attempt": attempt, "outcome": "needs_manual_review", "reason": "tick_limit"})
            return "needs_manual_review"

        print(f"[route {route_idx}] attempt {attempt} failed ({duration:.0f}s)")

    append_manifest({**meta, "attempt": MAX_RETRIES, "outcome": "needs_manual_review"})
    return "needs_manual_review"


# ======================================================================
# CLI args -- for scoping a test run. Deliberately does NOT touch
# TOTAL_ROUTES, which is a validation invariant against the 220-route
# XML, not a "how many to run" knob.
# ======================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="SimLingo/Bench2Drive batch runner")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N routes (position-index order). For testing.",
    )
    parser.add_argument(
        "--only", type=int, nargs="+", default=None,
        help="Only process these specific route indices, e.g. --only 167. Overrides --limit.",
    )
    parser.add_argument(
        "--gpu", type=int, default=None, choices=GPU_CANDIDATES,
        help="Force a specific GPU index, bypassing auto-selection entirely.",
    )
    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================


def main():
    args = parse_args()

    for d in (OUTPUT_DIR, RESULTS_DIR, VIZ_DIR, LOGS_DIR):
        os.makedirs(d, exist_ok=True)

    metadata = load_route_metadata()
    if args.only:
        wanted = set(args.only)
        metadata = [m for m in metadata if m["route_idx"] in wanted]
        missing = wanted - {m["route_idx"] for m in metadata}
        if missing:
            print(f"[WARN] requested route indices not found: {sorted(missing)}")
    elif args.limit:
        metadata = metadata[: args.limit]

    print(
        f"Starting batch: {len(metadata)} route(s) (of {TOTAL_ROUTES} total), "
        f"max {MAX_RETRIES} retries each. Killed only on genuine stall "
        f"({STALL_TIMEOUT_SECONDS}s with zero Game-time progress) -- no fixed "
        f"duration cap, since SimLingo's own internal tick_count cap already "
        f"bounds any healthy run regardless of how slow it is.\n"
        f"Manifest: {MANIFEST_PATH}\n"
    )

    summary = {"already_complete": 0, "success": 0, "needs_manual_review": 0}
    flagged_routes = []

    for meta in metadata:
        outcome = process_route(meta, forced_gpu=args.gpu)
        summary[outcome] = summary.get(outcome, 0) + 1
        if outcome == "needs_manual_review":
            flagged_routes.append(meta["route_idx"])
        print(f"[route {meta['route_idx']}] -> {outcome}\n")

    print("=== Batch complete ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if flagged_routes:
        print(f"\nRoutes needing manual review: {flagged_routes}")


if __name__ == "__main__":
    main()