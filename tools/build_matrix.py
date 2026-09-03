#!/usr/bin/env python3
"""
build_matrix.py -- aggregates eval220v2 results into the 44-scenario-type x
12-infraction-type analysis matrix, plus driving-score and pipeline stats.

READ-ONLY. Only reads bench2drive220.xml, eval220v2/results/*.json, and
eval220v2/batch_status.jsonl -- never touches CARLA. Deliberately does NOT
import batch_runner.py: that module installs signal handlers at import
time that kill CARLA processes on Ctrl+C, which would be actively
dangerous to run alongside a still-in-progress batch. Route-metadata
loading is duplicated here on purpose, not an oversight.

Safe to run at any point, including while batch_runner.py is still going --
reports on whatever has a valid result so far, nothing more.

Usage:
    python3 build_matrix.py
Outputs:
    eval220v2/analysis_matrix.csv   -- one row per scenario type
    eval220v2/analysis_summary.txt  -- overall aggregate, marginal infraction
                                      rates, ranked worst scenarios, pipeline stats
"""

import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# ======================================================================
# CONFIG -- same paths/constants as batch_runner.py, duplicated
# deliberately (see module docstring for why this isn't just imported).
# ======================================================================

WORK_DIR = "/data/ghazaleh/simlingo"
ROUTES_METADATA_XML = "/data/ghazaleh/Bench2Drive/leaderboard/data/bench2drive220.xml"
RESULTS_DIR = f"{WORK_DIR}/eval220v2/results"
MANIFEST_PATH = f"{WORK_DIR}/eval220v2/batch_status.jsonl"
OUTPUT_CSV = f"{WORK_DIR}/eval220v2/analysis_matrix.csv"
OUTPUT_SUMMARY = f"{WORK_DIR}/eval220v2/analysis_summary.txt"
TOTAL_ROUTES = 220
Z_95 = 1.959963985  # z-score for 95% confidence

# The 12 confirmed infraction keys, from statistics_manager.py's
# PENALTY_NAME_DICT (11 keys) + route_timeout (handled separately in
# source but still one of the 12 in practice).
INFRACTION_KEYS = [
    "collisions_layout", "collisions_pedestrian", "collisions_vehicle",
    "red_light", "stop_infraction", "outside_route_lanes",
    "min_speed_infractions", "yield_emergency_vehicle_infractions",
    "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout",
]


# ======================================================================
# Route metadata (scenario_type / town, by position-index)
# ======================================================================


def load_route_metadata():
    tree = ET.parse(ROUTES_METADATA_XML)
    routes = tree.getroot().findall("route")
    if len(routes) != TOTAL_ROUTES:
        raise RuntimeError(f"Expected {TOTAL_ROUTES} routes in {ROUTES_METADATA_XML}, found {len(routes)}")
    metadata = {}
    for idx, r in enumerate(routes):
        scen = r.find("scenarios/scenario")
        metadata[idx] = {
            "town": r.get("town"),
            "scenario_type": scen.get("type") if scen is not None else None,
        }
    return metadata


# ======================================================================
# Statistics helpers
# ======================================================================


def wilson_ci(successes, n, z=Z_95):
    """Wilson score interval for a binomial proportion. Returns (None, None)
    if n is 0 -- can't compute a rate with nothing to compute it from."""
    if not n:
        return None, None
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def mean_std(values):
    """Returns (mean, stdev, n). stdev is None for n<2 -- a 'spread' of one
    number isn't meaningful, shouldn't silently print as 0."""
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n < 2:
        return mean, None, n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance), n


def is_official_success(record):
    """status in {Completed, Perfect} AND every infraction list empty
    except min_speed_infractions -- confirmed rule, established early and
    used consistently throughout this whole project."""
    status = record.get("status") or ""
    if status not in ("Completed", "Perfect"):
        return False
    for key, values in record.get("infractions", {}).items():
        if key == "min_speed_infractions":
            continue
        if values:
            return False
    return True


# ======================================================================
# Per-route result loading
# ======================================================================


def load_route_result(route_idx):
    """Returns a dict of extracted fields, or None if no valid result
    exists yet (route incomplete, still pending, or exhausted retries).
    Mirrors batch_runner.py's result_is_valid() logic (progress check +
    Failed* status handling) since this needs the exact same notion of
    'valid', just extracting fields instead of returning a bool."""
    path = f"{RESULTS_DIR}/bench2drive_{route_idx:02d}_result.json"
    if not Path(path).exists():
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        checkpoint = d["_checkpoint"]
        progress = checkpoint.get("progress", [])
        if len(progress) < 2 or progress[0] < progress[1]:
            return None
        record = checkpoint["records"][0]
        scores = record.get("scores", {})
        if "score_composed" not in scores:
            return None
        infractions = record.get("infractions", {})
        return {
            "route_idx": route_idx,
            "status": record.get("status") or "",
            "official_success": is_official_success(record),
            "driving_score": scores.get("score_composed"),
            "route_completion": scores.get("score_route"),
            "infraction_occurred": {k: bool(infractions.get(k)) for k in INFRACTION_KEYS},
            "duration_game": record.get("meta", {}).get("duration_game"),
            "duration_system": record.get("meta", {}).get("duration_system"),
        }
    except Exception as e:
        print(f"[WARN] couldn't parse result for route {route_idx}: {e}")
        return None


# ======================================================================
# Pipeline / retry statistics from the manifest
#
# Routes can accumulate MORE than MAX_RETRIES total attempts if
# batch_runner.py was stopped and resumed while that route was still
# unresolved -- the retry counter is local to one process_route() call,
# not persisted across script restarts. So this counts ALL manifest
# entries for a route, not just the most recent run's.
# ======================================================================


def load_pipeline_stats():
    stats = defaultdict(lambda: {"total_attempts": 0, "hit_tick_limit": False, "final_outcome": None})
    if not Path(MANIFEST_PATH).exists():
        return stats
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = rec.get("route_idx")
            if idx is None:
                continue
            if rec.get("attempt", 0) >= 1 and "duration_seconds" in rec:
                stats[idx]["total_attempts"] += 1
            if rec.get("reason") == "tick_limit" or rec.get("outcome") == "failed_tick_limit":
                stats[idx]["hit_tick_limit"] = True
            stats[idx]["final_outcome"] = rec.get("outcome")
    return stats


# ======================================================================
# Main aggregation
# ======================================================================


def main():
    route_meta = load_route_metadata()
    pipeline_stats = load_pipeline_stats()

    by_scenario = defaultdict(list)
    for idx in range(TOTAL_ROUTES):
        meta = route_meta[idx]
        result = load_route_result(idx)
        by_scenario[meta["scenario_type"]].append(
            {"route_idx": idx, "town": meta["town"], "result": result, "pipeline": pipeline_stats.get(idx, {})}
        )

    rows = []
    all_valid_routes = []  # for the overall/marginal summary

    for scenario_type in sorted(by_scenario):
        routes = by_scenario[scenario_type]
        valid = [r for r in routes if r["result"] is not None]
        all_valid_routes.extend(valid)
        n_valid = len(valid)
        n_total = len(routes)  # always 5, structurally

        n_success = sum(1 for r in valid if r["result"]["official_success"])
        success_rate = n_success / n_total if n_total else None
        succ_ci_lo, succ_ci_hi = wilson_ci(n_success, n_total)

        all_scores = [r["result"]["driving_score"] for r in valid if r["result"]["driving_score"] is not None]
        succ_scores = [r["result"]["driving_score"] for r in valid if r["result"]["official_success"]]
        nonsucc_scores = [r["result"]["driving_score"] for r in valid if not r["result"]["official_success"]]

        row = {
            "scenario_type": scenario_type,
            "n_valid": n_valid,
            "n_total": n_total,
            "success_rate": success_rate,
            "success_ci_low": succ_ci_lo,
            "success_ci_high": succ_ci_hi,
            "success_ci_width": (succ_ci_hi - succ_ci_lo) if succ_ci_lo is not None else None,
        }

        for label, scores in (("all", all_scores), ("successful", succ_scores), ("nonsuccessful", nonsucc_scores)):
            mean, std, n = mean_std(scores)
            row[f"driving_score_{label}_mean"] = mean
            row[f"driving_score_{label}_std"] = std
            row[f"driving_score_{label}_n"] = n

        # Infraction occurrence rate -- denominator is n_valid, NOT n_total.
        # Deliberately different rule from success rate: see module-level
        # notes in the handoff message -- an incomplete route's infraction
        # status is unknown, not "didn't happen".
        for key in INFRACTION_KEYS:
            occurred = sum(1 for r in valid if r["result"]["infraction_occurred"][key])
            rate = occurred / n_valid if n_valid else None
            ci_lo, ci_hi = wilson_ci(occurred, n_valid) if n_valid else (None, None)
            row[f"infr_{key}_rate"] = rate
            row[f"infr_{key}_ci_low"] = ci_lo
            row[f"infr_{key}_ci_high"] = ci_hi

        row["total_attempts_sum"] = sum(r["pipeline"].get("total_attempts", 0) for r in routes)
        row["routes_hit_tick_limit"] = sum(1 for r in routes if r["pipeline"].get("hit_tick_limit"))
        row["routes_needs_manual_review"] = sum(
            1 for r in routes if r["pipeline"].get("final_outcome") == "needs_manual_review"
        )

        rows.append(row)

    # ---- write the per-scenario-type CSV ----
    if rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # ---- overall summary ----
    summary_lines = []
    total_valid = len(all_valid_routes)
    summary_lines.append(f"Total valid routes: {total_valid} / {TOTAL_ROUTES}")

    overall_success = sum(1 for r in all_valid_routes if r["result"]["official_success"])
    overall_rate = overall_success / TOTAL_ROUTES  # denominator always 220, same rule as per-scenario
    ov_lo, ov_hi = wilson_ci(overall_success, TOTAL_ROUTES)
    summary_lines.append(
        f"Overall success rate: {overall_rate:.1%} ({overall_success}/{TOTAL_ROUTES}), "
        f"95% CI [{ov_lo:.1%}, {ov_hi:.1%}]" if ov_lo is not None else "Overall success rate: n/a"
    )

    all_ds = [r["result"]["driving_score"] for r in all_valid_routes if r["result"]["driving_score"] is not None]
    mean, std, n = mean_std(all_ds)
    if mean is not None:
        summary_lines.append(f"Overall driving score: mean={mean:.1f}" + (f" std={std:.1f}" if std else "") + f" (n={n})")

    summary_lines.append("\n--- Marginal infraction rates (across all valid routes, all scenario types) ---")
    for key in INFRACTION_KEYS:
        occurred = sum(1 for r in all_valid_routes if r["result"]["infraction_occurred"][key])
        rate = occurred / total_valid if total_valid else None
        if rate is not None:
            summary_lines.append(f"  {key}: {rate:.1%} ({occurred}/{total_valid})")

    summary_lines.append("\n--- Worst 10 scenario types by success rate (min 1 valid route) ---")
    ranked = sorted((r for r in rows if r["n_valid"] > 0), key=lambda r: r["success_rate"])
    for r in ranked[:10]:
        ci = f"[{r['success_ci_low']:.0%}, {r['success_ci_high']:.0%}]" if r["success_ci_low"] is not None else "n/a"
        summary_lines.append(f"  {r['scenario_type']}: {r['success_rate']:.0%} (n_valid={r['n_valid']}/5, CI {ci})")

    summary_lines.append("\n--- Scenario types with widest success-rate CI (least reliable estimate) ---")
    wide = sorted((r for r in rows if r["success_ci_width"] is not None), key=lambda r: -r["success_ci_width"])
    for r in wide[:10]:
        summary_lines.append(
            f"  {r['scenario_type']}: CI width={r['success_ci_width']:.0%} (n_valid={r['n_valid']}/5)"
        )

    summary_lines.append("\n--- Pipeline stats ---")
    total_attempts = sum(r["total_attempts_sum"] for r in rows)
    total_tick_limited = sum(r["routes_hit_tick_limit"] for r in rows)
    total_manual_review = sum(r["routes_needs_manual_review"] for r in rows)
    summary_lines.append(f"  Total attempts across all routes: {total_attempts}")
    summary_lines.append(f"  Routes that hit the tick limit at some point: {total_tick_limited}")
    summary_lines.append(f"  Routes currently needs_manual_review: {total_manual_review}")

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Wrote {len(rows)} scenario-type rows to {OUTPUT_CSV}")
    print(f"Wrote summary to {OUTPUT_SUMMARY}")
    print()
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()