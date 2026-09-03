#!/usr/bin/env python3
"""
export_route_details.py -- exports one row per route (all 220) with full
detail: status, scores, per-infraction counts + an example message, and
pipeline history (attempts, tick-limit flag, final outcome).

Reuses the same parsing logic as build_matrix.py (progress check, Failed*
handling, is_official_success) so this is consistent with the aggregate
matrix you already have -- not a second, differently-defined source of truth.

Read-only, safe to run any time.

Usage:
    python3 export_route_details.py
Output:
    eval220v2/route_details.csv
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

WORK_DIR = "/data/ghazaleh/simlingo"
ROUTES_METADATA_XML = "/data/ghazaleh/Bench2Drive/leaderboard/data/bench2drive220.xml"
RESULTS_DIR = f"{WORK_DIR}/eval220v2/results"
MANIFEST_PATH = f"{WORK_DIR}/eval220v2/batch_status.jsonl"
OUTPUT_CSV = f"{WORK_DIR}/eval220v2/route_details.csv"
TOTAL_ROUTES = 220

INFRACTION_KEYS = [
    "collisions_layout", "collisions_pedestrian", "collisions_vehicle",
    "red_light", "stop_infraction", "outside_route_lanes",
    "min_speed_infractions", "yield_emergency_vehicle_infractions",
    "scenario_timeouts", "route_dev", "vehicle_blocked", "route_timeout",
]


def load_route_metadata():
    tree = ET.parse(ROUTES_METADATA_XML)
    routes = tree.getroot().findall("route")
    if len(routes) != TOTAL_ROUTES:
        raise RuntimeError(f"Expected {TOTAL_ROUTES} routes, found {len(routes)}")
    metadata = {}
    for idx, r in enumerate(routes):
        scen = r.find("scenarios/scenario")
        metadata[idx] = {
            "town": r.get("town"),
            "scenario_type": scen.get("type") if scen is not None else None,
            "xml_route_id": r.get("id"),
        }
    return metadata


def is_official_success(record):
    status = record.get("status") or ""
    if status not in ("Completed", "Perfect"):
        return False
    for key, values in record.get("infractions", {}).items():
        if key == "min_speed_infractions":
            continue
        if values:
            return False
    return True


def load_pipeline_stats():
    stats = {}
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
            s = stats.setdefault(idx, {"total_attempts": 0, "hit_tick_limit": False, "final_outcome": None})
            if rec.get("attempt", 0) >= 1 and "duration_seconds" in rec:
                s["total_attempts"] += 1
            if rec.get("reason") == "tick_limit" or rec.get("outcome") == "failed_tick_limit":
                s["hit_tick_limit"] = True
            s["final_outcome"] = rec.get("outcome")
    return stats


def load_route_detail(route_idx):
    """Returns a dict with full per-route detail, or None if no valid
    result exists. Mirrors build_matrix.py's validity rule exactly
    (progress check + presence of score_composed)."""
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
        detail = {
            "status": record.get("status") or "",
            "official_success": is_official_success(record),
            "driving_score": scores.get("score_composed"),
            "route_completion": scores.get("score_route"),
            "score_penalty": scores.get("score_penalty"),
            "duration_game": record.get("meta", {}).get("duration_game"),
            "duration_system": record.get("meta", {}).get("duration_system"),
        }
        summary_parts = []
        for key in INFRACTION_KEYS:
            values = infractions.get(key, [])
            detail[f"count_{key}"] = len(values)
            if values:
                example = values[0]
                if len(values) > 1:
                    example += f" (+{len(values)-1} more)"
                summary_parts.append(f"{key}: {example}")
        detail["infraction_summary"] = " | ".join(summary_parts) if summary_parts else "(none)"
        return detail
    except Exception as e:
        print(f"[WARN] couldn't parse result for route {route_idx}: {e}")
        return None


def main():
    route_meta = load_route_metadata()
    pipeline_stats = load_pipeline_stats()

    rows = []
    for idx in range(TOTAL_ROUTES):
        meta = route_meta[idx]
        detail = load_route_detail(idx)
        pipeline = pipeline_stats.get(idx, {})

        row = {
            "route_idx": idx,
            "xml_route_id": meta["xml_route_id"],
            "scenario_type": meta["scenario_type"],
            "town": meta["town"],
            "has_valid_result": detail is not None,
        }
        if detail:
            row.update(detail)
        else:
            row["status"] = ""
            row["official_success"] = False
            for key in ["driving_score", "route_completion", "score_penalty", "duration_game", "duration_system"]:
                row[key] = None
            for key in INFRACTION_KEYS:
                row[f"count_{key}"] = None
            row["infraction_summary"] = "NO VALID RESULT"

        row["pipeline_total_attempts"] = pipeline.get("total_attempts", 0)
        row["pipeline_hit_tick_limit"] = pipeline.get("hit_tick_limit", False)
        row["pipeline_final_outcome"] = pipeline.get("final_outcome", "")

        rows.append(row)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_valid = sum(1 for r in rows if r["has_valid_result"])
    n_success = sum(1 for r in rows if r["official_success"])
    print(f"Wrote {len(rows)} routes to {OUTPUT_CSV}")
    print(f"  valid results: {n_valid}/{TOTAL_ROUTES}")
    print(f"  official successes: {n_success}/{TOTAL_ROUTES}")


if __name__ == "__main__":
    main()