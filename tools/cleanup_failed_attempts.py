#!/usr/bin/env python3
"""
cleanup_failed_attempts.py -- reclaims disk from viz/ output of FAILED
attempts, which nothing downstream needs:

  - metrics: run_official_metrics.py keys each route's metric_info.json to
    the save_name recorded in its VALID result json -- failed attempts are
    never referenced.
  - videos: build_videos_220.sh picks the most recently modified images/
    dir; with failed attempts removed, the one that remains is exactly the
    attempt that produced the result.
  - debugging: the eval logs under eval220v2/logs/ (kept) contain the
    failure reasons; images of a crashed attempt add nothing.

Default is a DRY RUN: prints what would be deleted and the reclaimable
size per route. Pass --delete to actually remove directories.

Kept per route (never deleted):
  - the save_name dir referenced by a VALID result json (progress complete,
    score_composed present, status not Failed*)
"""

import argparse
import json
import shutil
from pathlib import Path

WORK_DIR = "/data/ghazaleh/simlingo"
RESULTS_DIR = f"{WORK_DIR}/eval220v2/results"
VIZ_DIR = f"{WORK_DIR}/eval220v2/viz"
TOTAL_ROUTES = 220


def valid_save_name(result_path: Path):
    """save_name of the route's valid result, else None. Same validity rule
    as batch_runner / build_matrix / run_official_metrics."""
    try:
        with open(result_path) as f:
            d = json.load(f)
        checkpoint = d["_checkpoint"]
        progress = checkpoint.get("progress", [])
        if len(progress) < 2 or progress[0] < progress[1]:
            return None
        record = checkpoint["records"][0]
        if record.get("status", "").startswith("Failed"):
            return None
        if "score_composed" not in record.get("scores", {}):
            return None
        return record.get("save_name")
    except Exception:
        return None


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Reclaim disk from failed-attempt viz output (dry run by default)")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete (default: dry run)")
    args = parser.parse_args()

    total_reclaim = 0
    for idx in range(TOTAL_ROUTES):
        route_dir = Path(VIZ_DIR) / f"bench2drive_{idx:02d}"
        if not route_dir.is_dir():
            continue
        keep = valid_save_name(Path(RESULTS_DIR) / f"bench2drive_{idx:02d}_result.json")
        delete_dirs = []
        for child in sorted(route_dir.iterdir()):
            if not child.is_dir():
                continue
            if keep is not None and child.name == keep:
                continue
            delete_dirs.append(child)
        if not delete_dirs:
            continue
        reclaim = sum(dir_size(d) for d in delete_dirs)
        total_reclaim += reclaim
        keep_note = f"keep: {keep}" if keep else "KEEP: none (no valid result)"
        print(f"[{idx:03d}] reclaim {reclaim / (1024 ** 3):.2f} GB "
              f"({len(delete_dirs)} dir(s)) | {keep_note}", flush=True)
        if args.delete:
            for d in delete_dirs:
                shutil.rmtree(d)

    print()
    print(f"Total reclaimable: {total_reclaim / (1024 ** 3):.2f} GB")
    if not args.delete:
        print("DRY RUN -- re-run with --delete to remove these directories.")
    else:
        print("Deleted.")


if __name__ == "__main__":
    main()
