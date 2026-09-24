#!/usr/bin/env python3
"""
monitor_resources.py -- tracks PEAK resource usage while the eval runs,
not a before/after snapshot (which is unreliable on a shared box -- see
usage notes below).

System RAM (RSS) IS attributed per-process correctly, since `ps` sees our
own process tree fine regardless of anything else on the box.

GPU memory is whole-GPU, not per-process -- confirmed on this specific
machine that nvidia-smi's per-process queries don't work at all (every
entry shows process_name=[Not Found], PIDs that don't match ours, and the
standard `nvidia-smi` table's own Processes section comes back empty even
while CARLA and the eval process are both confirmed running and using
memory -- looks like PID-namespace isolation from however GPUs are shared
across users here). So GPU numbers below are contaminated by whatever else
is running on the same index during the sampling window -- same caveat as
any aggregate nvidia-smi read on a shared box, just made explicit rather
than presented as more precise than it is.

PERSISTENCE -- built for multi-day unattended runs on a box that can
crash or reboot (it already did once mid-run):
  --csv PATH    every sample is appended to PATH as a timestamped CSV row.
                Raw time series -- survives Ctrl+C, crashes and reboots,
                and is the input for the supervisor summary.
  --peaks PATH  the running peaks are rewritten to PATH every
                --snapshot-every samples (default 12 = ~1 min at 5 s) so
                the peak numbers survive a crash/reboot. Defaults to
                <csv>_peaks.json when --csv is given.
  --summary     do not sample -- read the CSV back and print + write
                <csv>_summary.txt: per-metric peaks/averages, monitored
                duration, busy fractions and the whole-GPU caveat.
                This is the report to hand to the lab supervisor.

CPU% note: `ps -o %cpu=` is per-process and is the process's LIFETIME
average, not an instantaneous value -- good enough for "how much CPU does
CARLA/eval need" reporting; stated again in the summary output.

Usage while the batch runs:
    pane 1: cd /data/ghazaleh/simlingo && python3 -u tools/batch_runner.py --gpu 2
    pane 2: python3 tools/monitor_resources.py --gpu 2 --csv eval220v2/resource_samples.csv

Ctrl+C stops sampling and prints the peak summary (the CSV keeps everything).
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone

CSV_FIELDS = [
    "timestamp_iso", "unix_ts", "gpu_index",
    "carla_rss_mb", "carla_cpu_pct",
    "eval_rss_mb", "eval_cpu_pct",
    "gpu_mem_mb", "gpu_util_pct",
    "disk_free_mb",
]

# the volume whose free space matters for the run outputs
WATCH_DIR = "/data/ghazaleh/simlingo"


def get_pid(pattern):
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def rss_mb(pid):
    """System RAM (resident set size) for a PID, in MB. This one IS
    correctly per-process -- no namespace issue for our own process tree."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
        return int(out.strip()) / 1024
    except Exception:
        return None


def cpu_pct(pid):
    """CPU% for a PID from ps -- the process's LIFETIME average, not an
    instantaneous value (fine for capacity-reporting purposes)."""
    try:
        out = subprocess.check_output(["ps", "-o", "%cpu=", "-p", str(pid)], text=True)
        return float(out.strip())
    except Exception:
        return None


def gpu_stats(gpu_index):
    """Whole-GPU memory used and utilization%, in one query. Whole-GPU, not
    per-process -- see module docstring for why (confirmed, not assumed)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return None, None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and int(parts[0]) == gpu_index:
            return float(parts[1]), float(parts[2])
    return None, None


def write_peaks(path, peaks, meta):
    """Atomic rewrite (tmp + rename) so a crash/reboot never leaves a
    half-written snapshot behind."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({**meta, "peaks": peaks}, f, indent=2)
    os.replace(tmp, path)

def build_summary_text(csv_path):
    """Reads back the CSV written by sampling mode and returns the
    supervisor report as a multi-line string (None if no usable rows)."""
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if any((r.get(k) or "").strip() for k in CSV_FIELDS)]
    if not rows:
        return None

    def fnum(row, key):
        try:
            return float(row[key])
        except (TypeError, ValueError, KeyError):
            return None

    metric_names = CSV_FIELDS[3:]  # carla_rss_mb .. gpu_util_pct
    stats = {}
    for key in metric_names:
        vals = [v for v in (fnum(r, key) for r in rows) if v is not None]
        stats[key] = {
            "peak": max(vals) if vals else None,
            "avg": (sum(vals) / len(vals)) if vals else None,
            "n": len(vals),
        }

    stamps = [s for s in (fnum(r, "unix_ts") for r in rows) if s is not None]
    duration_s = (max(stamps) - min(stamps)) if stamps else None
    gpu_idx = next((r["gpu_index"] for r in reversed(rows) if (r.get("gpu_index") or "").strip()), "?")
    carla_present = sum(1 for r in rows if fnum(r, "carla_rss_mb") is not None)
    eval_present = sum(1 for r in rows if fnum(r, "eval_rss_mb") is not None)
    disk_vals = [v for v in (fnum(r, "disk_free_mb") for r in rows) if v is not None]
    disk_min = min(disk_vals) if disk_vals else None
    disk_last = disk_vals[-1] if disk_vals else None

    def fmt(v, nd=0, suffix=""):
        return f"{v:.{nd}f}{suffix}" if v is not None else "n/a"

    ram_parts = [stats["carla_rss_mb"]["peak"], stats["eval_rss_mb"]["peak"]]
    combined_ram_peak = sum(p for p in ram_parts if p is not None) if any(p is not None for p in ram_parts) else None

    dur_text = "n/a"
    if duration_s is not None:
        dur_text = f"{int(duration_s // 3600)}h {int((duration_s % 3600) // 60)}m {int(duration_s % 60)}s"

    lines = [
        "=" * 70,
        "Resource usage report -- SimLingo eval run (for the lab supervisor)",
        "=" * 70,
        f"CSV source        : {csv_path}",
        f"Samples           : {len(rows)}",
        f"Monitored duration: {dur_text}",
        f"GPU watched       : {gpu_idx}",
        "",
        "--- GPU (WHOLE-GPU readings for the watched index) ---",
        f"Peak memory used  : {fmt(stats['gpu_mem_mb']['peak'])} MB",
        f"Avg memory used   : {fmt(stats['gpu_mem_mb']['avg'], 0, ' MB')}  (over {stats['gpu_mem_mb']['n']} samples)",
        f"Peak utilization  : {fmt(stats['gpu_util_pct']['peak'], 0, '%')}",
        f"Avg utilization   : {fmt(stats['gpu_util_pct']['avg'], 0, '%')}",
        "",
        "--- CARLA server (per-process) ---",
        f"Peak system RAM   : {fmt(stats['carla_rss_mb']['peak'])} MB",
        f"CPU (lifetime avg): {fmt(stats['carla_cpu_pct']['avg'], 0, '% of one core')}",
        f"Present in        : {carla_present}/{len(rows)} samples",
        "",
        "--- leaderboard_evaluator process (includes the torch model) ---",
        f"Peak system RAM   : {fmt(stats['eval_rss_mb']['peak'])} MB",
        f"CPU (lifetime avg): {fmt(stats['eval_cpu_pct']['avg'], 0, '% of one core')}",
        f"Present in        : {eval_present}/{len(rows)} samples",
        "",
        f"Peak combined system RAM (CARLA + eval): {fmt(combined_ram_peak)} MB "
        f"(~{fmt((combined_ram_peak or 0) / 1024, 1, ' GB')})",
        "",
        "--- Disk free on the volume holding the run outputs ---",
        f"Minimum free seen: {fmt(disk_min)} MB   <-- the dangerous number",
        f"Latest free      : {fmt(disk_last)} MB",
        "",
        "Caveats:",
        " * GPU memory/utilization are WHOLE-GPU for the watched index -- on",
        "   this shared machine per-process GPU attribution is unavailable",
        "   (nvidia-smi shows no per-process entries), so other users' jobs on",
        "   the same index are included in these numbers.",
        " * CPU% is each process's lifetime average from `ps`, not an",
        "   instantaneous reading.",
        " * RSS (system RAM) is per-process and reliable.",
    ]
    return "\n".join(lines)


def run_sampling(args):
    peaks = {k: 0.0 for k in ("carla_rss_mb", "eval_rss_mb", "carla_cpu_pct",
                              "eval_cpu_pct", "gpu_mem_mb", "gpu_util_pct")}
    util_samples = []
    csv_path = args.csv
    peaks_path = args.peaks or (f"{csv_path}_peaks.json" if csv_path else None)

    if csv_path and not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDS)

    print(f"Sampling every {args.interval}s, watching GPU {args.gpu} "
          f"-- Ctrl+C to stop and see the peak summary")
    if csv_path:
        print(f"CSV time series : {csv_path}")
        print(f"Peaks snapshot  : {peaks_path} (rewritten every {args.snapshot_every} samples)")
    print()

    sample_no = 0
    try:
        while True:
            now = time.time()
            iso = datetime.now(timezone.utc).isoformat()

            carla_pid = get_pid("CarlaUE4-Linux-Shipping")
            eval_pid = get_pid("leaderboard_evaluator.py")
            carla_rss = rss_mb(carla_pid) if carla_pid else None
            carla_cpu = cpu_pct(carla_pid) if carla_pid else None
            eval_rss = rss_mb(eval_pid) if eval_pid else None
            eval_cpu = cpu_pct(eval_pid) if eval_pid else None
            gpu_mem, gpu_util = gpu_stats(args.gpu)
            disk_free_mb = shutil.disk_usage(WATCH_DIR).free // (1024 * 1024)

            row = {
                "timestamp_iso": iso,
                "unix_ts": f"{now:.0f}",
                "gpu_index": args.gpu,
                "carla_rss_mb": f"{carla_rss:.1f}" if carla_rss is not None else "",
                "carla_cpu_pct": f"{carla_cpu:.1f}" if carla_cpu is not None else "",
                "eval_rss_mb": f"{eval_rss:.1f}" if eval_rss is not None else "",
                "eval_cpu_pct": f"{eval_cpu:.1f}" if eval_cpu is not None else "",
                "gpu_mem_mb": f"{gpu_mem:.0f}" if gpu_mem is not None else "",
                "gpu_util_pct": f"{gpu_util:.0f}" if gpu_util is not None else "",
                "disk_free_mb": str(disk_free_mb),
            }

            # update peaks (only from observed values)
            for key in ("carla_rss_mb", "eval_rss_mb", "carla_cpu_pct",
                        "eval_cpu_pct", "gpu_mem_mb", "gpu_util_pct"):
                v = row[key]
                if v != "":
                    peaks[key] = max(peaks[key], float(v))
            if gpu_util is not None:
                util_samples.append(gpu_util)

            if csv_path:
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([row[k] for k in CSV_FIELDS])
            if csv_path and sample_no % args.snapshot_every == 0:
                write_peaks(peaks_path, peaks, {
                    "gpu_index": args.gpu,
                    "csv": os.path.abspath(csv_path),
                    "samples": sample_no + 1,
                })

            line_parts = []
            if carla_pid:
                line_parts.append(f"CARLA(pid={carla_pid}) RAM={row['carla_rss_mb'] or '?'}MB CPU={row['carla_cpu_pct'] or '?'}%")
            else:
                line_parts.append("CARLA: not running")
            if eval_pid:
                line_parts.append(f"eval(pid={eval_pid}) RAM={row['eval_rss_mb'] or '?'}MB CPU={row['eval_cpu_pct'] or '?'}%")
            else:
                line_parts.append("eval: not running")
            line_parts.append(
                f"GPU {args.gpu}: mem={row['gpu_mem_mb'] or '?'}MB util={row['gpu_util_pct'] or '?'}% "
                f"(whole-GPU, not per-process)"
            )
            print(" | ".join(line_parts), flush=True)

            sample_no += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        avg_util = sum(util_samples) / len(util_samples) if util_samples else 0.0
        if csv_path:
            write_peaks(peaks_path, peaks, {
                "gpu_index": args.gpu,
                "csv": os.path.abspath(csv_path),
                "samples": sample_no,
            })
        print("\n=== Peak usage observed this session ===")
        print(f"CarlaUE4 system RAM (RSS):    {peaks['carla_rss_mb']:.0f} MB")
        print(f"eval process system RAM:      {peaks['eval_rss_mb']:.0f} MB")
        print(f"CarlaUE4 CPU (lifetime avg):  {peaks['carla_cpu_pct']:.0f}% of one core")
        print(f"eval process CPU (lifetime):  {peaks['eval_cpu_pct']:.0f}% of one core")
        print(f"GPU {args.gpu} peak memory used: {peaks['gpu_mem_mb']:.0f} MB (whole-GPU, includes any other jobs on this index during the run)")
        print(f"GPU {args.gpu} peak utilization: {peaks['gpu_util_pct']:.0f}%")
        print(f"GPU {args.gpu} average utilization over {len(util_samples)} samples: {avg_util:.0f}%")
        print("(Peak alone doesn't tell you if that was sustained or a brief spike --")
        print(" the average across the run is what actually informs GPU_UTIL_THRESHOLD.)")
        if csv_path:
            print(f"\nTime series kept in : {os.path.abspath(csv_path)}")
            if peaks_path:
                print(f"Peaks snapshot in   : {os.path.abspath(peaks_path)}")
            print(f"Supervisor report   : python3 tools/monitor_resources.py --summary --csv {os.path.abspath(csv_path)}")

def main():
    parser = argparse.ArgumentParser(description="Peak resource monitor for the eval run")
    parser.add_argument("--interval", type=float, default=5.0, help="Sample interval, seconds")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to watch (required for sampling; whole-GPU, not per-process)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Append every sample to this CSV (recommended -- survives crashes/reboots)")
    parser.add_argument("--peaks", type=str, default=None,
                        help="Path for the running peaks JSON (default: <csv>_peaks.json)")
    parser.add_argument("--snapshot-every", type=int, default=12,
                        help="Rewrite the peaks JSON every N samples (default 12 = ~1 min at 5 s)")
    parser.add_argument("--summary", action="store_true",
                        help="Do not sample -- build the supervisor report from --csv")
    args = parser.parse_args()

    if args.summary:
        if not args.csv:
            parser.error("--summary requires --csv PATH (the recorded samples)")
        text = build_summary_text(args.csv)
        if text is None:
            print(f"No usable rows in {args.csv} -- nothing to summarize")
            return
        print(text)
        out_path = f"{args.csv}_summary.txt"
        with open(out_path, "w") as f:
            f.write(text + "\n")
        print(f"\n(also written to {out_path})")
        return

    if args.gpu is None:
        parser.error("--gpu is required for sampling (or use --summary --csv PATH)")
    run_sampling(args)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()