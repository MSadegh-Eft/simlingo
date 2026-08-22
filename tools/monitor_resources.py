#!/usr/bin/env python3
"""
monitor_resources.py -- tracks PEAK resource usage while a route runs,
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

Run this in a second pane while a route is actually executing, pointed at
the SAME GPU index the run is using:
    pane 1: cd /data/ghazaleh/simlingo/tools && python3 batch_runner.py --only 4 --gpu 2
    pane 2: python3 monitor_resources.py --gpu 2

Ctrl+C to stop and print the peak summary.
"""

import argparse
import subprocess
import time


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=5.0, help="Sample interval, seconds")
    parser.add_argument("--gpu", type=int, required=True, help="GPU index to watch (whole-GPU, not per-process)")
    args = parser.parse_args()

    peaks = {"carla_rss_mb": 0.0, "eval_rss_mb": 0.0, "gpu_mem_mb": 0.0, "gpu_util_pct": 0.0}
    util_samples = []
    print(f"Sampling every {args.interval}s, watching GPU {args.gpu} -- Ctrl+C to stop and see peak summary\n")

    try:
        while True:
            carla_pid = get_pid("CarlaUE4-Linux-Shipping")
            eval_pid = get_pid("leaderboard_evaluator.py")
            gpu_mem, gpu_util = gpu_stats(args.gpu)

            line_parts = []

            if carla_pid:
                r = rss_mb(carla_pid)
                if r is not None:
                    peaks["carla_rss_mb"] = max(peaks["carla_rss_mb"], r)
                line_parts.append(f"CARLA(pid={carla_pid}) RAM={r or '?'}MB")
            else:
                line_parts.append("CARLA: not running")

            if eval_pid:
                r = rss_mb(eval_pid)
                if r is not None:
                    peaks["eval_rss_mb"] = max(peaks["eval_rss_mb"], r)
                line_parts.append(f"eval(pid={eval_pid}) RAM={r or '?'}MB")
            else:
                line_parts.append("eval: not running")

            if gpu_mem is not None:
                peaks["gpu_mem_mb"] = max(peaks["gpu_mem_mb"], gpu_mem)
            if gpu_util is not None:
                peaks["gpu_util_pct"] = max(peaks["gpu_util_pct"], gpu_util)
                util_samples.append(gpu_util)
            line_parts.append(
                f"GPU {args.gpu}: mem={gpu_mem or '?'}MB util={gpu_util if gpu_util is not None else '?'}% "
                f"(whole-GPU, not per-process)"
            )

            print(" | ".join(line_parts))
            time.sleep(args.interval)

    except KeyboardInterrupt:
        avg_util = sum(util_samples) / len(util_samples) if util_samples else 0.0
        print("\n=== Peak usage observed this session ===")
        print(f"CarlaUE4 system RAM (RSS):    {peaks['carla_rss_mb']:.0f} MB")
        print(f"eval process system RAM:      {peaks['eval_rss_mb']:.0f} MB")
        print(f"GPU {args.gpu} peak memory used: {peaks['gpu_mem_mb']:.0f} MB (whole-GPU, includes any other jobs on this index during the run)")
        print(f"GPU {args.gpu} peak utilization: {peaks['gpu_util_pct']:.0f}%")
        print(f"GPU {args.gpu} average utilization over {len(util_samples)} samples: {avg_util:.0f}%")
        print("(Peak alone doesn't tell you if that was sustained or a brief spike --")
        print(" the average across the run is what actually informs GPU_UTIL_THRESHOLD.)")


if __name__ == "__main__":
    main()