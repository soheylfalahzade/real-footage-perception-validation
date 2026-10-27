"""Resource check before heavy runs: free memory, CPU load, and a short timing benchmark."""
from __future__ import annotations

import argparse
import os
import time

from rfv.data import VIDEOS, fetch, load_frames
from rfv.detector import DarknetDetector


def free_ram_gb() -> float:
    with open("/proc/meminfo", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MemAvailable"):
                return float(line.split()[1]) / 1e6
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    """Print resource status and an estimate of the full run time (CPU only, one thread)."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--total-frames", type=int, default=2900, help="frames of a full run")
    args = ap.parse_args(argv)
    ram, cores = free_ram_gb(), os.cpu_count() or 1
    load = os.getloadavg()[0]
    print(f"available RAM : {ram:.1f} GB   (needs about 1.0 GB)")
    print(f"CPU load (1m) : {load:.1f} on {cores} logical cores   (run uses one core)")
    fetch(args.data_dir)
    det = DarknetDetector(args.data_dir)
    frames = load_frames(f"{args.data_dir}/{VIDEOS[0]}", count=args.frames)
    t0 = time.perf_counter()
    for f in frames:
        det(f)
    ms = (time.perf_counter() - t0) / max(len(frames), 1) * 1e3
    est = ms * args.total_frames / 6e4
    print(f"detector speed: {ms:.0f} ms/frame -> full run about {est:.0f} min")
    ok = ram >= 1.5 and load < 0.85 * cores
    msg = ("OK to run (use `nice -n 19`)" if ok else
           "WAIT: memory or CPU is tight; use --max-streams to run a subset")
    print("verdict       :", msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
