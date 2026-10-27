"""Run an off-the-shelf detector on real footage under synthetic degradations of the real frames."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from relmon.degrade import Degrader
from relmon.detector import Box, match_recall
from relmon.health import DetectorHealth
from relmon.quality import QualityExtractor

from rfv.data import PLAN, VIDEOS, fetch, load_frames
from rfv.detector import DarknetDetector

CONDITIONS: list[tuple[str, float]] = (
    [("clear", 0.0)]
    + [(c, s) for c in ("night", "rain", "fog", "glare", "dirt") for s in (0.5, 1.0)]
    + [("blackout", 1.0)]
)
SEGMENTS = (0, 1)          # two 60-frame segments (stride 3) per video
GT_CONF = 0.5              # pseudo ground truth: clear-frame detections above this score
FEATURES = [f"f{i}" for i in range(10)]


def run_segment(det: DarknetDetector, frames: list[np.ndarray], quality: QualityExtractor,
                ref_count: float, seed: int, conditions: list[tuple[str, float]],
                health_window: int = 20) -> list[dict[str, float | int | str]]:
    """Per-frame records for every condition of one segment (clear must come first)."""
    h, w = frames[0].shape[:2]
    light = np.zeros((h, w, 3), np.float32)
    rows: list[dict[str, float | int | str]] = []
    gt: list[list[Box]] = []
    for cond, sev in conditions:
        deg = Degrader(seed=seed, h=h, w=w)
        health = DetectorHealth(window=health_window, reference_count=ref_count)
        quality.reset_obstruction()
        for i, f in enumerate(frames):
            img = deg.apply(f, light, cond, sev)
            boxes = det(img)
            if cond == "clear":
                gt.append([b for b in boxes if b.score >= GT_CONF])
            feats = np.concatenate([quality.extract(img), health.update(boxes)])
            tp, n_gt, n_det = match_recall([(b.x1, b.y1, b.x2, b.y2) for b in gt[i]], boxes,
                                           thr=0.5)
            rows.append({"cond": cond, "sev": sev, "frame": i, "tp": tp, "n_gt": n_gt,
                         "n_det": n_det, **{f"f{k}": float(v) for k, v in enumerate(feats)}})
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``rfv-run``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/frames.csv")
    ap.add_argument("--max-streams", type=int, default=0, help="0 = all conditions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    fetch(args.data_dir)
    det = DarknetDetector(args.data_dir)
    conds = CONDITIONS[: args.max_streams] if args.max_streams else CONDITIONS
    cache = Path(args.out).parent / "segments"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts: list[pd.DataFrame] = []
    for video in VIDEOS:
        quality = QualityExtractor()
        ref_count = 1.0
        for seg in SEGMENTS:
            part = cache / f"{Path(video).stem}_seg{seg}_{len(conds)}.csv"
            stride, starts = PLAN[video]
            frames = load_frames(Path(args.data_dir) / video, stride=stride, start=starts[seg],
                                 count=60)
            if seg == 0:
                quality.calibrate(frames[:30])
                ref_count = max(float(np.mean([len(det(f)) for f in frames[:30]])), 0.5)
            if part.exists():
                parts.append(pd.read_csv(part))
                continue
            rows = run_segment(det, frames, quality, ref_count, args.seed, conds)
            seg_df = pd.DataFrame([{"video": video, "seg": seg, **r} for r in rows])
            seg_df.to_csv(part, index=False)
            parts.append(seg_df)
            print(f"{video} seg{seg} done, {time.time() - t0:.0f}s elapsed", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts).to_csv(args.out, index=False)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
