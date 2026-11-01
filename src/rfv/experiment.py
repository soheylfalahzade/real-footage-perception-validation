"""Run an off-the-shelf detector on real footage under synthetic degradations of the real frames."""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from relmon.degrade import Degrader
from relmon.detector import Box, match_recall
from relmon.health import DetectorHealth
from relmon.quality import QualityExtractor

from rfv.data import PLAN, VIDEOS, discover_clips, fetch, load_frames
from rfv.detector import make_detector

CONDITIONS: list[tuple[str, float]] = (
    [("clear", 0.0)]
    + [(c, s) for c in ("night", "rain", "fog", "glare", "dirt") for s in (0.5, 1.0)]
    + [("blackout", 1.0)]
)
SEGMENTS = (0, 1)          # two 60-frame segments (stride 3) per video
GT_CONF = 0.5              # pseudo ground truth: clear-frame detections above this score
FEATURES = [f"f{i}" for i in range(10)]


def run_segment(det: Callable[..., list[Box]], frames: list[np.ndarray], quality: QualityExtractor,
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
    ap.add_argument("--detector", choices=("yolov4-tiny", "yolov8n-onnx"), default="yolov4-tiny")
    ap.add_argument("--extra-dir", default=None,
                    help="folder with additional clips (mp4/avi/mov/mkv, >= 2 x 60 sampled frames)")
    args = ap.parse_args(argv)
    fetch(args.data_dir)
    det = make_detector(args.detector, args.data_dir)
    plan = dict(PLAN)
    plan.update(discover_clips(args.extra_dir))
    videos = list(VIDEOS) + [v for v in plan if v not in VIDEOS]
    tag = "" if args.detector == "yolov4-tiny" else f"_{args.detector}"
    conds = CONDITIONS[: args.max_streams] if args.max_streams else CONDITIONS
    cache = Path(args.out).parent / "segments"
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts: list[pd.DataFrame] = []
    for video in videos:
        quality = QualityExtractor()
        ref_count = 1.0
        for seg in SEGMENTS:
            part = cache / f"{Path(video).stem}_seg{seg}_{len(conds)}{tag}.csv"
            stride, starts = plan[video]
            root = args.data_dir if video in VIDEOS else args.extra_dir
            src = Path(root) / video
            frames = load_frames(src, stride=stride, start=starts[seg],
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
