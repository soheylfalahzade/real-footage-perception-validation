"""UA-DETRAC adapter: human-annotated boxes and native weather labels.

``rfv-detrac`` turns a local copy of UA-DETRAC (image folders and XML annotations) into 25 fps
clips with a ``<clip>.gt.json`` of human-labelled vehicle boxes. ``rfv-run --extra-dir`` then
uses those boxes as ground truth instead of the detector's own clear-frame detections.
Nothing is downloaded or redistributed.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from relmon.detector import Box

WIDTH = 640
FPS = 25.0
WEATHER = ("sunny", "cloudy", "rainy", "night")
Rect = tuple[float, float, float, float]


def parse_sequence_xml(path: str | Path) -> tuple[str, list[Rect], dict[int, list[Rect]]]:
    """Weather label, ignored regions and per-frame vehicle boxes (1-based frame numbers)."""
    root = ET.parse(path).getroot()
    weather = ""
    attr = root.find("sequence_attribute")
    if attr is not None:
        for key in ("sence_weather", "scene_weather", "weather"):   # the dataset spells it "sence"
            if attr.get(key):
                weather = str(attr.get(key)).strip().lower()
                break

    def rect(b: ET.Element) -> Rect:
        left, top = float(b.get("left", 0)), float(b.get("top", 0))
        return (left, top, left + float(b.get("width", 0)), top + float(b.get("height", 0)))

    ignored = [rect(b) for b in root.findall("ignored_region/box")]
    frames: dict[int, list[Rect]] = {}
    for fr in root.findall("frame"):
        boxes = [rect(b) for tg in fr.findall("target_list/target")
                 if (b := tg.find("box")) is not None]
        frames[int(fr.get("num", 0))] = boxes
    return weather, ignored, frames


def keep_box(b: Rect, ignored: list[Rect], min_h: float) -> bool:
    """Drop boxes that are too small to be a fair detection target or lie in an ignored region."""
    if b[3] - b[1] < min_h:
        return False
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return not any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in ignored)


def index_image_dirs(root: str | Path) -> dict[str, Path]:
    """Sequence name -> image folder, found by walking ``root`` (stops descending at ``MVI_*``)."""
    found: dict[str, Path] = {}
    for dirpath, dirs, _ in os.walk(root):
        for d in list(dirs):
            if d.startswith("MVI_"):
                found.setdefault(d, Path(dirpath) / d)
                dirs.remove(d)
    return found


def write_clip(seq_dir: Path, out_mp4: Path, n_frames: int) -> tuple[int, int]:
    """Encode the first ``n_frames`` images as a 640-wide mp4; returns the original (w, h)."""
    first = cv2.imread(str(seq_dir / "img00001.jpg"))
    if first is None:
        raise FileNotFoundError(f"no img00001.jpg in {seq_dir}")
    h0, w0 = first.shape[:2]
    h = int(round(h0 * WIDTH / w0))
    wr = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
                         FPS, (WIDTH, h))
    for i in range(1, n_frames + 1):
        img = cv2.imread(str(seq_dir / f"img{i:05d}.jpg"))
        if img is None:
            break
        wr.write(cv2.resize(img, (WIDTH, h), interpolation=cv2.INTER_AREA))
    wr.release()
    return w0, h0


def prepare_clips(root: str | Path, out: str | Path, per_weather: int = 3, seed: int = 0,
                  n_frames: int = 600, min_h: float = 20.0) -> list[dict[str, Any]]:
    """Pick ``per_weather`` sequences of each weather class and write clips + ground truth."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    images = index_image_dirs(root)
    by_weather: dict[str, list[tuple[str, Path]]] = {w: [] for w in WEATHER}
    for xml in sorted(root.rglob("MVI_*.xml")):
        name = xml.stem
        if name not in images:
            continue
        weather, _, frames = parse_sequence_xml(xml)
        if weather in by_weather and len(frames) >= n_frames:
            by_weather[weather].append((name, xml))
    rng = random.Random(seed)
    manifest: list[dict[str, Any]] = []
    for weather, items in by_weather.items():
        for name, xml in rng.sample(items, min(per_weather, len(items))):
            _, ignored, frames = parse_sequence_xml(xml)
            w0, h0 = write_clip(images[name], out / f"{name}.mp4", n_frames)
            s = WIDTH / w0
            gt = {str(n - 1): [[round(c * s, 1) for c in b] for b in boxes
                               if keep_box(tuple(c * s for c in b),                # type: ignore[arg-type]
                                           [tuple(c * s for c in r) for r in ignored], min_h)]  # type: ignore[misc]
                  for n, boxes in frames.items() if n <= n_frames}
            (out / f"{name}.gt.json").write_text(
                json.dumps({"weather": weather, "width": WIDTH, "orig_size": [w0, h0],
                            "min_box_height_px": min_h, "frames": gt}), encoding="utf-8")
            manifest.append({"clip": f"{name}.mp4", "weather": weather,
                             "mean_gt_per_frame": float(np.mean([len(v) for v in gt.values()]))})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def gt_for_segment(path: str | Path, start: int, stride: int, count: int
                   ) -> tuple[list[list[Box]], str]:
    """Human ground-truth boxes for the sampled frames of one segment, and the clip's weather."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data["frames"]
    gt = [[Box(b[0], b[1], b[2], b[3], 1.0) for b in frames.get(str(start + i * stride), [])]
          for i in range(count)]
    return gt, str(data.get("weather", ""))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``rfv-detrac``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="folder holding UA-DETRAC images and XML files")
    ap.add_argument("--out", default="clips")
    ap.add_argument("--per-weather", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-height", type=float, default=20.0,
                    help="smallest ground-truth box height in pixels (at 640 px width)")
    args = ap.parse_args(argv)
    for m in prepare_clips(args.root, args.out, args.per_weather, args.seed, min_h=args.min_height):
        print(f"{m['clip']}: {m['weather']}, {m['mean_gt_per_frame']:.1f} GT boxes per frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
