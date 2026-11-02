"""Public sample footage and detector files: fetching with checksum verification, frame loading."""
from __future__ import annotations

import argparse
import hashlib
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

WIDTH = 640  # every frame is resized to this width (aspect ratio preserved)

FILES = {
    "highway-eu.mp4": (
        "https://raw.githubusercontent.com/devmdave/PythonTrafficCongestion/main/video3.mp4",
        "aa6acb40e4468657fc9aaeabb000739f9045b8970af20e9daaa0b2b4bffe9e9d"),
    "highway-us.avi": (
        "https://raw.githubusercontent.com/andrewssobral/vehicle_detection_haarcascades/master/"
        "dataset/video1.avi",
        "a7c7d98162e1ca5a971d29916b47cd15120cfc60f5bc435fb81d8807e254bade"),
    "yolov4-tiny.weights": (
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/"
        "yolov4-tiny.weights",
        "cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87"),
    "yolov4-tiny.cfg": (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "f858e3724962eedf3ac44e3b6cb3f0c3d9ed067c306bb831f539c578b924c90e"),
    "coco.names": (
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
        "634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84"),
}
VIDEOS = ("highway-eu.mp4", "highway-us.avi")
# (frame stride, first source frame of segment 0 and of segment 1); 60 sampled frames per segment
PLAN: dict[str, tuple[int, tuple[int, int]]] = {
    "highway-eu.mp4": (6, (300, 900)),       # 29.97 fps -> 5 fps
    "highway-us.avi": (3, (0, 240)),         # 15 fps -> 5 fps
}


VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
SAMPLE_FPS = 5.0
SEG_FRAMES = 60


def clip_plan(path: str | Path) -> tuple[int, tuple[int, int]] | None:
    """Sampling plan for an arbitrary clip: ~5 fps, two consecutive 60-frame segments.

    Returns ``(stride, (start0, start1))`` or ``None`` when the clip is too short for two segments.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    stride = max(int(round(fps / SAMPLE_FPS)), 1)
    span = SEG_FRAMES * stride
    return (stride, (0, span)) if n >= 2 * span else None


def discover_clips(extra_dir: str | Path | None) -> dict[str, tuple[int, tuple[int, int]]]:
    """User-supplied clips in ``extra_dir`` that are long enough, keyed by file name."""
    out: dict[str, tuple[int, tuple[int, int]]] = {}
    if extra_dir is None or not Path(extra_dir).is_dir():
        return out
    for f in sorted(Path(extra_dir).iterdir()):
        if f.suffix.lower() in VIDEO_EXT:
            plan = clip_plan(f)
            if plan is not None:
                out[f.name] = plan
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, path: Path, tries: int = 4) -> None:
    """Download to a ``.part`` file and rename on success, so an interrupted run leaves no stub."""
    part = path.with_name(path.name + ".part")
    for k in range(tries):
        try:
            urllib.request.urlretrieve(url, part)  # noqa: S310 - fixed https URLs above
            part.replace(path)
            return
        except (OSError, urllib.error.ContentTooShortError):
            part.unlink(missing_ok=True)
            if k == tries - 1:
                raise


def fetch(data_dir: str | Path, videos: bool = True) -> dict[str, Path]:
    """Download missing files and verify their SHA-256 digests; nothing is redistributed.

    With ``videos=False`` only the detector files are fetched (enough when clips are supplied).
    A file that fails its checksum is removed and fetched once more.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (url, digest) in FILES.items():
        if not videos and Path(name).suffix.lower() in VIDEO_EXT:
            continue
        path = root / name
        if not path.exists():
            _download(url, path)
        if sha256(path) != digest:
            path.unlink()
            _download(url, path)
            if sha256(path) != digest:
                raise RuntimeError(f"checksum mismatch for {name}")
        out[name] = path
    return out


def load_frames(video: str | Path, stride: int = 3, start: int = 0, count: int = 60,
                width: int = WIDTH) -> list[np.ndarray]:
    """``count`` frames at ``stride`` from frame ``start``, resized to ``width`` (BGR)."""
    cap = cv2.VideoCapture(str(video))
    frames: list[np.ndarray] = []
    idx = 0
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start and (idx - start) % stride == 0:
            h = int(round(frame.shape[0] * width / frame.shape[1]))
            frames.append(cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA))
        idx += 1
    cap.release()
    return frames


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``rfv-fetch``)."""
    ap = argparse.ArgumentParser(description=fetch.__doc__)
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args(argv)
    for name, path in fetch(args.data_dir).items():
        print(f"{name}: ok ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
