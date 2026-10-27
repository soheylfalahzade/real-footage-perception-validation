"""Animation: the reliability monitor on a real clip that becomes degraded and recovers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from relmon.degrade import Degrader
from relmon.health import DetectorHealth
from relmon.monitor import MonitorConfig, ReliabilityModel, ReliabilityMonitor
from relmon.quality import QualityExtractor

from rfv.analysis import prepare
from rfv.data import PLAN, VIDEOS, fetch, load_frames
from rfv.detector import DarknetDetector


def render_gif(data_dir: str | Path, frames_csv: str | Path, out: str | Path,
               condition: str = "rain", severity: float = 1.0, fps: int = 4) -> dict[str, int]:
    """Clear -> ``condition`` -> clear on the second video; the model is fitted on the first."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Rectangle
    from PIL import Image

    df = prepare(pd.read_csv(frames_csv))
    fcols = [f"f{i}" for i in range(10)]
    train = df[df.video == VIDEOS[0]]
    model = ReliabilityModel().fit(train[fcols].to_numpy(), train["label"].to_numpy())
    paths = fetch(data_dir)
    det = DarknetDetector(data_dir)
    stride, starts = PLAN[VIDEOS[1]]
    frames = load_frames(paths[VIDEOS[1]], stride=stride, start=starts[1], count=60)
    q = QualityExtractor()
    q.calibrate(frames[:30])
    ref = max(float(np.mean([len(det(f)) for f in frames[:30]])), 0.5)
    health = DetectorHealth(window=20, reference_count=ref)
    mon = ReliabilityMonitor(model, MonitorConfig(fps=4.0, dwell_down_s=1.5, dwell_up_s=3.0))
    h, w = frames[0].shape[:2]
    deg = Degrader(seed=0, h=h, w=w)
    light = np.zeros((h, w, 3), np.float32)
    recs: list[dict[str, Any]] = []
    for i, f in enumerate(frames):
        bad = 20 <= i < 40
        img = deg.apply(f, light, condition if bad else "clear", severity if bad else 0.0)
        boxes = det(img)
        r, mode = mon.update(np.concatenate([q.extract(img), health.update(boxes)]))
        recs.append({"img": img, "boxes": boxes, "r": r, "mode": mode.value, "bad": bad})
    colors = {"normal": "#d9f2d9", "degraded": "#fff1c1", "fallback": "#f8c9c9"}
    images = []
    for k, rec in enumerate(recs):
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=72,
                                   gridspec_kw={"width_ratios": [1.3, 1]})
        a.imshow(rec["img"][..., ::-1])
        for d in rec["boxes"]:
            a.add_patch(Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1, fill=False,
                                  ec="#d62728", lw=1.4))
        a.axis("off")
        a.set_title(f"real frame, {condition if rec['bad'] else 'clear'}: "
                    f"{len(rec['boxes'])} detections", fontsize=9)
        t = np.arange(k + 1) / fps
        start = 0
        for i in range(1, k + 2):
            if i == k + 1 or recs[i]["mode"] != recs[start]["mode"]:
                b.axvspan(t[start], t[min(i, k)], color=colors[recs[start]["mode"]], lw=0)
                start = i
        b.plot(t, [x["r"] for x in recs[: k + 1]], color="k", lw=1.6)
        for edge in (20 / fps, 40 / fps):
            b.axvline(edge, color="gray", ls="--", lw=0.8)
        b.set_xlim(0, len(recs) / fps)
        b.set_ylim(0, 1.05)
        b.set_xlabel("time [s]", fontsize=8)
        b.set_ylabel("reliability r(t)", fontsize=8)
        b.set_title(f"mode: {rec['mode']}", fontsize=9)
        fig.tight_layout()
        canvas = cast(FigureCanvasAgg, fig.canvas)
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB")
                      .quantize(colors=96))
        plt.close(fig)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return {"frames": len(images)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``rfv-gif``)."""
    ap = argparse.ArgumentParser(description=render_gif.__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--frames", default="results/frames.csv")
    ap.add_argument("--out", default="docs/assets/real_monitor.gif")
    ap.add_argument("--condition", default="rain")
    ap.add_argument("--severity", type=float, default=1.0)
    args = ap.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(render_gif(args.data_dir, args.frames, args.out, args.condition, args.severity))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
