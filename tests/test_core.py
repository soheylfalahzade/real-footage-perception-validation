import numpy as np
import pandas as pd
from relmon.detector import Box
from relmon.quality import QualityExtractor

from rfv.analysis import KEY, evaluate_split, prepare, retention_table
from rfv.data import WIDTH, load_frames, sha256
from rfv.experiment import run_segment


def test_sha256_and_frame_loading(tmp_path):
    import cv2

    p = tmp_path / "t.bin"
    p.write_bytes(b"abc")
    assert sha256(p).startswith("ba7816bf8f01cfea")
    v = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(v), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    frames = load_frames(v, stride=2, start=2, count=5)
    assert len(frames) == 5 and frames[0].shape == (int(round(48 * WIDTH / 64)), WIDTH, 3)


class StubDetector:
    """Finds a box whenever the image is bright enough (stand-in for a real detector)."""

    def __call__(self, img, conf=None):
        return [Box(100, 100, 200, 180, 0.9)] if img.mean() > 40 else []


def test_run_segment_pseudo_ground_truth_and_retention():
    rng = np.random.default_rng(0)
    frames = [np.clip(rng.normal(120, 20, (360, 640, 3)), 0, 255).astype(np.uint8)
              for _ in range(30)]
    q = QualityExtractor()
    q.calibrate(frames[:10])
    rows = run_segment(StubDetector(), frames, q, 1.0, 0, [("clear", 0.0), ("blackout", 1.0)])
    df = pd.DataFrame(rows)
    clear, black = df[df.cond == "clear"], df[df.cond == "blackout"]
    assert clear["tp"].sum() == clear["n_gt"].sum() == 30
    assert black["tp"].sum() == 0 and black["n_gt"].sum() == 30


def _fake_frames() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for video in ("a.mp4", "b.mp4"):
        for seg in (0, 1):
            for cond, sev in (("clear", 0.0), ("rain", 1.0), ("night", 0.5)):
                bad = cond == "rain"
                for i in range(40):
                    f = rng.normal(0, 1, 10) + (np.array([-3.0] * 10) if bad else 0.0)
                    rows.append({"video": video, "seg": seg, "cond": cond, "sev": sev, "frame": i,
                                 "tp": 0 if bad else 3, "n_gt": 3, "n_det": 1 if bad else 3,
                                 **{f"f{k}": v for k, v in enumerate(f)}})
    return pd.DataFrame(rows)


def test_analysis_labels_and_transfer_on_separable_data():
    df = prepare(_fake_frames())
    assert set(df.groupby(KEY)["frame"].min()) == {14}          # warm-up dropped
    t = retention_table(df)
    assert t.loc[t.cond == "rain", "retention"].iloc[0] == 0.0
    assert t.loc[t.cond == "clear", "retention"].iloc[0] == 1.0
    r = evaluate_split(df[df.seg == 0], df[df.seg == 1], list(range(10)), n_boot=50)
    assert r["auc"] > 0.95 and r["auc_ci95"][0] > 0.9
