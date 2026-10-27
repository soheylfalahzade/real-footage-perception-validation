"""Analysis of the real-footage experiment: detector retention and monitor transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from relmon.evaluate import auc, ece
from relmon.monitor import ReliabilityModel

WINDOW = 15
RELIABLE_RETENTION = 0.6
FEATURE_SETS = {"quality": list(range(7)), "health": [7, 8, 9], "combined": list(range(10)),
                "brightness_only": [0], "count_ratio_only": [8]}
KEY = ["video", "seg", "cond", "sev"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the rolling retention and the binary reliability label (warm-up frames dropped)."""
    df = df.sort_values([*KEY, "frame"]).copy()
    g = df.groupby(KEY)
    df["tp_w"] = g["tp"].transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).sum())
    df["gt_w"] = g["n_gt"].transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).sum())
    df["retention"] = df["tp_w"] / df["gt_w"].where(df["gt_w"] > 0)
    df = df.dropna(subset=["retention"])
    df["label"] = (df["retention"] >= RELIABLE_RETENTION).astype(float)
    df["stream"] = df[KEY].astype(str).agg("|".join, axis=1)
    return df


def retention_table(df: pd.DataFrame) -> pd.DataFrame:
    """Detector behaviour per condition and severity (sums over all frames of all streams)."""
    g = df.groupby(["cond", "sev"])
    out = pd.DataFrame({"retention": g["tp"].sum() / g["n_gt"].sum(),
                        "count_ratio": g["n_det"].sum() / g["n_gt"].sum(),
                        "reliable_frac": g["label"].mean()})
    return out.reset_index()


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, cols: list[int],
                   n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Fit on ``train``, report AUC / ECE on ``test`` with a stream-level bootstrap interval."""
    fcols = [f"f{c}" for c in cols]
    model = ReliabilityModel().fit(train[fcols].to_numpy(), train["label"].to_numpy())
    test = test.assign(p=[model.predict(r) for r in test[fcols].to_numpy()])
    per = {k: (g["label"].to_numpy(), g["p"].to_numpy()) for k, g in test.groupby("stream")}
    y, p = test["label"].to_numpy(), test["p"].to_numpy()
    rng = np.random.default_rng(seed)
    keys = list(per)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        yy = np.concatenate([per[keys[i]][0] for i in pick])
        pp = np.concatenate([per[keys[i]][1] for i in pick])
        if 0 < yy.mean() < 1:
            boots.append(auc(pp, yy))
    return {"auc": auc(p, y), "auc_ci95": [float(np.quantile(boots, 0.025)),
                                           float(np.quantile(boots, 0.975))],
            "ece": ece(p, y), "n_frames": int(len(y)), "reliable_frac": float(y.mean())}


def splits(df: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Temporal split and leave-one-video-out (both directions)."""
    vids = sorted(df["video"].unique())
    out = {"temporal (train seg 0, test seg 1)": (df[df.seg == 0], df[df.seg == 1])}
    for a, b in ((vids[0], vids[1]), (vids[1], vids[0])):
        out[f"cross-video (train {a[:12]}, test {b[:12]})"] = (df[df.video == a], df[df.video == b])
    return out


def run_analysis(frames_csv: str | Path, n_boot: int = 1000) -> dict[str, Any]:
    df = prepare(pd.read_csv(frames_csv))
    res: dict[str, Any] = {"retention": retention_table(df).to_dict("records"),
                           "monitor": {}, "n_streams": int(df["stream"].nunique()),
                           "n_frames": int(len(df)), "reliable_frac": float(df["label"].mean()),
                           "window": WINDOW, "reliable_retention": RELIABLE_RETENTION}
    for name, (tr, te) in splits(df).items():
        res["monitor"][name] = {fs: evaluate_split(tr, te, cols, n_boot)
                                for fs, cols in FEATURE_SETS.items()}
    return res


def plot(res: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ret = pd.DataFrame(res["retention"])
    labels = [f"{c}\n{s:g}" if c != "clear" else "clear" for c, s in zip(ret.cond, ret.sev,
                                                                        strict=True)]
    ax[0].bar(range(len(ret)), ret["retention"], color="#4c78a8")
    ax[0].set_xticks(range(len(ret)), labels, fontsize=7)
    ax[0].set_ylabel("detections retained vs. clear frame")
    ax[0].set_title("Real footage, YOLOv4-tiny: retention under degradation")
    names = list(FEATURE_SETS)
    w = 0.8 / len(res["monitor"])
    for k, (split, r) in enumerate(res["monitor"].items()):
        a = np.array([r[n]["auc"] for n in names])
        lo = np.array([r[n]["auc_ci95"][0] for n in names])
        hi = np.array([r[n]["auc_ci95"][1] for n in names])
        ax[1].bar(np.arange(len(names)) + k * w, a, w, yerr=[a - lo, hi - a], capsize=2,
                  label=split[:28])
    ax[1].set_xticks(np.arange(len(names)) + w, names, fontsize=8)
    ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_ylim(0.3, 1.02)
    ax[1].set_ylabel("AUC (held-out streams)")
    ax[1].set_title("Reliability monitor on real frames")
    ax[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``rfv-analyze``)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default="results/frames.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    res = run_analysis(args.frames, args.n_boot)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    plot(res, out / "analysis.png")
    print(json.dumps({k: res[k] for k in ("n_streams", "n_frames", "reliable_frac")}))
    for split, r in res["monitor"].items():
        print(split)
        for fs, v in r.items():
            print(f"   {fs:16s} AUC {v['auc']:.3f} [{v['auc_ci95'][0]:.3f}, {v['auc_ci95'][1]:.3f}]"
                  f"  ECE {v['ece']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
