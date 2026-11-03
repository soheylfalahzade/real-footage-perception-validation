"""Figure for the leave-one-clip-out results on UA-DETRAC (``rfv-figures``)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLORS = {"sunny": "#e0a030", "cloudy": "#7a8ba0", "rainy": "#4c78a8", "night": "#3b3b6d"}
ORDER = ("sunny", "cloudy", "rainy", "night")


def plot_loco(csv: str | Path, out: str | Path) -> None:
    """AUC with intervals and ECE per held-out clip: combined features vs. the count baseline."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv)
    clips = (df[["clip", "weather"]].drop_duplicates()
             .assign(o=lambda d: d.weather.map(ORDER.index)).sort_values(["o", "clip"]))
    y = {c: i for i, c in enumerate(reversed(list(clips["clip"])))}
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for feat, marker, off in (("combined", "o", 0.17), ("count_ratio_only", "D", -0.17)):
        d = df[df.features == feat]
        for _, r in d.iterrows():
            col = COLORS[r["weather"]]
            yy = y[r["clip"]] + off
            fill = col if feat == "combined" else "white"
            a.plot([r.auc_lo, r.auc_hi], [yy, yy], color=col, lw=1.4)
            a.plot(r.auc, yy, marker, color=col, mfc=fill, ms=6.5)
            b.plot(r.ece, yy, marker, color=col, mfc=fill, ms=6.5)
    a.set_xlim(0.66, 1.01)
    a.set_xlabel("AUC (95% interval)")
    b.set_xlabel("ECE")
    a.set_yticks(list(y.values()))
    a.set_yticklabels([f"{c}  ({clips.set_index('clip').weather[c]})" for c in y], fontsize=8)
    a.set_title("Discrimination on the held-out clip", fontsize=10)
    b.set_title("Calibration on the held-out clip (lower is better)", fontsize=10)
    for ax in (a, b):
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    a.plot([], [], "o", color="#555", label="quality + health (filled)")
    a.plot([], [], "D", color="#555", mfc="white", label="count ratio only (open)")
    a.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("UA-DETRAC, human-labelled boxes: train on 11 clips, test on the 12th",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="docs/results/detrac_leave_one_clip_out.csv")
    ap.add_argument("--out", default="docs/assets/detrac_loco.png")
    args = ap.parse_args(argv)
    plot_loco(args.csv, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
