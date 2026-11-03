from pathlib import Path

from rfv.figures import plot_loco


def test_plot_loco_renders_the_shipped_summary(tmp_path):
    csv = Path(__file__).parents[1] / "docs" / "results" / "detrac_leave_one_clip_out.csv"
    out = tmp_path / "f.png"
    plot_loco(csv, out)
    assert out.stat().st_size > 5000
