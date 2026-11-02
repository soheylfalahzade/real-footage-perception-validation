import json

import cv2
import numpy as np
import pandas as pd

from rfv.analysis import prepare
from rfv.detrac import gt_for_segment, keep_box, parse_sequence_xml, prepare_clips

XML = """<sequence name="{n}">
 <sequence_attribute camera_state="stable" sence_weather="{w}"/>
 <ignored_region><box left="0" top="0" width="20" height="20"/></ignored_region>
 {frames}
</sequence>"""
BIG = '<box left="40" top="30" width="30" height="24"/>'
SMALL = '<box left="5" top="5" width="10" height="10"/>'
FRAME = ('<frame density="2" num="{k}"><target_list>'
         f'<target id="1">{BIG}</target><target id="2">{SMALL}</target></target_list></frame>')


def _make(tmp, name, weather, n=600):
    d = tmp / "DETRAC-Images" / name
    d.mkdir(parents=True)
    for i in range(1, n + 1):
        cv2.imwrite(str(d / f"img{i:05d}.jpg"), np.full((54, 96, 3), i % 255, np.uint8))
    ann = tmp / "DETRAC-Train-Annotations-XML"
    ann.mkdir(exist_ok=True)
    frames = "".join(FRAME.format(k=k) for k in range(1, n + 1))
    (ann / f"{name}.xml").write_text(XML.format(n=name, w=weather, frames=frames), encoding="utf-8")


def test_parse_filter_and_clip_pipeline(tmp_path):
    _make(tmp_path, "MVI_00001", "rainy")
    _make(tmp_path, "MVI_00002", "night")
    weather, ignored, frames = parse_sequence_xml(tmp_path / "DETRAC-Train-Annotations-XML"
                                                  / "MVI_00001.xml")
    assert weather == "rainy" and len(ignored) == 1 and len(frames[1]) == 2
    assert not keep_box((5, 5, 15, 15), ignored, 5.0)           # inside the ignored region
    assert keep_box((40, 30, 70, 54), ignored, 20.0)
    out = tmp_path / "clips"
    man = prepare_clips(tmp_path, out, per_weather=1, seed=0, min_h=20.0)
    assert {m["weather"] for m in man} == {"rainy", "night"}
    assert (out / "MVI_00001.mp4").exists()
    gt, w = gt_for_segment(out / "MVI_00001.gt.json", start=300, stride=5, count=60)
    assert w == "rainy" and len(gt) == 60 and len(gt[0]) == 1    # the small box was filtered out
    meta = json.loads((out / "MVI_00001.gt.json").read_text())
    assert meta["orig_size"] == [96, 54]


def test_label_is_relative_to_the_clear_stream_recall():
    rows = []
    for cond, tp in (("clear", 2), ("rain", 1), ("blackout", 0)):     # 2 of 4 GT boxes at best
        for i in range(30):
            rows.append({"video": "a.mp4", "seg": 0, "cond": cond, "sev": 0.0, "frame": i,
                         "tp": tp, "n_gt": 4, "n_det": tp, **{f"f{k}": 0.0 for k in range(10)}})
    df = prepare(pd.DataFrame(rows))
    assert df[df.cond == "clear"]["label"].eq(1.0).all()          # 0.50 >= 0.6 * 0.50
    assert df[df.cond == "rain"]["label"].eq(0.0).all()           # 0.25 <  0.6 * 0.50
