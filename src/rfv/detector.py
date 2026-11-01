"""Off-the-shelf YOLOv4-tiny (Darknet weights) through OpenCV's DNN module, CPU only."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from relmon.detector import Box

VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class DarknetDetector:
    """Vehicle detector; no training is performed."""

    def __init__(self, data_dir: str | Path, conf: float = 0.3, nms: float = 0.45,
                 input_size: int = 608, threads: int = 1) -> None:
        d = Path(data_dir)
        cv2.setNumThreads(threads)
        dnn: Any = cv2.dnn
        self.net = dnn.readNetFromDarknet(str(d / "yolov4-tiny.cfg"),
                                          str(d / "yolov4-tiny.weights"))
        self.conf, self.nms, self.size = conf, nms, input_size
        self.out_names = self.net.getUnconnectedOutLayersNames()

    def __call__(self, img: np.ndarray, conf: float | None = None) -> list[Box]:
        thr = self.conf if conf is None else conf
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        boxes, scores = [], []
        for out in self.net.forward(self.out_names):
            for row in out:
                cls = int(np.argmax(row[5:]))
                if cls not in VEHICLE_IDS:
                    continue
                s = float(row[4] * row[5 + cls])
                if s < thr:
                    continue
                cx, cy, bw, bh = row[0] * w, row[1] * h, row[2] * w, row[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                scores.append(s)
        keep = cv2.dnn.NMSBoxes(boxes, scores, thr, self.nms) if boxes else []
        res = []
        for i in np.array(keep).reshape(-1):
            x, y, bw, bh = boxes[int(i)]
            res.append(Box(max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h), scores[int(i)]))
        return res


def decode_yolov8(out: np.ndarray, w: int, h: int, conf: float, nms: float,
                  size: int = 640) -> list[Box]:
    """Decode a YOLOv8 ONNX output of shape ``(1, 84, N)`` into vehicle boxes (pixels)."""
    pred = np.asarray(out).reshape(out.shape[-2], out.shape[-1]).T          # (N, 84)
    cls_ids = list(VEHICLE_IDS)
    scores_all = pred[:, 4:][:, cls_ids]
    best = scores_all.argmax(axis=1)
    score = scores_all[np.arange(len(pred)), best]
    keep = score >= conf
    pred, score = pred[keep], score[keep]
    sx, sy = w / size, h / size
    boxes = [[int((c[0] - c[2] / 2) * sx), int((c[1] - c[3] / 2) * sy),
              int(c[2] * sx), int(c[3] * sy)] for c in pred]
    idx = cv2.dnn.NMSBoxes(boxes, score.tolist(), conf, nms) if boxes else []
    res = []
    for i in np.array(idx).reshape(-1):
        x, y, bw, bh = boxes[int(i)]
        res.append(Box(max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h), float(score[int(i)])))
    return res


class OnnxYolov8Detector:
    """YOLOv8 exported to ONNX (``yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12``).

    A second, more modern off-the-shelf detector for the same protocol; no training is performed.
    """

    def __init__(self, data_dir: str | Path, name: str = "yolov8n.onnx", conf: float = 0.3,
                 nms: float = 0.45, input_size: int = 640, threads: int = 1) -> None:
        cv2.setNumThreads(threads)
        dnn: Any = cv2.dnn
        self.net = dnn.readNetFromONNX(str(Path(data_dir) / name))
        self.conf, self.nms, self.size = conf, nms, input_size

    def __call__(self, img: np.ndarray, conf: float | None = None) -> list[Box]:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        out = self.net.forward()
        return decode_yolov8(out, w, h, self.conf if conf is None else conf, self.nms, self.size)


def make_detector(kind: str, data_dir: str | Path) -> DarknetDetector | OnnxYolov8Detector:
    """``yolov4-tiny`` (default, fetched with checksum) or ``yolov8n-onnx`` (user-exported file)."""
    if kind == "yolov8n-onnx":
        return OnnxYolov8Detector(data_dir)
    return DarknetDetector(data_dir)
