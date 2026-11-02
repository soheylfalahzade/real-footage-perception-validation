# Methods

## 1. Data
Two public highway clips are downloaded at run time from the public GitHub repositories that host them
and verified by SHA-256; they are not redistributed here (the licence of the footage is set by the source
repositories): `highway-eu.mp4` (1280 x 720, 29.97 fps, 5,396 frames; source
`devmdave/PythonTrafficCongestion`) and `highway-us.avi` (320 x 240, 15 fps, 500 frames; source
`andrewssobral/vehicle_detection_haarcascades`). Each clip yields two segments of 60 frames sampled at
5 fps (source frames 300-654 and 900-1254 for the first clip, 0-177 and 240-417 for the second), resized to a
width of 640 pixels with the aspect ratio preserved. Traffic in both clips is light (about one to four
detectable vehicles per frame).

## 2. Detector
YOLOv4-tiny (Darknet weights, 608 x 608 input) through OpenCV's DNN module on one CPU thread; classes
car, motorcycle, bus, truck; score threshold 0.3, NMS 0.45. No training or fine-tuning.

## 3. Degradations
The degradations of `perception-reliability-monitor` (`relmon.degrade`: night, rain, fog, glare, lens
dirt, blackout) are applied to the *real* frames at severities 0.5 and 1.0 (blackout 1.0), one stream per
condition and segment (12 streams per segment). The night model has no headlight layer on real frames.

## 4. Pseudo ground truth and label
Detections with score >= 0.5 on the clear stream are the pseudo ground truth of each frame. A degraded
frame *retains* a pseudo-GT box if some detection has IoU >= 0.5 with it. Rolling retention over 15 frames
is `sum(tp) / sum(n_gt)`; a frame is *reliable* if retention >= 0.6. The label therefore measures how
much of what the detector sees in clear conditions it still sees, not absolute accuracy. The first 14
frames of every stream are discarded (window warm-up).

## 5. Features and models
Seven reference-free image-quality features (calibrated on the first 30 clear frames of each clip) and
three detector-health features (window of 20 frames, reference count from the same calibration frames),
as defined in `perception-reliability-monitor`. A standardised logistic model (`relmon.monitor`)
is fitted on training streams and evaluated on held-out streams.

## 6. Splits and statistics
*Temporal*: train on segment 0 of both clips, test on segment 1. *Cross-video*: train on one clip, test
on the other (both directions). AUC is rank-based; ECE uses 10 equal-width bins. Confidence intervals are
percentile bootstraps over *streams* (1000 resamples) because frames inside a stream are strongly
dependent. Baselines: a model using only log-brightness and a model using only the detector count ratio.

## 7. Extending the evaluation (additional clips, second detector)
`rfv-run --extra-dir DIR` adds every clip in `DIR` (mp4, avi, mov, mkv) with at least 2 x 60 sampled frames.
Each is sampled at about 5 fps (stride = round(fps / 5)) in two consecutive 60-frame segments, so the protocol
of sections 3 to 6 applies unchanged. With three or more clips the cross-video split becomes
*leave-one-video-out* (train on all other clips, test on the held-out clip); with the two shipped clips the
original two directional splits are kept, so the reported numbers are unaffected.

`rfv-run --detector yolov8n-onnx` replaces the detector by YOLOv8-nano exported to ONNX and run through
OpenCV's DNN module (640 x 640 input; classes car, motorcycle, bus, truck; score 0.3, NMS 0.45; no training).
Results are written to a separate segment cache, so both detectors can be compared on identical frames.

## 8. Human ground truth (UA-DETRAC)
`rfv-detrac` encodes the first 600 frames of selected UA-DETRAC sequences at 25 fps (640 px width) and writes
`<clip>.gt.json` with the human boxes, the sequence weather label and the filter settings. Boxes with height under
20 px (at 640 px width) or centre inside an ignored region are removed. `rfv-run` uses these boxes for every
degradation stream of the clip instead of the clear-frame detections, so `retention` is recall against human
boxes. The reliability label becomes `retention >= 0.6 * c`, where `c` is the recall of the undegraded stream of
the same clip segment (floored at 0.05); with pseudo ground truth `c = 1` and the definition of section 3 is
recovered. The cross-clip split is leave-one-video-out (section 7).
