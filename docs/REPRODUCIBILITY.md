# Reproducibility

| step | command | approx. time |
|---|---|---|
| resource check (run this first) | `rfv-preflight --data-dir data` | < 1 min |
| fetch and verify public files | `rfv-fetch --data-dir data` | < 1 min |
| detector on all degraded streams | `nice -n 19 rfv-run --data-dir data --out results/frames.csv` | 10-15 min, one CPU core |
| analysis, table and figure | `rfv-analyze --frames results/frames.csv --out results` | < 1 min |
| animation | `rfv-gif --data-dir data --frames results/frames.csv --out docs/assets/real_monitor.gif` | 1 min |

The run uses one CPU thread, no GPU and about 1 GB of RAM; `rfv-preflight` prints free memory, load and
an estimate of the run time, and exits non-zero if the machine is too busy. `rfv-run --max-streams 3`
runs a small subset. The detector and all random degradations are deterministic given the seed
(`--seed`), so rerunning on the same package versions reproduces the CSV. Versions: `requirements-lock.txt`
(Python 3.12, OpenCV 4.x). The dependency on `perception-reliability-monitor` is pinned to tag `v0.1.0`.

## Independent re-run

The detector stage (`rfv-run`) was re-run on a second machine (Ubuntu on WSL2, OpenCV 4.10.0, one CPU
thread, about 5 minutes). The per-frame detection counts (`tp`, `n_gt`, `n_det`) of all 2,880 rows were
identical to `docs/results/frames.csv`, and the AUC point estimates of `rfv-analyze` were identical; the
bootstrap intervals differ in the third decimal because the shipped table was analysed with a different
resample count. OpenCV 5 is not supported (its DNN module no longer reads Darknet models), hence the
`opencv-python-headless>=4.8,<5` pin.
