# PAC: path-aware criticality for 2D detection (QPAIN 2027)

Research code for a velocity-free, path-aware criticality metric for 2D object detection on Cityscapes,
and a YOLO11m vs. YOLOE case study. See `Risk-Weighted 2D Detection Evaluation — Research Plan (QPAIN 2027).md`.

## Layout
- `pac/gt.py` – Cityscapes instance masks -> boxes, ignore regions, stereo depth, contact surface
- `pac/attrs.py` – per-object attributes (depth, surface, lateral offset, rider/bike pairing)
- `pac/criticality.py` – PAC and baseline criticality functions
- `pac/metrics.py` – matching and weighted recall / precision / AP
- `pac/e0.py` – failure demos F1–F3
- `pac/viz.py` – visual check overlays
- `tests/` – unit tests (`pytest`)

## Data
Cityscapes is not included (license). Download `gtFine_trainvaltest`, `leftImg8bit_trainvaltest`,
`disparity_trainvaltest`, `camera_trainvaltest` from cityscapes-dataset.com into `data/cityscapes/`, then:

    python -m pac.convert data/cityscapes && python -m pac.attrs data/cityscapes val && python -m pac.e0
