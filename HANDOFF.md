# Handoff (5 Nov 2026)

Project: path-aware criticality metric ("PAC", name not final) for 2D detection on Cityscapes, plus YOLO11m vs. YOLOE case study.
Full plan: `Risk-Weighted 2D Detection Evaluation — Research Plan (QPAIN 2027).md`. Gate: 17/18 Nov; paper ready 15 Dec.

## Done
- Metric core: `pac/criticality.py` (PAC + baselines), `pac/metrics.py` (matching, weighted recall/precision/AP, critical miss rate), `pac/stats.py` (rank AUROC). Unit tests pass (`pytest`), including the sanity property (weights = 1 -> standard recall/precision/AP).
- Ground truth: `pac/convert.py` (instance masks -> boxes + YOLO labels), `pac/gt.py` (depth, surface), `pac/attrs.py` (per-object depth/surface/lateral offset, rider-bicycle pairing).
- E0 failure demos on val (`pac/e0.py`), visual check of 20 val frames (`pac/viz.py`).
- Cityscapes fully downloaded and verified on the first PC: gtFine, leftImg8bit, disparity, camera.

## E0 results (val, 500 images, 10,119 objects) - preliminary
- F1: OCM without velocity gives weight 1 to every object (true by construction; present as explanation, not as a measured result).
- F2: 2,833 persons; median height-proxy distance error 10.6%; 30.2% off by more than 25%; 9.2% on the wrong side of the 22 m foreground line. H2 asked for more than 10%, so it is NOT met on that measure. Decide which F2 measure the paper reports before looking again. Also check that Feifel's 190 px cutoff is consistent with the 1.7 m formula (about 175 px at 22 m for this camera).
- F3: of 3,186 objects within d_stop (34.6 m) with a known surface, about 52% are off-road (persons about 70%, cars about 30%). About 480 have unknown surface and are excluded.
- Depth: stereo for 8,874 objects, ground-plane fallback 890, none 355 (small far objects, excluded).

## Bugs fixed (keep in mind)
- Ground-plane fallback horizon is `cy - fy*tan(pitch)` (sign checked against stereo, 14% median error). The principal row was wrong.
- Surface lookup ignores occluders and reads only ground classes (6, 7, 8, 9, 10, 22) in a band under the box; UNKNOWN (-1) if none found.
- Rider and bicycle/motorcycle are paired (overlap >= 0.3 of the bike box) and share a surface; 420 of 543 val riders paired. Pairs are still two objects in matching. Open decision: score a pair as one road user?

## Not done / next steps
1. Download Cityscapes again on the new PC (`data/` is not in the repo; the license forbids redistribution). Packages: `gtFine_trainvaltest`, `leftImg8bit_trainvaltest`, `disparity_trainvaltest`, `camera_trainvaltest` from cityscapes-dataset.com (account needed). Tool: `pip install cityscapesscripts`, then `csDownload -r -d data/cityscapes <package>.zip`. The server drops connections and can hang with no timeout: use `-r` (resume) in a retry loop and kill a stalled process. Unzip all four into `data/cityscapes/`.
2. Regenerate derived files: `python -m pac.convert data/cityscapes && python -m pac.attrs data/cityscapes val && python -m pac.e0` (also `attrs ... train` if needed).
3. Re-render about 5 frames on the real photos: `python -m pac.viz data/cityscapes 20 data/viz_check rgb` (the first visual check used label colours, not photos). Not yet re-checked: cyclist-crowd frames after rider pairing.
4. Detectors: official COCO YOLO11m (decided to try first instead of training; no `rider` class in COCO, map person detections to rider GT and report it), YOLOE-11m zero-shot with the 8 class prompts (ultralytics 8.3.131 imports YOLOE fine). Fine-tuned YOLO11m stays optional (plan's overnight run). Write inference to COCO-format JSON.
5. E1 human-agreement labelling page (200 objects, stratified), E2 pilot, E3 case study, E4-E6 per plan.
6. Reading before claiming novelty (due 12 Nov): Volk CSM, Gamerdinger ICVES 2025, EPSM; advisor sign-off on novelty statement.

## Environment notes
- First PC used conda env `gaze` (Python 3.10, torch 2.12, ultralytics 8.3.131, opencv, numpy). `scipy` and `scikit-learn` did not import there (macOS binary incompatibility), so the code deliberately avoids both: OpenCV for erosion, own AUROC. Don't add scipy/sklearn imports without testing on the new machine.
- Use `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon. Metric code is CPU-only.

## Open decisions (from the plan, plus new)
- Metric name; YOLOE-11m vs. 26m; fine-tuned YOLOE or not; final D_max/path/class settings; false-positive criticality; second annotator; advisor sign-off.
- New: official COCO YOLO11m as the closed-set baseline instead of a fine-tuned one (affects Contribution 4 wording); score rider+bike as one road user; soften the road-edge jump in g; KITTI-360 as an optional second dataset (stereo, Cityscapes-style labels, but no depth maps or 2D boxes provided; CC BY-NC-SA).

## Repo is public
Do not add data, weights, credentials or personal details. Double-blind: make it private or use an anonymized mirror before linking from a submission.

## Update (6 Nov): detector results and decisions
Predictions (not in repo; regenerate with `python -m pac.infer`): official COCO `yolo11m` (`--rider-from-person`, MPS, 500 val images in about 70 s) and zero-shot `yoloe-11m` (CPU, about 150 s). YOLOE needed `pip install git+https://github.com/ultralytics/CLIP.git` in the env. Scoring: `python -m pac.evaluate data/cityscapes data/preds/yolo11m_coco.json data/preds/yoloe11m_zs.json [--min-h 10] [--fp unit] [--merge-rider]`. Result JSONs are in `results/`.

Decisions taken (change if the advisor disagrees):
- Primary metric: mean of per-class AP at IoU 0.5 (reduces to standard mAP with weights 1). Pooled AP and operating-point (score >= 0.25) PAC-R/P are secondary: the class-vulnerability factor h cancels inside per-class AP and only matters there.
- Primary protocol: rider merged into person (`--merge-rider`; rider detections dropped). 8-class numbers are secondary because COCO has no rider class.
- False positives get criticality from stereo depth + surface under the predicted box (`--fp crit`); `--fp unit` is reported as a sensitivity check.

Findings so far (single run, no confidence intervals):
- 8-class, per-class mean AP: unweighted gap YOLO11m minus YOLOE 8.4 points, PAC gap 5.6 (H5 needs a difference of at least 5: NOT met; 2.8). Rider merged: 7.7 vs 3.9 (difference 3.8, still not met).
- Pooled AP flips the 8-class ranking under PAC (YOLOE ahead by 1.1), so the case-study conclusion depends on the aggregation. Do not claim a winner before bootstrap (E7).
- Boxes under 10 px and FP weighting change the PAC gap by at most 0.5 points.
- Critical miss rate (kappa >= 0.8, score >= 0.25): YOLO11m 2.0%, YOLOE 3.8% (8-class), both 2.0% rider-merged.
- Not done: bootstrap CIs, human-agreement study E1, PAC variants pilot E2, no training run (needs the user's permission).

Standing rules from the user: ask before any training; run anything over 2 minutes detached; answer timing questions from measurement, never estimates.
