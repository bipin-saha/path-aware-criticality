# Risk-Weighted 2D Detection Evaluation — Research Plan (QPAIN 2027)

Nov 5, 2026 · @Bipin Saha

## Summary

We propose a path-aware criticality metric (working name PAC) for 2D object detection that fixes three demonstrated failures of existing criticality metrics on single camera frames, validate it against human judgments of which objects matter, and use it to compare a closed-set detector (YOLO11m) with an open-vocabulary detector (YOLOE). The paper must be ready by 15 December 2026.

| Item | Plan |
| --- | --- |
| Venue | 2027 IEEE 3rd QPAIN, Track 03 (AI and ML: computer vision; AI in autonomous systems, LiDAR and robotics) |
| Format | 4–6 pages, IEEE conference template, double-blind |
| Paper ready (internal) | 15 December 2026 |
| Submission deadline | 20 December 2026 (confirm on the official site) |
| Conference | 31 March – 3 April 2027, Chattogram, Bangladesh |
| Contribution 1 | Evidence that existing criticality metrics fail on 2D single frames (no velocity, height proxy, no path) |
| Contribution 2 | PAC: stereo depth × path relevance × class vulnerability, no velocity needed |
| Contribution 3 | Validation: agreement with about 200 human-labelled critical / non-critical objects |
| Contribution 4 | Case study: YOLO11m (fine-tuned) vs. YOLOE (zero-shot) |
| Phase 2 | More detectors, only after a clear result |
| Dataset | Cityscapes (gtFine, stereo disparity, camera calibration) |
| Compute | RTX A1000 laptop 6 GB and/or MacBook Air M5 24 GB |
| Method status | Metric form is a starting point; the pilot and human-agreement study decide the final version |

Source: [QPAIN 2027 call for papers](https://conf-photonics.pages.dev/call-for-papers).

## Problem and research questions

Existing criticality metrics were built for 3D detection with velocity, or for pedestrians with pixel height as a distance proxy. On single 2D camera frames they either collapse to plain recall or misjudge which objects matter, so 2D detectors, including the open-vocabulary models now proposed for driving, have no reliable safety-oriented evaluation.

**Research questions**

1. **RQ1 — Failures:** Do existing criticality metrics (OCM without velocity, height-based foreground, inverse distance) fail measurably on Cityscapes?
2. **RQ2 — Validity:** Does PAC agree with human judgments of object criticality better than those metrics?
3. **RQ3 — Components:** Which PAC factors (distance, path, class) are needed for that agreement?
4. **RQ4 — Use case:** Does PAC reveal differences between YOLO11m and YOLOE that mAP and existing metrics hide?

**Follow-up questions (only after a clear result)**

5. **RQ5 — Robustness:** Are the conclusions stable across parameters and under bootstrap?
6. **RQ6 — Generality:** Does the YOLO11m vs. YOLOE pattern hold for more detectors?

**Working hypotheses (checked, not assumed)**

- **H1:** OCM without velocity gives criticality 1 to every object, so its safety-weighted recall equals plain recall.
- **H2:** Height-based distance misplaces a noticeable share of persons relative to stereo depth (target to measure, not assume: above 10%).
- **H3:** A meaningful share of objects within stopping distance are off-road, so distance-only weighting overstates them.
- **H4:** PAC has the highest agreement (AUROC) with human critical labels.
- **H5:** The YOLO11m–YOLOE gap under PAC differs from the mAP gap by at least 5 points.

If H4 fails, the paper becomes a failure analysis of existing metrics plus the YOLO11m vs. YOLOE case study.

## Literature review

Criticality-weighted detection metrics exist, but each fails on single 2D camera frames in a specific way, which justifies a new metric. No existing work evaluates open-vocabulary detectors with criticality metrics.

| Work | What it does | Gap it leaves for us |
| --- | --- | --- |
| [Ceccarelli & Montecchi, Object Criticality Model (IEEE Access; arXiv 2022)](https://arxiv.org/abs/2203.02205) | Weights each object by distance, collision distance and time-to-collision; defines safety-weighted recall, reliability-weighted precision and critical AP; re-ranks 9 detectors on nuScenes | 3D, cars only, needs velocity; no 2D or open-vocabulary detectors |
| [Object criticality for safer navigation (2024)](https://arxiv.org/abs/2406.10232) and [validation of safety metrics (SAC 2024)](https://leonarmo.folk.ntnu.no/publications/pdf/2024SAC.pdf) | Tests whether OCM and PKL actually reflect safety | Still 3D nuScenes |
| [Schreier et al., offline evaluation of 3D detection (2023)](https://arxiv.org/abs/2308.12779) | 16 detectors in CARLA closed loop: inverse-distance AP correlates 0.955 with collision count, higher than plain AP | Our substitute for planner validation; 3D LiDAR only |
| [Philion et al., PKL (CVPR 2020)](https://arxiv.org/abs/2004.08745) | Planner-centric metric from a trained planner | Needs a planner and 3D tracks |
| [Feifel, Franke et al., revisiting pedestrian detection evaluation (2022; arXiv 2025)](https://arxiv.org/abs/2511.10308) | On CityPersons, uses Cityscapes segmentation to define safety-critical "foreground" pedestrians via braking distance (22 m at 30 km/h, mapped to ≥190 px height); shows rankings change | Pedestrians only, pixel height instead of depth; names depth as future work |
| [Lyssenko et al., relevance metrics (CVPRW 2021)](https://mlanthology.org/cvprw/2021/lyssenko2021cvprw-evaluation) and [safety-adapted loss (2024)](https://arxiv.org/abs/2402.02986) | Distance-conditioned pedestrian metric (CARLA); zone-based recall | Synthetic data, pedestrians only |
| [Volk et al., comprehensive safety metric (ITSC 2020)](https://www.embedded.uni-tuebingen.de/assets/publications/Volk-Gamerdinger-vonBernuth-Bringmann_Safety_Metric.pdf), [Gamerdinger et al. (ICVES 2025)](https://arxiv.org/abs/2512.15181), [EPSM (2025)](https://arxiv.org/abs/2512.15195) | Safety scores using RSS distances, criticality classification, lane + object safety | Need dynamics or maps; no open-vocabulary models |
| Bansal et al., risk ranked recall (MECO 2021) | Recall split by collision-risk category | Needs trajectories |
| [DetLH (2024)](https://arxiv.org/abs/2410.20371), [YOLOE docs](https://docs.ultralytics.com/models/yoloe) | Zero-shot open-vocabulary detection evaluated on Cityscapes with plain AP | No criticality analysis |

**Failures of existing metrics on 2D single frames**

| Metric | Failure | Evidence we will show (E0) |
| --- | --- | --- |
| OCM (Ceccarelli & Montecchi) | Without velocity, collision-distance and time terms default to 1; the combination rule then makes every object maximally critical | F1: OCM safety-weighted recall equals plain recall on Cityscapes val |
| OCM | Safety-weighted recall uses predicted criticality, which 2D detectors cannot output; values above 1 are clipped | Discussed, not measured |
| OCM | 1,500 parameter settings with changing rankings; no principled default | Discussed |
| Feifel et al. foreground | Pixel height as distance proxy assumes 1.7 m persons; fails for children, sitting, truncated or occluded people | F2: share of persons misassigned versus stereo depth |
| Feifel et al. foreground | Hard cutoff at 22 m and no path: a person on the far sidewalk counts like one in the lane | F3: share of objects within stopping distance that are off-road |
| Inverse-distance AP | 1/Z explodes near zero; no path; tracks collisions but rewards overly defensive behaviour (Schreier et al.) | Included as a baseline in the human-agreement test |

**Design requirements for PAC (each answers a failure)**

1. Needs no velocity, so it works on single frames (fixes F1).
2. Uses measured depth, not a height proxy, and works for all classes (fixes F2).
3. Continuous in distance and aware of path, with no hard cutoff (fixes F3).
4. Parameters come from stopping distance; with uniform weights it reduces exactly to standard recall and AP.

**Draft novelty statement:** A velocity-free, path-aware criticality metric for 2D detection, built on stereo depth, validated against human criticality judgments, and used for the first criticality-aware comparison of an open-vocabulary detector with a closed-set detector for driving.

**Remaining reading (before claiming novelty)**

- [ ] Read Volk et al. CSM, Gamerdinger et al. (ICVES 2025) and EPSM in full: check whether any already handles path without velocity on 2D data
- [ ] Read Gamerdinger et al.'s criticality-classification evaluation and reuse its protocol for the human-agreement test
- [ ] Check the [ErrorCategories code](https://github.com/BeFranke/ErrorCategories) from Feifel et al. for reuse as a baseline
- [ ] One more search for 2025–2026 work combining open-vocabulary detection with safety metrics
- [ ] Agree the novelty statement with the advisor

## Dataset: Cityscapes

Cityscapes gives everything the metric needs without LiDAR: instance masks for boxes, stereo disparity for depth, semantic labels for road vs. sidewalk, and per-image calibration.

| Split | Images | Use |
| --- | --- | --- |
| train (gtFine) | 2,975 | Fine-tune YOLO11m; never tune metric parameters on val |
| val (gtFine) | 500 | Evaluation set for all results |
| test | 1,525 | Labels withheld; not used |

**Packages to download:** `gtFine_trainvaltest`, `leftImg8bit_trainvaltest`, `disparity_trainvaltest`, `camera_trainvaltest`.

**Ground-truth preparation**

1. **Boxes from instances.** From `*_instanceIds.png`, values ≥ 1000 are instances (class = value // 1000). Keep the 8 instance classes: person, rider, car, truck, bus, train, motorcycle, bicycle. Box = tight bounds of each mask.
2. **Ignore regions.** Pixels with an instance class but value < 1000 are group regions (e.g. a crowd of cars). Treat them like COCO `iscrowd`: detections inside are neither TP nor FP.
3. **Tiny objects.** Keep all boxes, but log height in pixels. Report results with and without boxes under 10 px tall.
4. **One label set for everything.** The same converted boxes are the YOLO11m training labels and the metric ground truth, so both models are scored against identical labels.
5. **Semantic map.** Use `*_labelIds.png` to look up road, sidewalk, parking and other classes at each object's ground contact point.

**Known limits:** one city domain (Germany, urban speeds), single annotated frame per snippet (no temporal metrics), disparity noisy beyond about 50–60 m.

## Method: the PAC metric (starting point)

PAC gives each object a criticality from stereo depth, path relevance and class, with no velocity needed. The exact forms below are first versions; the pilot and the human-agreement study decide the final ones.

```latex
\kappa_i = \kappa_d(Z_i)\; g(\mathrm{path}_i)\; h(c_i)
```

### 1. Distance Z per object (fixed)

1. Decode disparity: d = (p − 1) / 256 for pixel value p > 0; p = 0 is invalid.
2. Depth from stereo, with baseline B and focal length f\_x from each image's camera JSON:

```latex
Z = \frac{B\, f_x}{d}
```

3. Median of valid disparity inside the instance mask after a 3 px erosion; require at least 20 valid pixels.
4. Fallback when disparity is sparse: ground-plane geometry from box bottom row v\_b, camera height h\_cam and horizon row v\_h:

```latex
Z \approx \frac{f_y\, h_{\mathrm{cam}}}{v_b - v_h}
```

5. Store `depth_source` per object; check 20 images by eye.

### 2. Distance criticality κ\_d (continuous)

Starting form: the OCM parabola with a small floor so no object weighs exactly zero, and D\_max tied to stopping distance.

```latex
\kappa_d(Z) = \max\!\left(\epsilon,\; 1 - \frac{Z^2}{D_{\max}^2}\right), \qquad d_{\mathrm{stop}} = v\, t_r + \frac{v^2}{2a}
```

Defaults: v = 50 km/h, t\_r = 1.5 s, a = 7 m/s² (d\_stop ≈ 35 m), ε = 0.05. The pilot tests D\_max ∈ {d\_stop, 1.5 d\_stop, 2 d\_stop}.

### 3. Path relevance g (continuous, velocity-free)

Contact point = bottom-centre of the box; lateral offset X = (u\_b − c\_x) Z / f\_x; surface class from the semantic map at and just below the contact point.

| Surface at contact point | g |
| --- | --- |
| Road, \|X\| ≤ 2 m (ego corridor) | 1.0 |
| Road, 2 m < \|X\| < 5 m | Linear from 1.0 down to 0.5 |
| Road, \|X\| ≥ 5 m | 0.5 |
| Sidewalk, parking, terrain, other | 0.2 |

### 4. Class vulnerability h

Vulnerable road users (person, rider, bicycle, motorcycle) h = 1.0; vehicles (car, truck, bus, train) h = 0.6. Kept only if it improves human agreement (E1).

### 5. PAC scores

Matching: greedy per class at IoU ≥ 0.5, ignore regions removed. True positives use ground-truth criticality; false positives use criticality from depth and semantics at the predicted box.

```latex
\mathrm{PAC\text{-}R} = \frac{\sum_{i \in TP} \kappa_i}{\sum_{i \in GT} \kappa_i}, \qquad \mathrm{PAC\text{-}P} = \frac{\sum_{i \in TP} \kappa_i}{\sum_{i \in TP} \kappa_i + \sum_{j \in FP} \kappa^{fp}_j}
```

- **PAC-AP:** area under the PAC-P / PAC-R curve, 101-point interpolation.
- **Critical miss rate:** share of images with at least one missed object with κ ≥ 0.8.
- **Sanity property:** with κ = 1 for all objects, PAC-R, PAC-P and PAC-AP equal standard recall, precision and AP (unit test).

### 6. Baselines implemented on the same matches

| Baseline | Implementation on Cityscapes |
| --- | --- |
| mAP@0.5, mAP@\[.5:.95\] | Standard |
| OCM without velocity | Distance term from stereo depth; collision and time terms at their documented fallback of 1 |
| OCM distance-only | Distance term alone (a generous version of OCM) |
| Feifel et al. foreground miss rate | Persons only, height ≥ 190 px |
| Inverse-distance AP | Weight 1/max(Z, 1 m) |
| Distance-only PAC | κ\_d alone (ablation) |

## Models

Phase 1 uses two models and one training run; Phase 2 adds detectors only after Phase 1 shows a clear result.

**Phase 1**

| Model | Type | Setup | Training |
| --- | --- | --- | --- |
| YOLO11m | Closed-set, one-stage | Fine-tune from COCO weights on Cityscapes train (8 classes) | Yes, the only training run |
| YOLOE-11m (text prompt) | Open-vocabulary, zero-shot | `set_classes` with the 8 class names; no Cityscapes training | No |
| YOLOE-11m fine-tuned (optional) | Open-vocabulary, adapted | Fine-tune on Cityscapes train with the detection trainer | Only if time allows; separates "open-vocabulary" from "not trained on Cityscapes" |

YOLOE-11m is the default because it shares the YOLO11 family and scale, which keeps the comparison fair. YOLOE-26m is newer and needs ultralytics 8.4.0 or later ([YOLOE docs](https://docs.ultralytics.com/models/yoloe)); it is a Phase 2 option.

**YOLO11m training settings**

| Setting | A1000 laptop 6 GB (CUDA) | MacBook Air M5 24 GB (MPS) |
| --- | --- | --- |
| imgsz | 896 (768 if OOM) | 1024 |
| batch | 2–4 | 4–8 |
| Precision | AMP on | fp32 (no AMP on MPS) |
| Common | 80 epochs, patience 20, `cache=disk`, fixed seed | same |

**YOLOE inference settings**

- Prompts: person, rider, car, truck, bus, train, motorcycle, bicycle; report them verbatim.
- imgsz 1024 (same as YOLO11m evaluation where possible), score threshold 0.001, at most 100 detections per image.
- Use box outputs only; masks are ignored.

**Phase 2 candidates (after a good Phase 1 result)**

Faster R-CNN (MMDetection Cityscapes checkpoint), Grounding DINO Swin-T, YOLO-World, YOLOE-26m, and YOLO11 at other sizes.

## Experiments

The paper rests on three pieces of evidence: existing metrics fail (E0), PAC matches human judgment best (E1), and PAC reveals something about YOLO11m vs. YOLOE (E3). Robustness checks and extra models come only after these hold.

| ID | Experiment | When | Output |
| --- | --- | --- | --- |
| E0 | Failure demos: F1 OCM collapse, F2 height-proxy error vs. stereo depth, F3 off-road share within stopping distance | Always, first | Failure table for Section 2 of the paper |
| E1 | Human-agreement study (protocol below) | Always | AUROC and accuracy per metric vs. human labels |
| E2 | Pilot: PAC variants (D\_max, path, class) on both models | Always | Final PAC form |
| E3 | Case study: YOLO11m vs. YOLOE on mAP, baselines and PAC | Always | Main table + gap chart |
| E4 | Error breakdown by distance band, path zone and class | Always | Recall heatmap per model |
| E5 | Component ablation: distance only, + path, + class | Always (justifies the metric design) | Ablation table |
| E6 | Qualitative examples of critical misses | Always | 2–3 figure panels |
| E7 | Robustness: parameter sensitivity + bootstrap over images | Only after E1 and E3 hold | Confidence intervals, stability |
| E8 | Phase 2 detectors | Only after a clear result and if time allows | Generality |

**Human-agreement protocol (E1)**

1. **Sample:** 200 ground-truth objects from Cityscapes val, stratified by distance band (0–15, 15–35, 35–60, 60+ m) and surface (road vs. off-road), across all 8 classes.
2. **Question:** shown the image with one box highlighted, the annotator answers: "Driving this car at city speed, would you need to watch or react to this object in the next few seconds?" Yes or no.
3. **Annotators:** you and one labmate, independently, blind to depth values and metric scores. Report Cohen's kappa; resolve disagreements by discussion for the final label.
4. **Score:** for each metric's criticality (PAC, OCM without velocity, OCM distance-only, height rule, inverse distance, distance-only PAC), compute AUROC against the labels, plus accuracy at the best threshold.
5. **Tooling:** a simple labelling page or spreadsheet with image crops; about 2–3 hours per annotator.

**Pilot decision rule (17 November gate):** proceed with the full story if F1–F3 show clear failures and PAC has the highest AUROC in E1. Otherwise fall back to failure analysis + case study.

**Statistics:** plain counts during exploration. Once results hold, bootstrap 1,000 resamples for confidence intervals on the main numbers, including AUROC differences in E1.

**Reproducibility:** fixed seeds, one config per run, predictions as COCO-format JSON, human labels released with the code.

## Compute plan

Phase 1 needs one training run (YOLO11m) and two inference passes over 500 images, which either machine can handle; all metric work runs on CPU.

| Task | A1000 laptop 6 GB | MacBook Air M5 24 GB | Recommended |
| --- | --- | --- | --- |
| Cityscapes conversion, depth, weights | Yes (CPU) | Yes (CPU) | Mac |
| YOLO11m fine-tuning | Yes, imgsz 896, AMP | Yes via MPS, imgsz 1024, larger batch | Time 1 epoch on each, use the faster |
| YOLO11m and YOLOE inference (500 images) | Minutes | Minutes | Either |
| Metrics, pilot variants, plots | Yes | Yes | Mac |
| Bootstrap and sensitivity (later) | Yes | Yes | Mac |
| Phase 2 two-stage models (MMDetection) | Yes, CUDA | Unreliable on Apple Silicon | A1000 or lab machine |

**Practical notes**

- Laptop GPU: plugged in, high-performance mode, watch `nvidia-smi -l 5` for throttling, train overnight.
- MacBook Air is fanless: check epoch time after 30 minutes before committing a full run.
- MPS: set `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Sync predictions as JSON between machines, not weights.

## Timeline

Paper ready on 15 December, with a go/fallback gate on 18 November once the failure demos, human labels and PAC pilot are in. Robustness checks run only if results hold; writing starts 1 December regardless.

&#91;embedded content: QPAIN 2027 plan · 6 weeks, 1 decision gate\]

YOLO11m trains overnight while depth and metric code are built, and YOLOE needs no training, so the pilot can start before training finishes.

## Paper outline

Working title: "PAC: A Velocity-Free, Path-Aware Criticality Metric for 2D Object Detection in Driving." Target 6 pages including references.

| Section | Pages | Content |
| --- | --- | --- |
| 1. Introduction | 0.75 | Example: an off-road person counted as critical, an in-lane person missed; contributions |
| 2. Related work and failures | 1.0 | OCM, CSM, Feifel et al., inverse-distance AP; failure table with F1–F3 numbers (E0) |
| 3. PAC metric | 1.0 | Depth, distance and path criticality, class weight, PAC-R/P/AP, sanity property |
| 4. Validation | 0.75 | Human-agreement study (E1) and component ablation (E5) |
| 5. Case study | 1.5 | YOLO11m vs. YOLOE (E3, E4, E6) |
| 6. Limitations and conclusion | 0.5 | Static frames, one city, 200 labels, two models, no closed-loop test |
| References | 0.5 | About 20–25 |

**Figures**

1. Motivating example: same image scored by mAP, OCM and PAC.
2. PAC illustration: objects coloured by criticality, D\_max arc and ego corridor drawn.
3. Human agreement: AUROC per metric (E1).
4. Gap chart: YOLO11m vs. YOLOE under mAP, baselines and PAC (E3).

**Tables**

1. Failures of existing metrics (E0).
2. Component ablation (E5).
3. Case-study results (E3), with confidence intervals if bootstrap was run.

**Contributions to claim (draft)**

- Evidence that existing criticality metrics fail on single 2D frames.
- PAC, a velocity-free, path-aware criticality metric built on stereo depth.
- Validation against human criticality labels, released with the code.
- The first criticality-aware comparison of an open-vocabulary and a closed-set detector for driving.

## Risks and open decisions

The biggest risks are that a less-read paper (CSM, EPSM, Gamerdinger) already covers path without velocity, and that PAC does not beat simpler baselines in human agreement; the reading tasks and the 18 November gate address both.

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| An existing metric already handles path without velocity on 2D | Medium | Read CSM, EPSM, Gamerdinger in full by 12 November; if found, position PAC as its 2D stereo variant and keep the failure + validation story |
| PAC does not win human agreement (E1) | Medium | Fall back to failure analysis + case study; report honestly which simple metric works best |
| 200 labels from 2 annotators judged too few or subjective | Medium | Stratified sample, Cohen's kappa, release labels; state as limitation |
| Reviewers call weights arbitrary | Medium | Stopping-distance defaults, component ablation (E5), sensitivity after a good result (E7) |
| YOLOE gap is just "not trained on Cityscapes" | High | Optional fine-tuned YOLOE run; otherwise frame carefully |
| No closed-loop validation | Certain | Human agreement + Schreier et al.'s evidence for distance weighting; state as limitation |
| Noisy far-range disparity | Medium | Ground-plane fallback; most weight sits within about 35 m |
| 15 December deadline is tight | Medium | Labelling runs in parallel with training; E7 and E8 optional; writing starts 1 December regardless |

**Open decisions**

- [ ] Metric name (PAC is a placeholder)
- [ ] YOLOE-11m (default) or YOLOE-26m
- [ ] Run the optional fine-tuned YOLOE, or zero-shot only
- [ ] Final D\_max, path and class settings (decided by E1 and E2)
- [ ] Whether false positives get criticality weights
- [ ] Who is the second annotator
- [ ] Advisor sign-off on positioning and novelty statement
