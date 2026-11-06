"""Run a detector on Cityscapes val and write predictions as JSON: {stem: [{cls, box xyxy, score}]}.
  python -m pac.infer yolo11m  data/cityscapes data/preds/yolo11m_coco.json
  python -m pac.infer yoloe-11m data/cityscapes data/preds/yoloe11m_zs.json
COCO YOLO11m has no 'rider' class: with --rider-from-person every person detection is also emitted as a rider
candidate (same score), so rider AP pays for the persons that are not riders. Report this mapping."""
import json, sys, pathlib, argparse, time
from .gt import CLASS_NAMES

COCO2CS = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--split", default="val"); ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None); ap.add_argument("--rider-from-person", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(pathlib.Path(a.root, "leftImg8bit", a.split).rglob("*_leftImg8bit.png"))
    if a.limit: files = files[:a.limit]
    yoloe = a.model.startswith("yoloe")
    if yoloe:
        from ultralytics import YOLOE
        m = YOLOE(a.model + "-seg.pt" if not a.model.endswith(".pt") else a.model)
        m.set_classes(CLASS_NAMES, m.get_text_pe(CLASS_NAMES))
    else:
        from ultralytics import YOLO
        m = YOLO(a.model + ".pt" if not a.model.endswith(".pt") else a.model)
    out, t0 = {}, time.time()
    for k, f in enumerate(files):
        r = m.predict(str(f), imgsz=a.imgsz, conf=0.001, iou=0.7, max_det=100, device=a.device, verbose=False)[0]
        dets = []
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            name = CLASS_NAMES[c] if yoloe else COCO2CS.get(c)
            if name is None: continue
            dets.append(dict(cls=name, box=[round(x, 1) for x in b], score=round(s, 4)))
            if (not yoloe) and a.rider_from_person and name == "person":
                dets.append(dict(cls="rider", box=dets[-1]["box"], score=dets[-1]["score"]))
        out[f.name.replace("_leftImg8bit.png", "")] = dets
        if k % 50 == 0: print(k, len(files), f"{time.time()-t0:.0f}s", flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w")); print("saved", a.out, len(out))


if __name__ == "__main__":
    main()
