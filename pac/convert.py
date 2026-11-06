"""Convert Cityscapes gtFine instance masks to boxes (JSON) + YOLO labels. Usage: python -m pac.convert data/cityscapes"""
import json, sys, pathlib
import numpy as np
from PIL import Image
from .gt import instances_to_boxes, CLASS_NAMES


def convert(root, out="data/boxes"):
    root, out = pathlib.Path(root), pathlib.Path(out)
    for split in ("train", "val"):
        recs = {}
        for p in sorted((root / "gtFine" / split).rglob("*_gtFine_instanceIds.png")):
            ids = np.array(Image.open(p))
            h, w = ids.shape
            objs, ign = instances_to_boxes(ids)
            stem = p.name.replace("_gtFine_instanceIds.png", "")
            recs[stem] = dict(w=w, h=h, objects=objs, ignore=ign, city=p.parent.name)
            lab = out / "yolo" / split / "labels"; lab.mkdir(parents=True, exist_ok=True)
            with open(lab / f"{stem}_leftImg8bit.txt", "w") as f:
                for o in objs:
                    x0, y0, x1, y1 = o["box"]
                    f.write(f"{CLASS_NAMES.index(o['cls'])} {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f}\n")
        out.mkdir(parents=True, exist_ok=True)
        json.dump(recs, open(out / f"{split}.json", "w"))
        n = sum(len(r["objects"]) for r in recs.values())
        print(split, len(recs), "images", n, "objects", sum(len(r["ignore"]) for r in recs.values()), "ignore regions")


if __name__ == "__main__":
    convert(sys.argv[1])
