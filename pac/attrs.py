"""Per-object attributes for a split: depth Z (+source), contact surface, lateral offset, height px. Cached to JSON.
Usage: python -m pac.attrs data/cityscapes val"""
import json, sys, pathlib
import numpy as np
from PIL import Image
from .gt import load_camera, decode_disparity, object_depth, contact_surface, is_road
from .criticality import lateral_offset


def pair_riders(objs):
    """Pair each bicycle/motorcycle with the rider on it (greedy by overlap >= 0.3 of the bike box); the pair is
    one road user, so the bike takes the rider's surface/road/X (consistent path weight). Sets pair_id on both."""
    riders = [o for o in objs if o["cls"] == "rider"]; used = set(); n = 0
    for b in sorted([o for o in objs if o["cls"] in ("bicycle", "motorcycle")], key=lambda o: o["mask_id"]):
        best, bi = 0.3, -1
        for i, r in enumerate(riders):
            if i in used: continue
            ix = max(0, min(b["box"][2], r["box"][2]) - max(b["box"][0], r["box"][0]))
            iy = max(0, min(b["box"][3], r["box"][3]) - max(b["box"][1], r["box"][1]))
            ov = ix * iy / max((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]), 1)
            if ov >= best: best, bi = ov, i
        if bi >= 0:
            used.add(bi); r = riders[bi]; n += 1
            b["pair_id"] = r["pair_id"] = n
            b["surface"], b["road"] = r["surface"], r["road"]


def build(root, split, boxes_dir="data/boxes"):
    root = pathlib.Path(root)
    recs = json.load(open(f"{boxes_dir}/{split}.json"))
    out = {}
    for k, (stem, r) in enumerate(recs.items()):
        city = r["city"]
        ids = np.array(Image.open(root / "gtFine" / split / city / f"{stem}_gtFine_instanceIds.png"))
        lab = np.array(Image.open(root / "gtFine" / split / city / f"{stem}_gtFine_labelIds.png"))
        disp = decode_disparity(np.array(Image.open(root / "disparity" / split / city / f"{stem}_disparity.png")))
        cam = load_camera(root / "camera" / split / city / f"{stem}_camera.json")
        objs = []
        for o in r["objects"]:
            Z, src = object_depth(disp, ids == o["mask_id"], cam, o["box"])
            surf = contact_surface(lab, o["box"])
            objs.append(dict(o, Z=Z if np.isfinite(Z) else None, depth_source=src, surface=surf,
                             road=bool(is_road(surf)), X=float(lateral_offset(o["box"], Z, cam)) if np.isfinite(Z) else None,
                             h_px=o["box"][3] - o["box"][1]))
        pair_riders(objs)
        out[stem] = dict(w=r["w"], h=r["h"], cam=cam, objects=objs, ignore=r["ignore"])
        if k % 100 == 0:
            print(split, k, flush=True)
    json.dump(out, open(f"{boxes_dir}/{split}_attrs.json", "w"))
    return out


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
