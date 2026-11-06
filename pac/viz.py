"""Visual check: sample val frames, draw boxes coloured by assigned surface with depth text.
Usage: python -m pac.viz data/cityscapes 20 out_dir [rgb]"""
import json, sys, pathlib, random
import numpy as np, cv2
from .gt import decode_disparity, UNKNOWN

COL = {"road": (0, 200, 0), "off": (0, 0, 255), "unk": (0, 220, 255), "none": (255, 255, 255)}


def render(root, split, stem, rec, use_rgb=False, scale=0.5, zmax=45.0):
    city = stem.split("_")[0]
    base = (root / "leftImg8bit" / split / city / f"{stem}_leftImg8bit.png") if use_rgb else \
           (root / "gtFine" / split / city / f"{stem}_gtFine_color.png")
    img = cv2.imread(str(base))[:, :, :3] if base.exists() else None
    if img is None:
        return None
    disp = decode_disparity(cv2.imread(str(root / "disparity" / split / city / f"{stem}_disparity.png"), cv2.IMREAD_UNCHANGED))
    dv = np.nan_to_num(disp, nan=0.0); dimg = cv2.applyColorMap(np.clip(dv * 255 / 128, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    dimg[dv == 0] = 0
    outs = []
    for im in (img, dimg):
        im = im.copy()
        for o in rec["objects"]:
            if o["Z"] is not None and o["Z"] > zmax:
                continue
            x0, y0, x1, y1 = o["box"]
            k = "none" if o["Z"] is None else ("unk" if o["surface"] == UNKNOWN else ("road" if o["road"] else "off"))
            cv2.rectangle(im, (x0, y0), (x1, y1), COL[k], 2)
            cv2.circle(im, ((x0 + x1) // 2, y1), 4, COL[k], -1)
            txt = "?" if o["Z"] is None else f"{o['Z']:.0f}{'g' if o['depth_source']=='ground_plane' else ''}"
            cv2.putText(im, txt, (x0, max(y0 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[k], 2)
        outs.append(cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    return np.vstack(outs)


if __name__ == "__main__":
    root = pathlib.Path(sys.argv[1]); n = int(sys.argv[2]); out = pathlib.Path(sys.argv[3]); rgb = len(sys.argv) > 4
    out.mkdir(parents=True, exist_ok=True)
    A = json.load(open("data/boxes/val_attrs.json"))
    stems = sorted(A); random.Random(0).shuffle(stems)
    done = 0
    for s in stems:
        if sum(1 for o in A[s]["objects"] if o["Z"] and o["Z"] < 45) < 3:
            continue
        im = render(root, "val", s, A[s], rgb)
        if im is None: continue
        cv2.imwrite(str(out / f"{done:02d}_{s}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 88]); done += 1
        if done >= n: break
    print("wrote", done)
