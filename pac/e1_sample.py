"""E1 step 1: sample 200 val GT objects (stratified: 4 distance bands x road/off-road, classes spread), render one image
per object with the box highlighted, write the blind labelling page. Annotators never see depth, surface or scores.
  python -m pac.e1_sample data/cityscapes data/e1 --seed 0
Outputs: data/e1/index.html + img/*.jpg (give to annotators; local only: Cityscapes images are not redistributable),
         data/e1_key.json (OUTSIDE the shared folder; private: object id -> stem, mask_id, cls, Z, surface; used only by e1_score)."""
import json, argparse, pathlib, random
import numpy as np, cv2

BANDS = [(0, 15), (15, 35), (35, 60), (60, 1e9)]
MIN_H = 12  # px: smaller boxes cannot be judged by eye


def sample(attrs, n_per=25, seed=0):
    rng = random.Random(seed); pool = {}
    for stem, A in sorted(attrs.items()):
        for o in A["objects"]:
            if o["Z"] is None or o["surface"] == -1 or o["h_px"] < MIN_H: continue
            b = next(i for i, (lo, hi) in enumerate(BANDS) if lo <= o["Z"] < hi)
            pool.setdefault((b, bool(o["road"])), []).append((stem, o))
    chosen = []
    for key in sorted(pool):
        by_cls = {}
        for stem, o in pool[key]: by_cls.setdefault(o["cls"], []).append((stem, o))
        for v in by_cls.values(): rng.shuffle(v)
        got, order = [], sorted(by_cls)
        while len(got) < n_per and any(by_cls[c] for c in order):   # round-robin over classes
            for c in order:
                if by_cls[c] and len(got) < n_per: got.append(by_cls[c].pop())
        chosen += [(key, s, o) for s, o in got]
    rng.shuffle(chosen)
    return chosen


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Criticality labelling</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--card:#f4f4f5;--yes:#1a7f37;--no:#b42318;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#9a9a9a;--card:#1c1c1e;--yes:#3fb950;--no:#f85149;--acc:#6ea8fe}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px system-ui,sans-serif}main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px}.q{background:var(--card);padding:12px 14px;border-radius:8px;margin:8px 0}
img{width:100%;border-radius:6px;display:block}.row{display:flex;gap:10px;margin:12px 0;flex-wrap:wrap}
button{flex:1;min-width:120px;padding:14px;font-size:18px;border:0;border-radius:8px;color:#fff;cursor:pointer}
#y{background:var(--yes)}#n{background:var(--no)}.g{background:var(--acc);flex:0 0 auto;min-width:90px;font-size:15px;padding:10px}
.m{color:var(--mut);font-size:14px}#bar{height:6px;background:var(--card);border-radius:3px}#fill{height:6px;background:var(--acc);border-radius:3px;width:0}
</style></head><body><main><h1>Which objects matter?</h1>
<div class=q><b>Driving this car at city speed, would you need to watch or react to the highlighted object in the next few seconds?</b>
<div class=m>Highlighted by the yellow box. Answer Yes or No. Keys: <b>Y</b>=Yes, <b>N</b>=No, <b>&larr;</b> back, <b>&rarr;</b> next.</div></div>
<div id=bar><div id=fill></div></div><p class=m id=pos></p><img id=im alt="scene with one highlighted object">
<div class=row><button id=y>Yes (Y)</button><button id=n>No (N)</button></div>
<div class=row><button class=g id=b>&larr; Back</button><button class=g id=f>Next &rarr;</button>
<button class=g id=x>Export labels</button><span class=m id=st></span></div>
<script>
const ITEMS=__ITEMS__;let A="",i=0,L={};const K="e1_labels_v1";
try{const s=JSON.parse(localStorage.getItem(K)||"{}");A=s.a||"";L=s.l||{};i=s.i||0}catch(e){}
if(!A){A=(prompt("Annotator name or initials (used only in the export filename)")||"anon").replace(/[^A-Za-z0-9_-]/g,"")}
function save(){try{localStorage.setItem(K,JSON.stringify({a:A,l:L,i:i}))}catch(e){}}
function show(){i=Math.max(0,Math.min(ITEMS.length-1,i));im.src=ITEMS[i];const d=Object.keys(L).length;
pos.textContent="Object "+(i+1)+" of "+ITEMS.length+" - answered "+d+(L[ITEMS[i]]!==undefined?" - your answer here: "+(L[ITEMS[i]]?"Yes":"No"):"");
fill.style.width=(100*d/ITEMS.length)+"%";save()}
function ans(v){L[ITEMS[i]]=v;i++;show()}
y.onclick=()=>ans(1);n.onclick=()=>ans(0);b.onclick=()=>{i--;show()};f.onclick=()=>{i++;show()};
x.onclick=()=>{const o={annotator:A,n:Object.keys(L).length,labels:L,exported:new Date().toISOString()};
const u=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{type:"application/json"}));const a=document.createElement("a");
a.href=u;a.download="labels_"+A+".json";a.click();st.textContent="Exported "+o.n+" labels"};
addEventListener("keydown",e=>{const k=e.key.toLowerCase();if(k=="y")ans(1);else if(k=="n")ans(0);else if(e.key=="ArrowLeft"){i--;show()}else if(e.key=="ArrowRight"){i++;show()}});
const im=document.getElementById("im"),pos=document.getElementById("pos"),fill=document.getElementById("fill"),st=document.getElementById("st");
show();
</script></main></body></html>"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); root, out = pathlib.Path(a.root), pathlib.Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    attrs = json.load(open("data/boxes/val_attrs.json")); chosen = sample(attrs, 25, a.seed)
    key, items = {}, []
    for k, ((b, road), stem, o) in enumerate(chosen):
        city = stem.split("_")[0]
        im = cv2.imread(str(root / "leftImg8bit" / "val" / city / f"{stem}_leftImg8bit.png"))
        x0, y0, x1, y1 = o["box"]; pad = 4
        cv2.rectangle(im, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), (0, 255, 255), 4)
        im = cv2.resize(im, (1280, 640), interpolation=cv2.INTER_AREA)
        oid = f"o{k:03d}"; cv2.imwrite(str(out / "img" / f"{oid}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        items.append(f"img/{oid}.jpg")
        key[f"img/{oid}.jpg"] = dict(stem=stem, mask_id=o["mask_id"], cls=o["cls"], Z=o["Z"], road=bool(o["road"]), band=b, h_px=o["h_px"],
                                      X=o["X"], box=o["box"])
    (out / "index.html").write_text(PAGE.replace("__ITEMS__", json.dumps(items)))
    json.dump(key, open(out.parent / "e1_key.json", "w"), indent=1)
    from collections import Counter
    print("objects", len(chosen), "strata", dict(Counter((b, r) for (b, r), _, _ in chosen)))
    print("classes", dict(Counter(o["cls"] for _, _, o in chosen)))


if __name__ == "__main__":
    main()
