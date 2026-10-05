"""Self-contained interactive HTML viewer for one hisim_thermal result directory.

    from hisim_thermal.viewer import write_viewer
    write_viewer(result_dir)                # -> <result_dir>/viewer.html

The page embeds all data (no CDN, works offline):
  * left  : clickable stack cross-section, one band per physical layer
            (name, thickness, k range); band heights are log-scaled.
  * right : the selected layer's thickness-weighted mean T map and, when the
            layer carries power, its areal power density map (W/cm^2), both
            drawn cell-by-cell on the true non-uniform x/y edges at true
            aspect ratio, with tile outlines, peak markers and a hover readout.
Planes larger than MAX_PLANE cells per axis are block-averaged (area weighted).
Plan-view convention: x to the right, y downwards, origin = package top-left.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

MAX_PLANE = 400
__all__ = ["write_viewer", "build_payload"]


# --------------------------------------------------------------------------- helpers
def _fmt(v, sig=5):
    if v is None or not np.isfinite(v):
        return "null"
    if v == 0:
        return "0"
    s = "%.*g" % (sig, v)
    if "e" in s:
        m, e = s.split("e")
        s = m + "e" + str(int(e))
    elif "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


class _Raw(str):
    """Pre-serialised JSON fragment."""


def _arr(a, sig=5):
    return _Raw("[" + ",".join(_fmt(float(v), sig) for v in np.ravel(a)) + "]")


def _iarr(a):
    return _Raw("[" + ",".join(str(int(v)) for v in np.ravel(a)) + "]")


def _dump(o):
    if isinstance(o, _Raw):
        return str(o)
    if isinstance(o, dict):
        return "{" + ",".join(json.dumps(str(k)) + ":" + _dump(v) for k, v in o.items()) + "}"
    if isinstance(o, (list, tuple)):
        return "[" + ",".join(_dump(v) for v in o) + "]"
    if isinstance(o, (float, np.floating)):
        return _fmt(float(o), 7)
    if isinstance(o, (np.integer,)):
        return str(int(o))
    return json.dumps(o)


def _load_json(d, name, default=None):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def _block_edges(e, f):
    idx = list(range(0, len(e) - 1, f)) + [len(e) - 1]
    return e[idx], idx


def _block_mean(a, w, iy, ix):
    """Weighted block mean of a 2-D plane with weights w on blocks iy x ix."""
    aw = np.add.reduceat(np.add.reduceat(a * w, iy[:-1], 0), ix[:-1], 1)
    ws = np.add.reduceat(np.add.reduceat(w, iy[:-1], 0), ix[:-1], 1)
    return aw / np.where(ws > 0, ws, 1)


# --------------------------------------------------------------------------- payload
def build_payload(d):
    g = np.load(os.path.join(d, "grid.npz"))
    xe, ye = np.asarray(g["x_edges"], float), np.asarray(g["y_edges"], float)
    T = np.load(os.path.join(d, "T_full.npy"))
    P = np.load(os.path.join(d, "P_full.npy"))
    K = np.load(os.path.join(d, "K_full.npy"))
    N = np.load(os.path.join(d, "N_full.npy"))
    lay = _load_json(d, "layers.json")
    regions = _load_json(d, "regions.json", {})
    meta = _load_json(d, "meta.json", {})
    peaks = _load_json(d, "peaks.json", {})
    tiles_out = _load_json(d, "tiles.json", []) or []
    inp = _load_json(d, "input.json", {}) or {}
    bc = (meta.get("config") or {}).get("bc") or {}
    T_amb = float(bc.get("T_amb", float(T.min())))

    nk, ny, nx = T.shape
    subs = lay["sublayers"]
    dz = np.array([s["dz_m"] for s in subs], float)
    phys_of_k = [s["phys"] for s in subs]

    fy, fx = max(1, math.ceil(ny / MAX_PLANE)), max(1, math.ceil(nx / MAX_PLANE))
    ye_d, iy = _block_edges(ye, fy)
    xe_d, ix = _block_edges(xe, fx)
    iy, ix = np.array(iy), np.array(ix)
    A = np.outer(np.diff(ye), np.diff(xe))                     # m^2
    down = fy > 1 or fx > 1

    def ds(a, w=A):
        return _block_mean(a, w, iy, ix) if down else a

    layers = []
    for pl in lay["phys_layers"]:
        k0, k1 = pl["k_range"]
        w = dz[k0:k1, None, None]
        Tl = (T[k0:k1] * w).sum(0) / w.sum()
        Keff = w.sum() / (w / K[k0:k1]).sum(0)                  # series (vertical) k
        Pcol = P[k0:k1].sum(0)                                  # W per column
        qf = os.path.join(d, "qsurf_%s.npy" % pl["name"])
        if os.path.exists(qf):
            q = np.load(qf).astype(float)
        elif Pcol.max() > 0:
            q = Pcol / A * 1e-4
        else:
            q = None
        Nl = N[(k0 + k1 - 1) // 2]
        if down:
            Nl = Nl[iy[:-1]][:, ix[:-1]]                        # top-left cell
            qd = None if q is None else _block_mean(q, A, iy, ix)
        else:
            qd = q
        Tsub = T[k0:k1]
        layers.append({
            "name": pl["name"], "role": pl["role"], "die": pl.get("die"),
            "tier": pl.get("tier"), "k0": k0, "k1": k1,
            "t_um": pl["thickness_m"] * 1e6,
            "z_um": subs[k0]["z_top_m"] * 1e6,
            "Tmin": float(Tsub.min()), "Tmax": float(Tsub.max()),
            "Tavg_min": float(Tl.min()), "Tavg_max": float(Tl.max()),
            "kmin": float(K[k0:k1].min()), "kmax": float(K[k0:k1].max()),
            "P_W": float(Pcol.sum()),
            "mats": sorted({regions.get(str(int(c)), str(int(c))) for c in np.unique(N[k0:k1])}),
            "dT": _arr(ds(Tl) - T_amb, 5),
            "k": _arr(ds(Keff), 4),
            "q": None if qd is None else _arr(qd, 4),
            "n": _iarr(Nl),
        })
    return dict(
        xe=xe_d, ye=ye_d, xe_mm=xe_d * 1e3, ye_mm=ye_d * 1e3, nx=nx, ny=ny,
        down=(fy, fx), T_amb=T_amb, layers=layers, regions=regions,
        tiles=_tiles(inp, tiles_out), peaks=_peaks(peaks, phys_of_k),
        meta=meta, bc=bc, inp=inp)


def _tiles(inp, tiles_out):
    """All tile rectangles of the neutral input, joined with solved T stats."""
    stats = {t["id"]: t for t in tiles_out}
    src = inp.get("tiles") or [
        dict(id=t["id"], kind=t["kind"], tier=t["tier"], chiplet=t["chiplet"],
             x0=t["x0_mm"], y0=t["y0_mm"], w=t["w_mm"], h=t["h_mm"], layer="device")
        for t in tiles_out]
    out = []
    for t in src:
        s = stats.get(t["id"], {})
        p = s.get("P_W", (t.get("p_dyn_W") or 0) + (t.get("p_leak_W") or 0))
        out.append([t["id"], t["kind"], int(t.get("tier") or 0), t.get("layer", "device"),
                    float(t["x0"]), float(t["y0"]), float(t["w"]), float(t["h"]),
                    s.get("T_mean_K"), s.get("T_max_K"), float(p or 0.0),
                    int(t.get("chiplet") or 0)])
    return out


def _peaks(peaks, phys_of_k):
    """Valid peak classes, merged when several classes share one cell."""
    out, at = [], {}
    for cls, v in (peaks or {}).items():
        if not v or not v.get("valid"):
            continue
        key = (v["k"], v["iy"], v["ix"])
        if key in at:
            at[key]["cls"].append(cls)
            continue
        e = dict(cls=[cls], T=float(v["T_K"]), x=float(v["x_mm"]), y=float(v["y_mm"]),
                 z=float(v["z_um"]), k=int(v["k"]),
                 layer=phys_of_k[int(v["k"])] if 0 <= int(v["k"]) < len(phys_of_k) else v.get("layer"),
                 tile=v.get("tile"))
        at[key] = e
        out.append(e)
    return out


def _title(meta, d):
    i = meta.get("input") or {}
    am = i.get("adapter_meta") or {}
    parts = [str(i.get("arch", "?")), "%s tier" % i.get("n_tier", "?"),
             "%s chiplet" % i.get("chiplet_num", "?"), str(i.get("source", ""))]
    if am.get("power_mode"):
        parts.append(am["power_mode"])
    return " · ".join(p for p in parts if p), os.path.basename(os.path.normpath(d))


def write_viewer(dir, out_html=None, title=None):
    """Write one self-contained HTML viewer for result directory `dir`."""
    d = dir
    pl = build_payload(d)
    meta = pl["meta"]
    head, case = _title(meta, d)
    i = meta.get("input") or {}
    bud = i.get("budget_W") or {}
    eb = meta.get("energy_balance") or {}
    data = {
        "title": title or case, "subtitle": head,
        "xe": _arr(pl["xe_mm"], 7), "ye": _arr(pl["ye_mm"], 7),
        "nx": pl["nx"], "ny": pl["ny"], "down": list(pl["down"]),
        "T_amb": pl["T_amb"], "bc": pl["bc"], "bbox": i.get("bbox_mm"),
        "P_total": bud.get("total"), "energy": eb,
        "version": meta.get("hisim_thermal_version"), "preset": meta.get("preset"),
        "layers": pl["layers"], "regions": pl["regions"],
        "tiles": pl["tiles"], "peaks": pl["peaks"],
    }
    js = _dump(data).replace("</", "<\\/")
    html = _TEMPLATE.replace("/*__DATA__*/null", js).replace("__TITLE__", _esc(data["title"]))
    out = out_html or os.path.join(d, "viewer.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--title")
    a = ap.parse_args(argv)
    print(write_viewer(a.dir, a.out, a.title))


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · hisim_thermal</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d232b;--mute:#5d6875;--line:#dde1e6;--acc:#2563eb;--sel:#111;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{padding:12px 18px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px 24px;align-items:baseline}
header h1{font-size:17px;margin:0;font-weight:650}
header .sub{color:var(--mute)}
header .kv{color:var(--mute);font-size:12px}
header .kv b{color:var(--ink);font-weight:600}
main{display:grid;grid-template-columns:minmax(260px,330px) 1fr;gap:14px;padding:14px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);margin:0 0 8px;font-weight:600}
#stack svg{width:100%;display:block}
#stack .band{cursor:pointer}
#stack .band:hover rect.bg{stroke:#555;stroke-width:1}
.note{color:var(--mute);font-size:11.5px;margin-top:6px}
.ctl{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-bottom:10px}
.ctl label{display:inline-flex;gap:4px;align-items:center;white-space:nowrap}
.ctl input[type=number]{width:84px;font:inherit;padding:2px 4px;border:1px solid var(--line);border-radius:4px}
.ctl button{font:inherit;padding:3px 10px;border:1px solid var(--line);border-radius:5px;background:#fafbfc;cursor:pointer}
.ctl button:hover{background:#eef2f7}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{border:0;border-radius:0;border-right:1px solid var(--line)}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--acc);color:#fff}
#lhead{display:flex;flex-wrap:wrap;gap:4px 18px;align-items:baseline;margin-bottom:8px}
#lhead .nm{font-size:16px;font-weight:650}
#lhead span{color:var(--mute)} #lhead span b{color:var(--ink);font-weight:600}
.maps{display:flex;flex-wrap:wrap;gap:16px}
.map{flex:1 1 420px;min-width:300px}
.map .ttl{font-weight:600;margin-bottom:4px}
.map .ttl small{color:var(--mute);font-weight:400}
.mapwrap{display:flex;gap:8px;align-items:stretch}
.mapwrap canvas.m{flex:1 1 auto;min-width:0;cursor:crosshair;border:1px solid var(--line);background:#fff}
canvas.cb{flex:0 0 64px}
#readout{font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#f3f5f8;border:1px solid var(--line);border-radius:6px;padding:8px 10px;min-height:5.2em;white-space:pre;overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:left;padding:3px 6px;border-bottom:1px solid var(--line)}
th{color:var(--mute);font-weight:600}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.pk{cursor:pointer} tr.pk:hover{background:#eef2f7}
.empty{display:flex;align-items:center;justify-content:center;color:var(--mute);border:1px dashed var(--line);min-height:160px;border-radius:6px}
</style></head>
<body>
<header><h1 id="h1"></h1><div class="sub" id="sub"></div><div class="kv" id="kv"></div></header>
<main>
 <div>
  <div class="card" id="stack"><h2>Stack cross-section</h2><div id="svgbox"></div>
   <div class="note">Band heights are log-scaled in thickness (not to scale). Right strip: layer peak T on the shared scale. Click a band to select it.</div></div>
  <div class="card" style="margin-top:14px"><h2>Peaks</h2><div id="peaks"></div></div>
 </div>
 <div class="card">
  <div id="lhead"></div>
  <div class="ctl">
   <span>Colour scale</span>
   <span class="seg" id="mode"><button data-m="auto" class="on">Auto (layer)</button><button data-m="shared">Shared (all layers)</button><button data-m="manual">Manual</button></span>
   <label>T min <input type="number" id="tmin" step="any"></label><label>T max <input type="number" id="tmax" step="any"></label>
   <label>q min <input type="number" id="qmin" step="any"></label><label>q max <input type="number" id="qmax" step="any"></label>
  </div>
  <div class="ctl">
   <label><input type="checkbox" id="oTiles" checked> tile outlines</label>
   <label><input type="checkbox" id="oSmall"> include strips (router/TSV/AIB)</label>
   <label><input type="checkbox" id="oPeaks" checked> peak markers</label>
   <label><input type="checkbox" id="oGrid"> grid lines</label>
   <label><input type="checkbox" id="oLog"> log power density</label>
   <button id="reset">Reset zoom</button>
   <span class="note" style="margin:0">Wheel = zoom, drag = pan (both maps linked)</span>
  </div>
  <div class="maps">
   <div class="map"><div class="ttl">Temperature <small id="tsub">thickness-weighted mean over the layer, K</small></div>
    <div class="mapwrap"><canvas class="m" id="cT"></canvas><canvas class="cb" id="cbT"></canvas></div></div>
   <div class="map"><div class="ttl">Power density <small id="qsub">W/cm²</small></div>
    <div class="mapwrap" id="qwrap"><canvas class="m" id="cQ"></canvas><canvas class="cb" id="cbQ"></canvas></div>
    <div class="empty" id="qnone" style="display:none">No power is dissipated in this layer</div></div>
  </div>
  <div id="readout">Hover over a map.</div>
  <div class="note" id="dsnote"></div>
 </div>
</main>
<script>
const D=/*__DATA__*/null;
const $=id=>document.getElementById(id);
const CM={
 inferno:"0000040b0724210c4a3d096557106e71196e8a226aa32c61bc3754d24644e45a31f1731df98e09fcac11f9cb35f2ea69fcffa4",
 viridis:"44015448186a472d7b4240863b528b33638d2c728e26828e21918c1fa08828ae803fbc735ec96284d44baddc30d8e219fde725"};
function lut(hex){const s=[];for(let i=0;i<hex.length;i+=6)s.push([0,2,4].map(o=>parseInt(hex.substr(i+o,2),16)));
 const L=[];for(let i=0;i<256;i++){const t=i/255*(s.length-1),a=Math.min(s.length-2,Math.floor(t)),f=t-a;
  L.push('rgb('+s[a].map((v,c)=>Math.round(v+(s[a+1][c]-v)*f)).join(',')+')');}return L;}
const LUT={T:lut(CM.inferno),q:lut(CM.viridis)};
const NX=D.xe.length-1,NY=D.ye.length-1,W=D.xe[NX],H=D.ye[NY];
const REG=D.regions, L=D.layers, TA=D.T_amb;
function fmt(v,d){if(v==null||!isFinite(v))return'–';const a=Math.abs(v);
 if(a!==0&&(a<1e-3||a>=1e6))return v.toExponential(2);return v.toFixed(d==null?(a>=100?2:a>=1?3:4):d);}
function sig(v,n){if(v==null||!isFinite(v))return'–';if(v===0)return'0';const a=Math.abs(v);return(a<1e-3||a>=1e6)?v.toExponential(n-1):(+v.toPrecision(n)).toString();}
function fmtT(um){return um>=1000?(um/1000).toPrecision(3)+' mm':(um>=10?um.toPrecision(3):um.toPrecision(2))+' µm';}
function fmtK(a,b){const f=v=>v>=10?v.toFixed(0):v.toPrecision(2);return Math.abs(a-b)<1e-6*b?f(a):f(a)+'–'+f(b);}
function nice(lo,hi,n){const sp=(hi-lo)||1,r=sp/n,p=Math.pow(10,Math.floor(Math.log10(r))),m=r/p,
 st=(m<1.5?1:m<3?2:m<7?5:10)*p,t=[];for(let v=Math.ceil(lo/st)*st;v<=hi+st*1e-6;v+=st)t.push(+v.toPrecision(12));return t;}
// per-layer stats of the embedded planes
L.forEach(l=>{let a=Infinity,b=-Infinity;for(const v of l.dT){if(v<a)a=v;if(v>b)b=v;}l.mapTmin=TA+a;l.mapTmax=TA+b;
 if(l.q){let c=Infinity,e=-Infinity,p=Infinity;for(const v of l.q){if(v<c)c=v;if(v>e)e=v;if(v>0&&v<p)p=v;}l.qmin=c;l.qmax=e;l.qpos=isFinite(p)?p:1e-6;
  const pos=[];let At=0;for(let j=0;j<D.ye.length-1;j++)for(let i=0;i<D.xe.length-1;i++){const v=l.q[j*(D.xe.length-1)+i];
   if(v>0){const a=(D.xe[i+1]-D.xe[i])*(D.ye[j+1]-D.ye[j]);pos.push([v,a]);At+=a;}}
  pos.sort((a,b)=>a[0]-b[0]);let c2=0;l.qmed=0;for(const [v,a] of pos){c2+=a;if(c2>=At/2){l.qmed=v;break;}}}});  // area-weighted median
const SH={Tmin:Math.min(...L.map(l=>l.mapTmin)),Tmax:Math.max(...L.map(l=>l.mapTmax)),
 qmin:0,qmax:Math.max(0,...L.filter(l=>l.q).map(l=>l.qmax)),qpos:Math.min(1e30,...L.filter(l=>l.q).map(l=>l.qpos))};
const ST={sel:0,mode:'auto',view:{x0:0,y0:0,s:1},manual:null,hover:null};
{const pk=D.peaks.find(p=>p.cls.includes('domain'))||D.peaks[0];
 const i=pk?L.findIndex(l=>l.name===pk.layer):-1;ST.sel=i>=0?i:Math.max(0,L.findIndex(l=>l.role==='active'));}
const ROLE={spreader:'#d9a273',tim1:'#c9c9c9',tim:'#c9c9c9',beol:'#a8c7e6',active:'#e8826b',si:'#b7bfcc',
 bond:'#c3b1df',c4:'#d8c79a',emib:'#8cc7bd',subs:'#9fc39a',interposer:'#9fb7c9',mold:'#bbb3a6'};
function tierTiles(l){ // which tile rectangles belong to (or project onto) layer l
 if(l.role==='emib')return D.tiles.filter(t=>t[3]==='emib');
 let tier=l.tier;
 if(tier==null){const dies=L.filter(x=>x.tier!=null);if(!dies.length)return[];
  tier=(l.k1<=dies[0].k0)?dies[0].tier:dies[dies.length-1].tier;}
 return D.tiles.filter(t=>t[2]===tier&&t[3]!=='emib');}
function drawStack(){
 const w=300,hs=L.map(l=>Math.max(17,15+9*Math.log10(Math.max(1,l.t_um)))),tot=hs.reduce((a,b)=>a+b,0);
 const lo=SH.Tmin,hi=Math.max(SH.Tmax,lo+1e-9);let y=0,s='';
 L.forEach((l,i)=>{const h=hs[i],c=ROLE[l.role]||'#ddd',on=i===ST.sel,ci=Math.round(255*Math.min(1,Math.max(0,(l.Tmax-lo)/(hi-lo))));
  const fs=h<22?10.5:11.5,two=h>=30;
  s+=`<g class="band" data-i="${i}"><title>${l.name} · ${fmtT(l.t_um)} · k ${fmtK(l.kmin,l.kmax)} W/m·K · materials: ${l.mats.join(', ')}</title>`+
   `<rect class="bg" x="0" y="${y}" width="${w-36}" height="${h}" fill="${c}" stroke="${on?'#111':'#fff'}" stroke-width="${on?2.5:1}"/>`+
   `<rect x="${w-34}" y="${y+1}" width="34" height="${h-2}" fill="${LUT.T[ci]}"/>`;
  const tx=`<tspan font-weight="650">${l.name}</tspan>`, meta=`${fmtT(l.t_um)} · k ${fmtK(l.kmin,l.kmax)}`+(l.P_W>0?` · ${fmt(l.P_W,l.P_W>=10?1:3)} W`:'');
  if(two)s+=`<text x="7" y="${y+h/2-2}" font-size="${fs}">${tx}</text><text x="7" y="${y+h/2+11}" font-size="10.5" fill="#333">${meta}</text>`;
  else s+=`<text x="7" y="${y+h/2+4}" font-size="${fs}">${tx}<tspan dx="8" font-size="10.5" fill="#333">${meta}</tspan></text>`;
  s+=`<text x="${w-17}" y="${y+h/2+3.5}" font-size="9.5" text-anchor="middle" fill="${ci>150?'#000':'#fff'}">${l.Tmax.toFixed(1)}</text></g>`;
  y+=h;});
 $('svgbox').innerHTML=`<svg viewBox="-1 -1 ${w+2} ${tot+2}" font-family="system-ui,sans-serif">${s}</svg>`;
 $('svgbox').querySelectorAll('.band').forEach(g=>g.onclick=()=>select(+g.dataset.i));}
function bis(e,v){let a=0,b=e.length-1;if(v<e[0]||v>e[b])return-1;while(b-a>1){const m=(a+b)>>1;if(e[m]<=v)a=m;else b=m;}return a;}
function range(kind){const l=L[ST.sel];
 if(ST.mode==='manual'){const a=parseFloat($(kind+'min').value),b=parseFloat($(kind+'max').value);if(isFinite(a)&&isFinite(b)&&b>a&&(kind==='T'||!$('oLog').checked||a>0))return[a,b];}
 if(kind==='T')return ST.mode==='shared'?[SH.Tmin,SH.Tmax]:[l.mapTmin,l.mapTmax];
 if(!l.q)return[0,1];const hi=ST.mode==='shared'?SH.qmax:l.qmax;
 if($('oLog').checked)return[Math.max(ST.mode==='shared'?SH.qpos:l.qpos,hi*1e-4),hi];return[ST.mode==='shared'?SH.qmin:Math.min(0,l.qmin),hi];}
const SB=24;
function geom(cv){const r=cv.getBoundingClientRect(),cw=r.width,ch=r.height-SB,s0=Math.min(cw/W,ch/H),v=ST.view,s=s0*v.s;
 return{cw,ch,s,ox:cw/2-(v.x0)*s,oy:ch/2-(v.y0)*s};}
function sizeCanvas(cv,cb){const box=cv.parentElement.getBoundingClientRect(),cw=Math.max(200,box.width-72),
 ch=SB+Math.min(Math.max(200,cw*H/W),Math.max(300,window.innerHeight*0.7)),dpr=window.devicePixelRatio||1;
 for(const [c,w] of [[cv,cw],[cb,64]]){c.style.width=w+'px';c.style.height=ch+'px';c.width=Math.round(w*dpr);c.height=Math.round(ch*dpr);
  c.getContext('2d').setTransform(dpr,0,0,dpr,0,0);}}
const ZC='#c8c9ce';   // log maps: cells with zero power (final audit viewer_b F1: they looked like the lowest decade)
function drawMap(cv,vals,off,lo,hi,lut,tiles,lg){const ctx=cv.getContext('2d'),g=geom(cv),X=D.xe,Y=D.ye;
 ctx.clearRect(0,0,g.cw,g.ch+SB);ctx.save();ctx.beginPath();ctx.rect(0,0,g.cw,g.ch);ctx.clip();ctx.fillStyle='#f0f1f3';ctx.fillRect(0,0,g.cw,g.ch);
 const px=x=>Math.round(g.ox+x*g.s),py=y=>Math.round(g.oy+y*g.s),sp=(hi-lo)||1,LA=lg?Math.log10(lo):0,LS=lg?(Math.log10(hi)-LA)||1:1;
 const i0=Math.max(0,bis(X,Math.max(X[0],-g.ox/g.s))),i1=Math.min(NX-1,Math.max(0,bis(X,Math.min(W,(g.cw-g.ox)/g.s)))),
  j0=Math.max(0,bis(Y,Math.max(Y[0],-g.oy/g.s))),j1=Math.min(NY-1,Math.max(0,bis(Y,Math.min(H,(g.ch-g.oy)/g.s))));
 const XP=[];for(let i=i0;i<=i1+1;i++)XP.push(px(X[i]));
 for(let j=j0;j<=j1;j++){const ya=py(Y[j]),yb=py(Y[j+1]);if(yb<=ya)continue;
  for(let i=i0;i<=i1;i++){const xa=XP[i-i0],xb=XP[i-i0+1];if(xb<=xa)continue;
   const v=vals[j*NX+i]+off;if(lg&&!(v>0)){ctx.fillStyle=ZC;ctx.fillRect(xa,ya,xb-xa,yb-ya);continue;}let t;if(lg)t=(Math.log10(v)-LA)/LS;else t=(v-lo)/sp;t=t<0?0:t>1?1:t;ctx.fillStyle=lut[Math.round(t*255)];ctx.fillRect(xa,ya,xb-xa,yb-ya);}}
 if($('oGrid').checked){ctx.strokeStyle='rgba(255,255,255,.25)';ctx.lineWidth=0.5;ctx.beginPath();
  for(let i=i0;i<=i1+1;i++){ctx.moveTo(px(X[i])+.25,py(Y[j0]));ctx.lineTo(px(X[i])+.25,py(Y[j1+1]));}
  for(let j=j0;j<=j1+1;j++){ctx.moveTo(px(X[i0]),py(Y[j])+.25);ctx.lineTo(px(X[i1+1]),py(Y[j])+.25);}ctx.stroke();}
 if($('oTiles').checked){const all=$('oSmall').checked;
  for(const t of tiles){const big=!['router','tsv','aib_tx','aib_rx','gap'].includes(t[1]);if(!big&&!all)continue;
   ctx.strokeStyle=big?'rgba(255,255,255,.9)':'rgba(120,220,255,.9)';ctx.lineWidth=big?1:0.7;
   ctx.strokeRect(g.ox+t[4]*g.s+.5,g.oy+t[5]*g.s+.5,Math.max(1,t[6]*g.s-1),Math.max(1,t[7]*g.s-1));}}
 ctx.strokeStyle='#555';ctx.lineWidth=1;ctx.strokeRect(px(0)-.5,py(0)-.5,px(W)-px(0)+1,py(H)-py(0)+1);
 if($('oPeaks').checked){const l=L[ST.sel];ctx.font='600 11px system-ui,sans-serif';
  for(const p of D.peaks){if(p.layer!==l.name)continue;const x=g.ox+p.x*g.s,y=g.oy+p.y*g.s;
   ctx.lineWidth=3;ctx.strokeStyle='#000';ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.stroke();
   ctx.lineWidth=1.6;ctx.strokeStyle='#fff';ctx.beginPath();ctx.arc(x,y,6,0,7);ctx.moveTo(x-10,y);ctx.lineTo(x-3,y);ctx.moveTo(x+3,y);ctx.lineTo(x+10,y);ctx.stroke();
   const lab=p.cls[0]+(p.cls.length>1?' +'+(p.cls.length-1):'')+'  '+p.T.toFixed(2)+' K',tw=ctx.measureText(lab).width,lx=x+9+tw>g.cw-4?x-9-tw:x+9,ly=y-8<12?y+18:y-8;
   ctx.lineWidth=3;ctx.strokeStyle='rgba(0,0,0,.75)';ctx.strokeText(lab,lx,ly);ctx.fillStyle='#fff';ctx.fillText(lab,lx,ly);}}
 if(ST.hover){const x=g.ox+ST.hover[0]*g.s,y=g.oy+ST.hover[1]*g.s;ctx.strokeStyle='rgba(0,180,255,.9)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,g.ch);ctx.moveTo(0,y);ctx.lineTo(g.cw,y);ctx.stroke();}
 ctx.restore();ctx.fillStyle='#fff';ctx.fillRect(0,g.ch,g.cw,SB);ctx.fillStyle='#d0d4da';ctx.fillRect(0,g.ch,g.cw,1);
 const vx0=Math.max(0,-g.ox/g.s),vx1=Math.min(W,(g.cw-g.ox)/g.s),vy0=Math.max(0,-g.oy/g.s),vy1=Math.min(H,(g.ch-g.oy)/g.s);
 ctx.fillStyle='#555';ctx.font='11px system-ui,sans-serif';ctx.fillText(`view x ${vx0.toFixed(2)}–${vx1.toFixed(2)} mm · y ${vy0.toFixed(2)}–${vy1.toFixed(2)} mm (y down)`,6,g.ch+16);
 // scale bar
 const tgt=g.cw*0.16/g.s,p10=Math.pow(10,Math.floor(Math.log10(tgt))),bar=[1,2,5,10].map(m=>m*p10).filter(v=>v<=tgt).pop()||p10,bl=bar*g.s;
 const bx=g.cw-bl-70;ctx.fillStyle='#111';ctx.fillRect(bx,g.ch+10,bl,3);ctx.fillRect(bx,g.ch+7,1,9);ctx.fillRect(bx+bl-1,g.ch+7,1,9);
 ctx.font='11px system-ui,sans-serif';ctx.fillText((bar>=1?bar:+(bar*1000).toPrecision(3))+(bar>=1?' mm':' µm'),bx+bl+6,g.ch+16);}
function drawCB(cb,lo,hi,lut,unit,lg){const ctx=cb.getContext('2d'),w=64,h=parseFloat(cb.style.height)-SB,t=14,b=h-10;
 ctx.clearRect(0,0,w,h+SB);for(let y=t;y<b;y++){ctx.fillStyle=lut[Math.round(255*(b-y)/(b-t))];ctx.fillRect(2,y,14,1.2);}
 ctx.strokeStyle='#666';ctx.strokeRect(2.5,t+.5,13,b-t-1);ctx.fillStyle='#333';ctx.font='10.5px system-ui,sans-serif';
 ctx.fillText(unit,1,10);const sp=(hi-lo)||1;
 if(lg){const a=Math.log10(lo),c=Math.log10(hi),s2=(c-a)||1;for(let e=Math.ceil(a-1e-9);e<=c+1e-9;e++){const y=b-(e-a)/s2*(b-t);ctx.fillRect(16,y-.5,4,1);ctx.fillText(e>=-2&&e<=3?(+Math.pow(10,e).toPrecision(1)).toString():'1e'+e,22,y+3.5);}
  ctx.fillText('log',1,h-1);ctx.fillStyle=ZC;ctx.fillRect(2,h+6,13,9);ctx.fillStyle='#333';ctx.fillText('= 0',18,h+14);return;}
 for(const v of nice(lo,hi,6)){const y=b-(v-lo)/sp*(b-t);ctx.fillRect(16,y-.5,4,1);ctx.fillText(+v.toPrecision(6)+'',22,y+3.5);}}
let TILES=[];
function render(){const l=L[ST.sel],[a,b]=range('T');
 drawMap($('cT'),l.dT,TA,a,b,LUT.T,TILES);drawCB($('cbT'),a,b,LUT.T,'T [K]');
 if(l.q){const lg=$('oLog').checked,[c,d]=range('q');drawMap($('cQ'),l.q,0,c,d,LUT.q,TILES,lg);drawCB($('cbQ'),c,d,LUT.q,'W/cm²',lg);}}
function layout(){sizeCanvas($('cT'),$('cbT'));if(L[ST.sel].q)sizeCanvas($('cQ'),$('cbQ'));render();}
function select(i){ST.sel=i;const l=L[i];TILES=tierTiles(l);ST.hover=null;
 const hasq=!!l.q;$('qwrap').style.display=hasq?'flex':'none';$('qnone').style.display=hasq?'none':'flex';
 $('qsub').textContent=hasq?(l.role==='active'?'areal heat flux q_surf, W/cm²':'layer-integrated power / cell area, W/cm²'):'';
 $('lhead').innerHTML=`<span class="nm">${l.name}</span><span>role <b>${l.role}</b></span>`+
  (l.tier!=null?`<span>tier <b>${l.tier}</b></span>`:'')+
  `<span>z <b>${fmt(l.z_um,1)}–${fmt(l.z_um+l.t_um,1)}</b> µm</span><span>t <b>${fmtT(l.t_um)}</b> (${l.k1-l.k0} sub-layer${l.k1-l.k0>1?'s':''})</span>`+
  `<span>k <b>${fmtK(l.kmin,l.kmax)}</b> W/m·K</span><span>T cells <b>${l.Tmin.toFixed(2)}–${l.Tmax.toFixed(2)}</b> K</span>`+
  (l.P_W>0?`<span>P <b>${fmt(l.P_W,3)}</b> W</span>`:'')+`<span>materials <b>${l.mats.join(', ')}</b></span>`;
 if(ST.mode!=='manual'){const[a,b]=range('T');$('tmin').value=+a.toPrecision(6);$('tmax').value=+b.toPrecision(6);
  if(hasq){const[c,d]=range('q');$('qmin').value=+c.toPrecision(3);$('qmax').value=+d.toPrecision(4);}}
 drawStack();drawPeaks();layout();$('readout').textContent='Hover over a map.';}
function tileAt(x,y){let best=null,ba=Infinity;for(const t of TILES){if(x>=t[4]&&x<=t[4]+t[6]&&y>=t[5]&&y<=t[5]+t[7]){const a=t[6]*t[7];if(a<ba){ba=a;best=t;}}}return best;}
function hover(cv,ev){const g=geom(cv),r=cv.getBoundingClientRect(),x=(ev.clientX-r.left-g.ox)/g.s,y=(ev.clientY-r.top-g.oy)/g.s,
 i=bis(D.xe,x),j=bis(D.ye,y),l=L[ST.sel];
 if(i<0||j<0||i>=NX||j>=NY){ST.hover=null;render();$('readout').textContent='outside the package footprint';return;}
 ST.hover=[x,y];render();const n=j*NX+i,T=TA+l.dT[n],t=tileAt(x,y),code=l.n[n];
 const lines=[`x = ${x.toFixed(4)} mm   y = ${y.toFixed(4)} mm   cell (ix ${i}, iy ${j})  ${fmt((D.xe[i+1]-D.xe[i])*1e3,1)} × ${fmt((D.ye[j+1]-D.ye[j])*1e3,1)} µm`,
  `T = ${T.toFixed(3)} K  (ΔT over ambient ${(T-TA).toFixed(3)} K)    k_eff,z = ${sig(l.k[n],4)} W/m·K    region: ${REG[code]||code}`,
  `power density = ${l.q?sig(l.q[n],4)+' W/cm²':'0 (no power in this layer)'}`,
  t?`tile ${t[0]}  [${t[1]}, chiplet ${t[11]<0?'–':t[11]}, tier ${t[2]}]  P = ${sig(t[10],4)} W`+(t[8]!=null?`  T_mean = ${t[8].toFixed(3)} K  T_max = ${t[9].toFixed(3)} K`:''):'tile –'];
 $('readout').textContent=lines.join('\n');}
function drawPeaks(){const rows=D.peaks.map((p,i)=>`<tr class="pk" data-i="${i}"><td>${p.cls.join(', ')}</td><td class="n">${p.T.toFixed(2)}</td><td>${p.layer}</td><td>${p.tile||'–'}</td></tr>`).join('');
 $('peaks').innerHTML=D.peaks.length?`<table><tr><th>class</th><th class="n">T [K]</th><th>layer</th><th>tile</th></tr>${rows}</table><div class="note">Click a row to jump to its layer.</div>`:'<div class="note">No peaks recorded.</div>';
 $('peaks').querySelectorAll('tr.pk').forEach(r=>r.onclick=()=>{const p=D.peaks[+r.dataset.i],k=L.findIndex(l=>l.name===p.layer);if(k>=0)select(k);});}
function header(){$('h1').textContent=D.title;$('sub').textContent=D.subtitle;const bc=D.bc||{},e=D.energy||{};
 const kv=[['package',W.toFixed(2)+' × '+H.toFixed(2)+' mm'],['grid (x×y×z)',D.nx+' × '+D.ny+' × '+L.reduce((s,l)=>s+l.k1-l.k0,0)],
  ['P',D.P_total!=null?fmt(D.P_total,2)+' W':'–'],['h_top',bc.h_top!=null?bc.h_top+' W/m²K':'–'],['h_bottom',bc.h_bottom!=null?bc.h_bottom+' W/m²K':'–'],
  ['T_amb',fmt(TA,2)+' K']];
 if(e.Q_top_W!=null)kv.push(['Q top/bottom',fmt(e.Q_top_W,2)+' / '+fmt(e.Q_bottom_W,3)+' W'],['balance err',(e.rel_err!=null?e.rel_err.toExponential(1):'–')]);
 $('kv').innerHTML=kv.map(([k,v])=>`${k} <b>${v}</b>`).join(' &nbsp;·&nbsp; ');
 $('dsnote').textContent=(D.down[0]>1||D.down[1]>1)?`Planes block-averaged ${D.down[1]}×${D.down[0]} (x×y) from ${D.nx}×${D.ny} cells for embedding (area-weighted; region code = top-left cell). Peak values and layer T ranges are from the full-resolution field.`:
  `Full-resolution planes (${D.nx}×${D.ny} cells) on the true non-uniform grid. Values stored to 4–5 significant digits.`;}
for(const id of ['cT','cQ']){const cv=$(id);let drag=null;
 cv.addEventListener('mousemove',ev=>{if(drag){const g=geom(cv);ST.view.x0=drag.x0-(ev.clientX-drag.cx)/g.s;ST.view.y0=drag.y0-(ev.clientY-drag.cy)/g.s;render();}else hover(cv,ev);});
 cv.addEventListener('mousedown',ev=>{drag={cx:ev.clientX,cy:ev.clientY,x0:ST.view.x0,y0:ST.view.y0};});
 window.addEventListener('mouseup',()=>drag=null);
 cv.addEventListener('mouseleave',()=>{ST.hover=null;render();});
 cv.addEventListener('wheel',ev=>{ev.preventDefault();const g=geom(cv),r=cv.getBoundingClientRect(),mx=(ev.clientX-r.left-g.ox)/g.s,my=(ev.clientY-r.top-g.oy)/g.s,
  f=Math.exp(-ev.deltaY*0.0015),v=ST.view,ns=Math.min(2000,Math.max(1,v.s*f)),k=v.s/ns;v.x0=mx+(v.x0-mx)*k;v.y0=my+(v.y0-my)*k;v.s=ns;if(ns===1){v.x0=W/2;v.y0=H/2;}render();},{passive:false});}
$('mode').querySelectorAll('button').forEach(b=>b.onclick=()=>{ST.mode=b.dataset.m;$('mode').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));select(ST.sel);});
for(const id of ['tmin','tmax','qmin','qmax'])$(id).addEventListener('input',()=>{if(ST.mode!=='manual'){ST.mode='manual';$('mode').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.m==='manual'));}render();});
for(const id of ['oTiles','oSmall','oPeaks','oGrid'])$(id).onchange=render;
$('oLog').onchange=()=>{ST.logUser=true;select(ST.sel);};
$('reset').onclick=()=>{ST.view={x0:W/2,y0:H/2,s:1};render();};
window.addEventListener('resize',layout);
ST.view={x0:W/2,y0:H/2,s:1};header();
$('oLog').checked=L.some(l=>l.q&&l.qmed>0&&l.qmax/l.qmed>30);
{const h=new URLSearchParams(location.hash.slice(1)),k=L.findIndex(l=>l.name===h.get('layer'));if(k>=0)ST.sel=k;
 if(h.get('mode')){ST.mode=h.get('mode');$('mode').querySelectorAll('button').forEach(x=>x.classList.toggle('on',x.dataset.m===ST.mode));}
 if(h.get('zoom')){const z=h.get('zoom').split(',').map(Number);ST.view={x0:z[0],y0:z[1],s:z[2]||1};}
 select(ST.sel);if(h.get('hover')){const p=h.get('hover').split(',').map(Number),cv=$('cT'),g=geom(cv),r=cv.getBoundingClientRect();
  hover(cv,{clientX:r.left+g.ox+p[0]*g.s,clientY:r.top+g.oy+p[1]*g.s});}}
</script></body></html>
"""


if __name__ == "__main__":
    _main()
