"""等面積の六角形タイル地図（カルトグラム）のレイアウトを作る。

面積の大きい町村（北海道など）が地図の見た目を支配しないよう、全市区町村を同じ大きさの六角形で表す。
1) 各市区町村の代表点（最大ポリゴン内の点）を Web メルカトルに投影
2) 沖縄県・小笠原村は本土寄りに平行移動（インセット）
3) 同じ半径の円として衝突回避しつつ元の位置へ引き戻す反復（Dorling 型）で密集地を押し広げる
4) 六角形グリッドのセルへ、二乗距離の総和が最小になるよう割り当て（線形割当問題）
→ web/data/tiles.json（六角形 GeoJSON と都道府県境の線）
"""
import json, math, os
import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
R_EARTH = 6378137.0
HEX_R = float(os.environ.get("HEX_R", 14000))   # 六角形の外接円半径（メルカトル m）
ANCHOR = float(os.environ.get("ANCHOR", 0.003))
SPRING = float(os.environ.get("SPRING", 0.05))
INSETS = {"47": (6.5e5, 6.0e5), "13421": (-1.5e5, 9.0e5)}   # 沖縄県, 小笠原村 の移動量 (dx, dy)

g = gpd.read_file(os.path.join(HERE, "..", "data", "geo", "muni_simplified.geojson"))
att = json.load(open(os.path.join(HERE, "..", "web", "data", "attributes.json")))
pref_of = {a["code"]: a["pref"] for a in att["areas"]}
g = g[g.code.isin(pref_of)].reset_index(drop=True)
gm = g.to_crs(3857)
pts = np.array([max(geom.geoms, key=lambda p: p.area).representative_point().coords[0]
                if geom.geom_type == "MultiPolygon" else geom.representative_point().coords[0]
                for geom in gm.geometry])
for key, (dx, dy) in INSETS.items():
    m = g.code.str.startswith(key) if len(key) == 2 else (g.code == key)
    pts[m.values] += (dx, dy)

# 力学的配置: 隣接する市区町村同士をばねで結び（自然長 = セル間隔）、全ペアの重なりを解消し、元の位置へ弱く引き戻す。
# 密集地は押し広げられ、北海道などの疎な地域は縮む。
spacing = math.sqrt(3) * HEX_R
gm["geometry"] = gm.geometry.buffer(300)
edges = set()
for i, j in zip(*gm.sindex.query(gm.geometry, predicate="intersects")):
    if i < j:
        edges.add((i, j))
deg = np.bincount(np.array(list(edges)).ravel(), minlength=len(gm))
_, nn = cKDTree(pts).query(pts, 3)
for i in np.where(deg == 0)[0]:          # 離島など隣接のない地域は最寄り2地域とつなぐ
    for j in nn[i, 1:]:
        edges.add((min(i, j), max(i, j)))
E_ = np.array(sorted(edges))
P0, P = pts.copy(), pts.copy()
for it in range(1500):
    F = (P0 - P) * ANCHOR
    d = P[E_[:, 1]] - P[E_[:, 0]]
    dist = np.linalg.norm(d, axis=1) + 1e-6
    f = (SPRING * (dist - spacing) / dist)[:, None] * d
    np.add.at(F, E_[:, 0], f); np.add.at(F, E_[:, 1], -f)
    pairs = cKDTree(P).query_pairs(spacing, output_type="ndarray")
    if len(pairs):
        d = P[pairs[:, 1]] - P[pairs[:, 0]]
        dist = np.linalg.norm(d, axis=1) + 1e-6
        f = (0.5 * (spacing - dist) / dist)[:, None] * d
        np.add.at(F, pairs[:, 0], -f); np.add.at(F, pairs[:, 1], f)
    P += F
    if it % 300 == 0:
        print("iter", it, "overlaps", len(pairs))

# 六角形グリッド（pointy-top）: 行間隔 1.5R、列間隔 √3R、奇数行は半セルずらす
xmin, ymin = P.min(0) - 3 * spacing
xmax, ymax = P.max(0) + 3 * spacing
rows = np.arange(0, int((ymax - ymin) / (1.5 * HEX_R)) + 1)
cols = np.arange(0, int((xmax - xmin) / spacing) + 1)
cr, cc = np.meshgrid(rows, cols, indexing="ij")
cx = xmin + cc * spacing + (cr % 2) * spacing / 2
cy = ymin + cr * 1.5 * HEX_R
cells = np.c_[cx.ravel(), cy.ravel()]
cell_rc = np.c_[cr.ravel(), cc.ravel()]
near = np.unique(np.concatenate(cKDTree(cells).query_ball_point(P, 4 * spacing)))
cells, cell_rc = cells[near], cell_rc[near]
cost = ((P[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
ri, ci = linear_sum_assignment(cost)
assign = np.empty(len(P), int); assign[ri] = ci
# 局所探索: 隣接セルとの入れ替え・空きセルへの移動で「都道府県が異なる隣接辺の数」を減らす（移動距離の増加とのトレードオフ）
def nbrs_rc(r_, c_):
    odd = r_ % 2
    return [(r_ + 1, c_ + odd), (r_ + 1, c_ + odd - 1), (r_, c_ - 1), (r_ - 1, c_ + odd - 1), (r_ - 1, c_ + odd), (r_, c_ + 1)]
pref_arr = np.array([pref_of[c] for c in g.code])
xy_of = {tuple(rc): xy for rc, xy in zip(cell_rc, cells)}
pos = {i: tuple(cell_rc[assign[i]]) for i in range(len(P))}
at = {rc: i for i, rc in pos.items()}
LAM = float(os.environ.get("PREF_PENALTY", 1.0))
EMPTY = float(os.environ.get("EMPTY_PENALTY", 0.4))
def edge_pen(u, v):
    a_, b_ = at.get(u), at.get(v)
    if a_ is None and b_ is None:
        return 0.0
    if a_ is None or b_ is None:
        return EMPTY * LAM
    return LAM if pref_arr[a_] != pref_arr[b_] else 0.0
def local_E(rcs):
    e = sum(0.05 * ((np.array(xy_of[rc]) - P[at[rc]]) ** 2).sum() / spacing ** 2 for rc in rcs if rc in at)
    seen = set()
    for rc in rcs:
        for n in nbrs_rc(*rc):
            k = (min(rc, n), max(rc, n))
            if k not in seen:
                seen.add(k); e += edge_pen(rc, n)
    return e
def apply(rc, n):
    i, j = at.pop(rc, None), at.pop(n, None)
    if i is not None: at[n] = i; pos[i] = n
    if j is not None: at[rc] = j; pos[j] = rc
for sweep in range(100):
    moved = 0
    for i in range(len(P)):
        rc = pos[i]
        for n in nbrs_rc(*rc):
            if n not in xy_of:
                continue
            j = at.get(n)
            if j is not None and pref_arr[j] == pref_arr[i]:
                continue
            before = local_E((rc, n))
            apply(rc, n)
            if local_E((rc, n)) < before - 1e-9:
                moved += 1
                break
            apply(rc, n)
    if not moved:
        break
print("local search sweeps", sweep + 1)
rc_index = {tuple(rc): k for k, rc in enumerate(cell_rc)}
assign = np.array([rc_index[pos[i]] for i in range(len(P))])
disp = np.linalg.norm(cells[assign] - pts, axis=1) / 1000
print(f"cells {len(cells)}  displacement km: median {np.median(disp):.0f}  p90 {np.percentile(disp, 90):.0f}  max {disp.max():.0f}")

def to_ll(x, y):
    return [round(math.degrees(x / R_EARTH), 5), round(math.degrees(2 * math.atan(math.exp(y / R_EARTH)) - math.pi / 2), 5)]

ANG = [math.radians(30 + 60 * k) for k in range(6)]
def corner(c, k):
    return (c[0] + HEX_R * math.cos(ANG[k]), c[1] + HEX_R * math.sin(ANG[k]))

feats = []
occ = {}
for i, code in enumerate(g.code):
    c = cells[assign[i]]
    ring = [to_ll(*corner(c, k)) for k in range(6)]
    feats.append({"type": "Feature", "properties": {"code": code},
                  "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}})
    occ[tuple(cell_rc[assign[i]])] = code

# 都道府県境: 隣接セルの都道府県が異なる（または空き）辺を線にする
def nbr(rc, k):
    r_, c_ = rc
    odd = r_ % 2
    # 辺 k は角 k と k+1 の間（角0=30°）。その方向の隣接セル
    return {0: (r_ + 1, c_ + odd), 1: (r_ + 1, c_ + odd - 1), 2: (r_, c_ - 1),
            3: (r_ - 1, c_ + odd - 1), 4: (r_ - 1, c_ + odd), 5: (r_, c_ + 1)}[k]
lines = []
for rc, code in occ.items():
    c = cells[np.where((cell_rc == rc).all(1))[0][0]]
    for k in range(6):
        o = occ.get(nbr(rc, k))
        if o is None or pref_of[o] != pref_of[code]:
            if o is not None and o < code:
                continue
            lines.append([to_ll(*corner(c, k)), to_ll(*corner(c, (k + 1) % 6))])
out = {"type": "FeatureCollection", "features": feats,
       "pref_lines": {"type": "Feature", "properties": {},
                      "geometry": {"type": "MultiLineString", "coordinates": lines}}}
json.dump(out, open(os.path.join(HERE, "..", "web", "data", "tiles.json"), "w"), separators=(",", ":"))
print("wrote tiles.json", len(feats), "tiles", len(lines), "border edges")
