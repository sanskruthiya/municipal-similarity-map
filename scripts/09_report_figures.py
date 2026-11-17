"""検証レポート（docs/report.md）用の図と集計値を作る。→ docs/figures/*.png, docs/report_stats.json"""
import json, os
from collections import Counter
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
FONT_DIR = os.environ.get("CJK_FONT_DIR", os.path.expanduser("~/work/fonts"))
for f in os.listdir(FONT_DIR):
    if f.endswith(".otf"):
        fm.fontManager.addfont(os.path.join(FONT_DIR, f))
plt.rcParams["font.family"] = "Noto Sans CJK JP"
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

att = json.load(open(os.path.join(ROOT, "web", "data", "attributes.json")))
sens = json.load(open(os.path.join(ROOT, "web", "data", "sensitivity.json")))
tiles = json.load(open(os.path.join(ROOT, "web", "data", "tiles.json")))
IND = att["meta"]["indicators"]
GROUPS = list(dict.fromkeys(d["group"] for d in IND))
gn = Counter(d["group"] for d in IND)
W = np.array([1 / gn[d["group"]] for d in IND])
by = {a["code"]: a for a in att["areas"]}
inc = [a for a in att["areas"] if not a.get("excluded")]
Z = np.array([a["z"] for a in inc])
codes = [a["code"] for a in inc]
label = lambda a: a["name"] if a["name"].startswith(a["pref"]) else f'{a["pref"]} {a["name"]}'
stats = {}

# 1) タイル地図（静的）
fig, ax = plt.subplots(figsize=(9, 11), dpi=130)
polys = [f["geometry"]["coordinates"][0] for f in tiles["features"]]
cols = ["#cccccc" if by[f["properties"]["code"]].get("excluded") else by[f["properties"]["code"]]["colors"]["umap_oklab"]
        for f in tiles["features"]]
ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors="white", linewidths=0.3))
ax.add_collection(LineCollection(tiles["pref_lines"]["geometry"]["coordinates"], colors="#333", linewidths=0.7))
ax.autoscale(); ax.set_aspect(1.25); ax.axis("off")
ax.set_title("等面積タイル地図（1市区町村＝1六角形、UMAP → OKLab）", fontsize=13)
plt.savefig(os.path.join(FIG, "tile_umap_oklab.png"), bbox_inches="tight"); plt.close()

# 2) 色変換の検証
m = att["meta"]["metrics"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=130)
for ax, (key, title) in zip(axes, [("knn_overlap_k10", "近傍10件の一致率（元の49次元 vs 色空間）"),
                                   ("dist_spearman", "全ペア距離の順位相関（元の49次元 vs 色空間）")]):
    x = np.arange(2)
    for k, (suf, lab, c) in enumerate([("embedding", "3次元埋め込み", "#999"), ("oklab_color", "OKLab 色", "#2a7"),
                                         ("rgb_color(perceptual)", "RGB 直接の色", "#c55")]):
        v = [m[e][f"{key}_{suf}"] for e in ("pca", "umap")]
        ax.bar(x + (k - 1) * 0.26, v, 0.26, label=lab, color=c)
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 1) * 0.26, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["PCA", "UMAP"]); ax.set_title(title, fontsize=10); ax.set_ylim(0, 1 if key == "dist_spearman" else 0.45)
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "color_validation.png")); plt.close()

# 3) 感度分析
nv = sens["summary"]["neighbor_variants"]
st = np.array([v for v in sens["stability"].values() if v is not None])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130, gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
y = np.arange(len(nv))[::-1]
ax.barh(y, [r["mean_overlap"] for r in nv], color="#4e79a7")
for yi, r in zip(y, nv):
    ax.text(r["mean_overlap"] + 0.01, yi, f'{r["mean_overlap"]:.0%}', va="center", fontsize=8)
ax.set_yticks(y, [r["variant"] for r in nv], fontsize=8); ax.set_xlim(0, 1.1)
ax.set_title("条件を変えたときの類似Top10の平均一致率", fontsize=10)
ax = axes[1]
ax.hist(st, bins=np.linspace(0, 1, 21), color="#59a14f")
ax.axvline(np.median(st), color="k", ls="--", lw=1); ax.text(np.median(st), ax.get_ylim()[1] * 0.92, f" 中央値 {np.median(st):.2f}", fontsize=8)
ax.set_title("市区町村ごとの安定度（指標80%抽出×30回）", fontsize=10); ax.set_xlabel("Top10が平均何割残るか")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "sensitivity.png")); plt.close()
stats["stability"] = {"median": float(np.median(st)), "share_lt_0.6": float((st < 0.6).mean()), "share_ge_0.8": float((st >= 0.8).mean())}

# 4) 類似Top10ペアの距離²の分野別内訳
shares, ind_sh = [], np.zeros(len(IND))
for a in inc:
    za = np.array(a["z"])
    for n in a["neighbors"]:
        c = W * (za - np.array(by[n["code"]]["z"])) ** 2
        shares.append([c[[i for i, d in enumerate(IND) if d["group"] == g]].sum() / c.sum() for g in GROUPS])
        ind_sh += c / c.sum()
share = np.mean(shares, 0)
ind_sh /= len(shares)
stats["top10_domain_share"] = dict(zip(GROUPS, map(float, share)))
stats["top10_indicator_share_top8"] = [(IND[i]["label"], float(ind_sh[i])) for i in np.argsort(-ind_sh)[:8]]
fig, ax = plt.subplots(figsize=(8, 3), dpi=130)
ax.barh(GROUPS[::-1], share[::-1], color="#f28e2b")
ax.axvline(1 / len(GROUPS), color="k", ls=":", lw=1)
for i, v in enumerate(share[::-1]):
    ax.text(v + 0.003, i, f"{v:.0%}", va="center", fontsize=8)
ax.set_title("類似Top10の組で、残っている差（距離²）がどの分野に由来するか（点線＝均等）", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "top10_domain_share.png")); plt.close()

# 5) 地理的な近さ
geo = gpd.read_file(os.path.join(ROOT, "data", "geo", "muni_simplified.geojson")).to_crs(6933)
cen = {c: (p.x, p.y) for c, p in zip(geo.code, geo.geometry.representative_point())}
XY = np.array([cen[c] for c in codes]); idx = {c: i for i, c in enumerate(codes)}
nd = np.array([np.linalg.norm(XY[idx[a["code"]]] - XY[idx[n["code"]]]) / 1000 for a in inc for n in a["neighbors"]])
rng = np.random.default_rng(0)
rp = rng.integers(0, len(codes), (20000, 2)); rp = rp[rp[:, 0] != rp[:, 1]]
rd = np.linalg.norm(XY[rp[:, 0]] - XY[rp[:, 1]], axis=1) / 1000
same = np.mean([np.mean([by[n["code"]]["pref"] == a["pref"] for n in a["neighbors"]]) for a in inc])
pc = Counter(a["pref"] for a in inc); N = len(inc)
stats["geography"] = {"top10_same_pref": float(same), "random_same_pref": sum(c * (c - 1) for c in pc.values()) / (N * (N - 1)),
                      "top10_dist_median_km": float(np.median(nd)), "random_dist_median_km": float(np.median(rd)),
                      "top10_within_100km": float((nd <= 100).mean()), "random_within_100km": float((rd <= 100).mean())}
fig, ax = plt.subplots(figsize=(7, 3.2), dpi=130)
bins = np.linspace(0, 2000, 41)
ax.hist(rd, bins, density=True, alpha=0.5, label="ランダムな組", color="#999")
ax.hist(nd, bins, density=True, alpha=0.6, label="類似Top10の組", color="#e15759")
ax.set_xlabel("代表点間の距離 (km)"); ax.legend(fontsize=8)
ax.set_title("統計的に似た組は地理的にも近い傾向", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "geo_distance.png")); plt.close()

# 6) k-means クラスタの特徴
cl = np.array([a["cluster"] for a in inc])
Xw = Z * np.sqrt(W)
cl_rows = []
for k in range(cl.max() + 1):
    mk = cl == k
    med = np.median(Z[mk], 0)
    order = np.argsort(-np.abs(med))[:4]
    cen_k = Xw[mk].mean(0)
    members = np.where(mk)[0][np.argsort(((Xw[mk] - cen_k) ** 2).sum(1))[:4]]
    cl_rows.append({"cluster": k, "n": int(mk.sum()),
                    "features": [(IND[i]["label"], float(med[i])) for i in order],
                    "typical": [label(inc[i]) for i in members]})
stats["clusters"] = cl_rows
key_ind = ["pop_density", "elderly_ratio", "pop_change", "day_night", "did_ratio", "rail_commute", "primary_ind", "secondary_ind",
           "emp_hospitality", "income", "univ_grad", "owner_ratio", "highrise_ratio", "beds_pc", "station_dist", "agri_ratio", "forest_ratio"]
ki = [next(i for i, d in enumerate(IND) if d["key"] == k) for k in key_ind]
H = np.array([np.median(Z[cl == k][:, ki], 0) for k in range(cl.max() + 1)])
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=130)
im = ax.imshow(H, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
ax.set_xticks(range(len(ki)), [IND[i]["label"] for i in ki], rotation=60, ha="right", fontsize=7)
ax.set_yticks(range(len(H)), [f'C{r["cluster"]}（{r["n"]}）' for r in cl_rows], fontsize=8)
plt.colorbar(im, label="中央値の標準化偏差")
ax.set_title("k-means 12クラスタの指標プロファイル（クラスタ内中央値）", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "cluster_profiles.png")); plt.close()

# 7) 典型例の類似先
EX = ["13113", "20321", "47382", "01209", "23211", "12227", "37364", "27127", "40100", "13363"]
stats["examples"] = []
for c in EX:
    a = by.get(c)
    if a and not a.get("excluded"):
        stats["examples"].append({"area": label(a), "stability": sens["stability"].get(c),
                                  "top5": [label(by[n["code"]]) for n in a["neighbors"][:5]]})
json.dump(stats, open(os.path.join(ROOT, "docs", "report_stats.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False, indent=1))
