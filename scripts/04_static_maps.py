"""色分けマップの静的画像（PNG）を出力する。"""
import json, os, sys
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.environ.get("CJK_FONT_DIR", os.path.expanduser("~/work/fonts"))
for f in os.listdir(FONT_DIR):
    if f.endswith(".otf"):
        fm.fontManager.addfont(os.path.join(FONT_DIR, f))
plt.rcParams["font.family"] = "Noto Sans CJK JP"

g = gpd.read_file(os.path.join(HERE, "..", "data", "geo", "muni_simplified.geojson"))
att = json.load(open(os.path.join(HERE, "..", "web", "data", "attributes.json")))
recs = {r["code"]: r for r in att["areas"]}
OUT = os.path.join(HERE, "..", "output")
os.makedirs(OUT, exist_ok=True)

TITLES = {"umap_oklab": "UMAP 3次元 → OKLab（等方スケーリング）", "pca_oklab": "PCA 3次元 → OKLab（等方スケーリング）",
          "umap_rgb": "UMAP 3次元 → RGB直接（比較用）", "pca_rgb": "PCA 3次元 → RGB直接（比較用）"}
methods = sys.argv[1:] or list(TITLES)

for m in methods:
    g["color"] = [recs.get(c, {}).get("colors", {}).get(m, "#d0d0d0") for c in g.code]
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    g.plot(ax=ax, color=g["color"], edgecolor="#ffffff", linewidth=0.08)
    ax.set_xlim(128.5, 146.2); ax.set_ylim(30.8, 45.7); ax.set_axis_off()
    ax.set_title(f"統計的に似た市区町村ほど似た色: {TITLES.get(m, m)}", fontsize=15)
    ins = fig.add_axes([0.13, 0.62, 0.22, 0.2])
    g.plot(ax=ins, color=g["color"], edgecolor="#ffffff", linewidth=0.08)
    ins.set_xlim(122.8, 131.5); ins.set_ylim(23.9, 28.6); ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("沖縄", fontsize=9)
    # 首都圏・関西の拡大
    for pos, ext, label in [([0.62, 0.08, 0.3, 0.26], (138.9, 140.4, 35.1, 36.2), "首都圏"),
                            ([0.62, 0.36, 0.22, 0.2], (134.9, 136.0, 34.3, 35.1), "関西")]:
        a = fig.add_axes(pos)
        g.plot(ax=a, color=g["color"], edgecolor="#ffffff", linewidth=0.2)
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_xticks([]); a.set_yticks([])
        a.set_title(label, fontsize=9)
    fig.text(0.02, 0.01, "出典: e-Stat 社会・人口統計体系（市区町村データ）、国土数値情報（行政区域データ N03 2024）を加工して作成。灰色は除外自治体。",
             fontsize=8, color="#555")
    p = os.path.join(OUT, f"map_{m}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(p)
