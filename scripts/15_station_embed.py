"""駅圏: UMAP 3次元埋め込み → OKLab 色（市区町村版と同じ色の付け方）→ Web 用 stations.json。"""
import importlib.util, json, os
import numpy as np
import pandas as pd
import umap
from pipeline import build_space

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed")
WEB = os.path.join(HERE, "..", "web", "data")
SEED, K = 42, 10


def load_module(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sim = load_module("station_similarity", "14_station_similarity.py")
URBAN_PROXY = {"r0_density": 1, "r1_density": 1, "r1_building": 1, "r1_forest": -1, "r1_agri": -1}

if __name__ == "__main__":
    st = pd.read_csv(os.path.join(PROC, "station_features.csv"), dtype={"id": str, "muni_code": str}).set_index("id")
    nb = pd.read_csv(os.path.join(PROC, "station_neighbors.csv"), dtype={"id": str, "nb": str})
    M = sim.meta()
    keys = [m["key"] for m in M]
    Xw, Z, mask = build_space(st[keys], M)
    X = Xw.to_numpy()

    E = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.3, random_state=SEED).fit_transform(X)
    # 03_embed.py の色関数を、市区町村データを読まずに使うため関数定義部分だけ実行する
    src = open(os.path.join(HERE, "03_embed.py"), encoding="utf-8").read()
    a, b = src.index("# --- OKLab → sRGB"), src.index("def color_rgb_naive")
    ns = {"np": np, "PCA": __import__("sklearn.decomposition", fromlist=["PCA"]).PCA}
    exec(src[a:b], ns)
    urb = sum(w * Z[k].to_numpy() for k, w in URBAN_PROXY.items())
    lab, hx = ns["color_oklab"](E, urb)

    pos = {c: i for i, c in enumerate(Xw.index)}
    nbs = {v: g.sort_values("rank") for v, g in nb.groupby("variant")}
    lists = {v: g.groupby("id")[["nb", "dist"]].apply(lambda d: [[pos[n], round(float(x), 3)] for n, x in zip(d.nb, d.dist)]).to_dict()
             for v, g in nbs.items()}

    def r(v, nd=4):
        return None if pd.isna(v) else float(f"{v:.{nd}g}")

    recs = []
    for i, c in enumerate(Xw.index):
        s = st.loc[c]
        recs.append({
            "id": c, "name": s["name"], "muni": s["muni"], "pref": s["pref"],
            "ll": [round(float(s.lon), 5), round(float(s.lat), 5)],
            "ops": s["operators"], "lines": s["lines"], "col": hx[i],
            "z": [round(float(v), 1) for v in Z.loc[c, keys]],
            "raw": [r(v, 3) for v in st.loc[c, keys]],
            "imp": [j for j, k in enumerate(keys) if mask.loc[c, k]],
            # far: 2km以内を除いた Top10 [index, 距離]。near: 除外しない Top10 に入る 2km 以内の駅だけ
            # （画面側で far と合わせて距離順に並べ直すと、除外しない Top10 になる）
            "far": lists["far"][c], "near": [x for x in lists["all"][c] if x[0] not in {y[0] for y in lists["far"][c]}],
        })
    groups = list(dict.fromkeys(m["group"] for m in M))
    out = {"meta": {"indicators": M, "groups": groups, "rings": sim.RINGS, "k": K, "excl_km": sim.EXCL_KM,
},
           "st": recs}
    json.dump(out, open(os.path.join(WEB, "stations.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print("stations.json", round(os.path.getsize(os.path.join(WEB, "stations.json")) / 1e6, 2), "MB")
