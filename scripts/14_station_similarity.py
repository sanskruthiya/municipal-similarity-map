"""駅圏の類似度: 指標の変換 → ロバスト標準化 → KNN補完 → 分野重み（市区町村版と同じ build_space）→ 重み付きユークリッド距離の Top10。
既定の Top10 は 2km 以内の駅を除く（隣の駅どうしは駅圏が重なり、似ていて当然なため）。除かない Top10 も残す。
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree, NearestNeighbors
from pipeline import build_space

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed")
K, EXCL_KM, R_EARTH = 10, 2.0, 6371.0

RINGS = {"r0": "駅前0〜500m", "r1": "徒歩圏500m〜1km", "r2": "周辺1〜2km"}
PER_RING = [  # key, label, group, transform
    ("density", "人口密度(人/km²)", "人口密度", "log1p"),
    ("child", "15歳未満の割合", "年齢", "none"),
    ("old", "65歳以上の割合", "年齢", "none"),
    ("foreign", "外国人の割合", "外国人", "sqrt"),
    ("single", "単身世帯の割合", "世帯", "none"),
    ("young_single", "20代の単身世帯の割合", "世帯", "sqrt"),
    ("old_single", "高齢単身世帯の割合", "世帯", "none"),
    ("building", "建物用地の割合", "土地利用", "none"),
    ("agri", "農地の割合", "土地利用", "none"),
    ("forest", "森林・荒地の割合", "土地利用", "none"),
    ("road", "道路・鉄道の割合", "土地利用", "none"),
    ("other", "公園・学校などその他の用地の割合", "土地利用", "none"),
]
STATION = [
    ("pax", "乗降客数(人/日, 2022)", "駅・交通", "log1p"),
    ("n_lines", "乗り入れ路線数", "駅・交通", "log"),
    ("nearest_station_km", "隣の駅までの距離(km)", "駅・交通", "log"),
    ("stations_2km", "2km以内の駅の数", "駅・交通", "log1p"),
    ("hospitals_2km", "2km以内の病院の数", "医療", "log1p"),
    ("clinics_1km", "1km以内の診療所の数", "医療", "log1p"),
]


def meta():
    m = [dict(key=k, label=l, group=g, transform=t) for k, l, g, t in STATION]
    for r, rl in RINGS.items():
        m += [dict(key=f"{r}_{k}", label=f"{rl}: {l}", group=g, transform=t, ring=r) for k, l, g, t in PER_RING]
    return m


if __name__ == "__main__":
    st = pd.read_csv(os.path.join(PROC, "station_features.csv"), dtype={"id": str, "muni_code": str}).set_index("id")
    M = meta()
    keys = [m["key"] for m in M]
    Xw, Z, mask = build_space(st[keys], M)
    print("features", len(keys), "stations", len(Xw))
    print("missing share", round(float(mask.to_numpy().mean()), 4))

    X = Xw.to_numpy()
    nn = NearestNeighbors(n_neighbors=200).fit(X)
    dist, ind = nn.kneighbors(X)
    geo = BallTree(np.radians(st[["lat", "lon"]].to_numpy()), metric="haversine")
    close = geo.query_radius(np.radians(st[["lat", "lon"]].to_numpy()), EXCL_KM / R_EARTH)
    ids = Xw.index.to_numpy()
    rows = []
    for i in range(len(X)):
        near = set(close[i])
        far = [(j, d) for j, d in zip(ind[i], dist[i]) if j != i and j not in near][:K]
        alln = [(j, d) for j, d in zip(ind[i], dist[i]) if j != i][:K]
        for variant, lst in (("far", far), ("all", alln)):
            for rank, (j, d) in enumerate(lst, 1):
                rows.append((ids[i], variant, rank, ids[j], round(float(d), 4)))
    nb = pd.DataFrame(rows, columns=["id", "variant", "rank", "nb", "dist"])
    nb.to_csv(os.path.join(PROC, "station_neighbors.csv"), index=False)
    Z.round(3).to_csv(os.path.join(PROC, "station_zscores.csv"))
    json.dump(M, open(os.path.join(PROC, "station_meta.json"), "w"), ensure_ascii=False, indent=1)

    # 近い駅を除いたときに Top10 がどれだけ変わるか
    far = nb[nb.variant == "far"].groupby("id").nb.apply(set)
    alln = nb[nb.variant == "all"].groupby("id").nb.apply(set)
    ov = np.mean([len(far[i] & alln[i]) / K for i in far.index])
    within = np.mean([len(alln[i] & set(ids[list(close[ids.tolist().index(i)])])) for i in alln.index[:2000]])
    print("Top10 overlap far vs all:", round(ov, 3), "| mean # of <2km stations in unrestricted Top10 (first 2000):", round(within, 2))
