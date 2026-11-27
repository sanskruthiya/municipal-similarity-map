"""駅圏の指標を作る。

- 駅: 国土数値情報 S12-23。同じ駅グループコード（S12_001g）の駅は 1 駅にまとめ、乗降客数（2022年）は事業者ごとの値を合計する。
- 駅圏: 0〜500m（駅前）、500m〜1km（徒歩圏）、1〜2km（周辺の住宅地）の 3 つの輪。
  0〜500m だけは最寄り駅に割り当てて重ならないようにし、外側の輪は重なりを許す。
- 人口: 令和2年国勢調査 250mメッシュ（メッシュ中心が輪に入れば、その輪に数える）。
- 土地利用: 国土数値情報 土地利用細分メッシュ L03-b-21（100m）。河川湖沼・海水域を除いた陸域が分母。人口密度の分母にも使う。
- 施設: 国土数値情報 医療機関 P04-20（病院は 2km 以内、診療所は 1km 以内の件数）。
"""
import glob, io, os, zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "processed", "station_features.csv")
R_EARTH = 6371.0
EDGES = (0.5, 1.0, 2.0)
POP_COLS = {"T001142001": "pop", "T001142004": "a0_14", "T001142019": "a65", "T001142031": "foreign",
            "T001142035": "hh", "T001142036": "hh1", "T001142048": "hh1_20s", "T001142049": "hh1_old"}
LU = {100: "agri", 200: "agri", 500: "forest", 600: "forest", 700: "building", 901: "road", 902: "road",
      1000: "other", 1400: "other", 1600: "other"}


def load_stations():
    s = gpd.read_file(f"zip://{RAW}/ksj/S12-23_GML.zip!UTF-8/S12-23_NumberOfPassengers.shp", encoding="utf-8")
    s["pax"] = np.where(s["S12_051"] == 1, s["S12_053"].fillna(0), np.nan)
    p = s.to_crs(6933)
    s["geometry"] = p.geometry.centroid
    s = s.set_geometry("geometry").set_crs(6933, allow_override=True)
    g = s.groupby("S12_001g")
    st = pd.DataFrame({
        "name": g["S12_001"].agg(lambda x: x.value_counts().index[0]),
        "operators": g["S12_002"].agg(lambda x: "・".join(dict.fromkeys(x))),
        "lines": g["S12_003"].agg(lambda x: "・".join(dict.fromkeys(x))),
        "n_lines": g["S12_003"].nunique(),
        "shinkansen": g["S12_005"].agg(lambda x: int((x == 1).any())),
        "pax": g["pax"].agg(lambda x: x.sum() if x.notna().any() else np.nan),
        "x": g.geometry.apply(lambda x: x.x.mean()), "y": g.geometry.apply(lambda x: x.y.mean()),
    })
    pt = gpd.GeoSeries(gpd.points_from_xy(st.x, st.y), crs=6933).to_crs(4326)
    st["lat"], st["lon"] = pt.y.values, pt.x.values
    st.index.name = "id"
    return st.drop(columns=["x", "y"])


def rad(lat, lon):
    return np.radians(np.column_stack([lat, lon]))


class Rings:
    """点を駅の輪に割り当てる。戻り値は (点 index, 駅 index, 輪番号) の配列。"""

    def __init__(self, st):
        self.tree = BallTree(rad(st.lat, st.lon), metric="haversine")

    def assign(self, lat, lon):
        X = rad(lat, lon)
        d1, i1 = self.tree.query(X, k=1)
        near = d1[:, 0] * R_EARTH <= EDGES[-1]
        X, idx, nearest = X[near], np.where(near)[0], i1[near, 0]
        if not len(X):
            e = np.array([], int)
            return e, e, e
        ind, dist = self.tree.query_radius(X, r=EDGES[-1] / R_EARTH, return_distance=True)
        n = np.array([len(a) for a in ind])
        P = np.repeat(np.arange(len(X)), n)
        S = np.concatenate(ind) if len(ind) else np.array([], int)
        D = np.concatenate(dist) * R_EARTH if len(ind) else np.array([])
        ring = np.digitize(D, EDGES)
        keep = (ring > 0) | (S == nearest[P])
        return idx[P[keep]], S[keep], ring[keep]


def digits(code, n):
    """整数のメッシュコードを、上の桁から n 桁ぶんの数字の配列に分ける。"""
    code = np.asarray(code, dtype=np.int64)
    return [(code // 10 ** (n - 1 - k)) % 10 for k in range(n)]


def third_mesh_sw(d):
    """3次メッシュ（先頭 8 桁）の南西端の緯度経度と大きさ。"""
    lat = (d[0] * 10 + d[1]) / 1.5 + d[4] * (2 / 3 / 8) + d[6] * (2 / 3 / 80)
    lon = (d[2] * 10 + d[3]) + 100.0 + d[5] / 8 + d[7] / 80
    return lat, lon, 2 / 3 / 80, 1 / 80


def pop_mesh(rings, nst):
    agg = np.zeros((nst, 3, len(POP_COLS)))
    for z in sorted(glob.glob(os.path.join(RAW, "mesh", "pop250_*.zip"))):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.endswith(".txt")][0]
            d = pd.read_csv(io.BytesIO(zf.read(name)), encoding="cp932", skiprows=[1], dtype=str,
                            usecols=["KEY_CODE", *POP_COLS])
        v = d[list(POP_COLS)].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
        # 250m メッシュ = 3次メッシュ + 1/2 地域メッシュ番号 + 1/4 地域メッシュ番号
        dg = digits(d["KEY_CODE"].astype(np.int64), 10)
        lat, lon, dlat, dlon = third_mesh_sw(dg)
        q1, q2 = dg[8] - 1, dg[9] - 1
        lat = lat + (q1 // 2) * dlat / 2 + (q2 // 2) * dlat / 4 + dlat / 8
        lon = lon + (q1 % 2) * dlon / 2 + (q2 % 2) * dlon / 4 + dlon / 8
        P, S, R = rings.assign(lat, lon)
        np.add.at(agg, (S, R), v[P])
    return agg


def landuse(rings, nst):
    names = sorted(set(LU.values()))
    agg = np.zeros((nst, 3, len(names) + 1))  # 最後の列は陸域セル数
    lut = np.full(1700, -1)
    for k, v in LU.items():
        lut[k] = names.index(v)
    for f in sorted(glob.glob(os.path.join(RAW, "ksj", "L03b", "L03b_*.parquet"))):
        d = pd.read_parquet(f)
        d = d[~d.cls.isin([1100, 1500, 0])]
        dg = digits(d.mesh.to_numpy(), 10)  # 100m メッシュ = 3次メッシュ + 緯度方向 1 桁 + 経度方向 1 桁
        lat, lon, dlat, dlon = third_mesh_sw(dg)
        lat = lat + (dg[8] + 0.5) * dlat / 10
        lon = lon + (dg[9] + 0.5) * dlon / 10
        P, S, R = rings.assign(lat, lon)
        c = lut[d.cls.to_numpy()[P]]
        ok = c >= 0
        np.add.at(agg, (S[ok], R[ok], c[ok]), 1)
        np.add.at(agg, (S, R, len(names)), 1)
    return agg, names


def facilities(st):
    frames = []
    for shp in sorted(glob.glob(os.path.join(RAW, "ksj", "P04", "P04-20_*.shp"))):
        g = gpd.read_file(shp, columns=["P04_001"])
        frames.append(pd.DataFrame({"kind": g["P04_001"].astype(str), "lat": g.geometry.y, "lon": g.geometry.x}))
    f = pd.concat(frames)
    out = {}
    X = rad(st.lat, st.lon)
    for kind, key, r in (("1", "hospitals_2km", 2.0), ("2", "clinics_1km", 1.0)):
        t = f[f.kind == kind]
        out[key] = BallTree(rad(t.lat, t.lon), metric="haversine").query_radius(X, r / R_EARTH, count_only=True)
    return pd.DataFrame(out, index=st.index)


if __name__ == "__main__":
    st = load_stations()
    print("station groups", len(st), "with pax", st.pax.notna().sum(), flush=True)
    muni = gpd.read_file(os.path.join(HERE, "..", "data", "geo", "muni_simplified.geojson"))
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326)
    j = gpd.sjoin_nearest(pts.to_crs(6933), muni.to_crs(6933)[["code", "pref", "name", "geometry"]], how="left")
    j = j[~j.index.duplicated()]
    st["muni_code"], st["pref"], st["muni"] = j["code"], j["pref"], j["name_right"]

    rings = Rings(st)
    tree = rings.tree
    d, _ = tree.query(rad(st.lat, st.lon), k=2)
    st["nearest_station_km"] = d[:, 1] * R_EARTH
    st["stations_2km"] = tree.query_radius(rad(st.lat, st.lon), 2.0 / R_EARTH, count_only=True) - 1

    P = pop_mesh(rings, len(st)); print("pop done", flush=True)
    L, lu_names = landuse(rings, len(st)); print("landuse done", flush=True)
    F = facilities(st)

    out = st.join(F)
    for r, tag in enumerate(("r0", "r1", "r2")):
        p = pd.DataFrame(P[:, r], columns=list(POP_COLS.values()), index=st.index)
        land_km2 = L[:, r, -1] * 0.01
        out[f"{tag}_pop"] = p["pop"]
        out[f"{tag}_land_km2"] = land_km2
        out[f"{tag}_density"] = p["pop"] / np.where(land_km2 > 0, land_km2, np.nan)
        enough = p["pop"] >= 100  # 人口が少ない輪では秘匿処理の影響で割合が不安定になるので欠損扱い
        for k, num, den in (("child", "a0_14", "pop"), ("old", "a65", "pop"), ("foreign", "foreign", "pop"),
                            ("single", "hh1", "hh"), ("young_single", "hh1_20s", "hh"), ("old_single", "hh1_old", "hh")):
            out[f"{tag}_{k}"] = (p[num] / p[den].replace(0, np.nan)).clip(upper=1).where(enough)
        for i, n in enumerate(lu_names):
            out[f"{tag}_{n}"] = L[:, r, i] / np.where(L[:, r, -1] > 0, L[:, r, -1], np.nan)
    out.to_csv(OUT)
    print(out.describe().T.to_string())
