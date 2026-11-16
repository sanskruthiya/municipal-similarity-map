"""国土数値情報・メッシュ人口から「立地・土地利用」分野の指標を作る。

- 人口加重人口密度: 250mメッシュ人口で重み付けした、住民が実際に住む場所の密度
- 最寄り駅・最寄り病院までの人口加重平均距離: 各メッシュ中心から最寄り地点までの大圏距離を人口で平均
- 農用地率・建物用地率: 土地利用3次メッシュ（1km）の面積を、メッシュ中心が含まれる市区町村に集計（水面・範囲外を除く陸域が分母）
"""
import glob, io, os, zipfile
import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(HERE, "..", "data", "processed", "ksj_features.csv")
EARTH_KM = 6371.0


def mesh_center(code):
    """地域メッシュコード（1次〜5次以降）の中心緯度経度と大きさ（度）。"""
    code = str(code)
    lat = int(code[:2]) / 1.5
    lon = int(code[2:4]) + 100
    dlat, dlon = 2 / 3, 1.0
    if len(code) >= 6:
        dlat, dlon = dlat / 8, dlon / 8
        lat += int(code[4]) * dlat; lon += int(code[5]) * dlon
    if len(code) >= 8:
        dlat, dlon = dlat / 10, dlon / 10
        lat += int(code[6]) * dlat; lon += int(code[7]) * dlon
    for c in code[8:]:
        dlat, dlon = dlat / 2, dlon / 2
        q = int(c) - 1
        lat += (q // 2) * dlat; lon += (q % 2) * dlon
    return lat + dlat / 2, lon + dlon / 2, dlat, dlon


def to_points(df, code_col):
    c = np.array([mesh_center(x) for x in df[code_col]])
    df = df.assign(lat=c[:, 0], lon=c[:, 1], dlat=c[:, 2], dlon=c[:, 3])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)


def assign_muni(pts, muni):
    j = gpd.sjoin(pts, muni[["code", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j["code"].isna()
    if miss.any():  # 海岸付近でポリゴン外に落ちた中心点は最寄りの市区町村へ（2km以内）
        near = gpd.sjoin_nearest(pts[miss].to_crs(6933), muni[["code", "geometry"]].to_crs(6933),
                                 how="left", max_distance=2000)
        near = near[~near.index.duplicated()]
        j.loc[miss, "code"] = near["code"]
    return j.drop(columns=["index_right"], errors="ignore")


def read_mesh_pop():
    frames = []
    for z in sorted(glob.glob(os.path.join(RAW, "mesh", "pop250_*.zip"))):
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.endswith(".txt")][0]
            d = pd.read_csv(io.BytesIO(zf.read(name)), encoding="cp932", skiprows=[1], dtype=str,
                            usecols=["KEY_CODE", "T001142001"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["pop"] = pd.to_numeric(d["T001142001"], errors="coerce").fillna(0)
    return d[d["pop"] > 0][["KEY_CODE", "pop"]]


def nearest_km(pts, targets):
    tree = BallTree(np.radians(targets[["lat", "lon"]].to_numpy()), metric="haversine")
    d, _ = tree.query(np.radians(pts[["lat", "lon"]].to_numpy()), k=1)
    return d[:, 0] * EARTH_KM


def stations():
    s = gpd.read_file(f"zip://{RAW}/ksj/S12-23_GML.zip!UTF-8/S12-23_NumberOfPassengers.shp")
    s = s.to_crs(6933)
    s["geometry"] = s.geometry.centroid
    s = s.dissolve(by="S12_001g").centroid.to_crs(4326)
    return pd.DataFrame({"lat": s.y, "lon": s.x})


def extract_shp(z, d, prefix):
    """zip 内のパス区切りが "\\" の場合があるので、正規化して展開し .shp のパスを返す。"""
    os.makedirs(d, exist_ok=True)
    shp = None
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            base = n.replace("\\", "/").split("/")[-1]
            if base.startswith(prefix) and base.endswith((".shp", ".shx", ".dbf", ".prj", ".cpg")):
                with open(os.path.join(d, base), "wb") as f:
                    f.write(zf.read(n))
                if base.endswith(".shp"):
                    shp = os.path.join(d, base)
    return shp


def hospitals():
    frames = []
    for z in sorted(glob.glob(os.path.join(RAW, "ksj", "P04-20_*_GML.zip"))):
        shp = extract_shp(z, os.path.join(RAW, "ksj", "P04"), "P04")
        g = gpd.read_file(shp)
        g = g[g["P04_001"].astype(str) == "1"]
        frames.append(pd.DataFrame({"lat": g.geometry.y, "lon": g.geometry.x}))
    return pd.concat(frames, ignore_index=True)


def landuse(muni):
    cls = ["田", "他農用地", "森林", "荒地", "建物用地", "道路", "鉄道", "他用地", "河川湖沼", "海浜", "海水域", "ゴルフ場", "範囲外"]
    frames = []
    for z in sorted(glob.glob(os.path.join(RAW, "ksj", "L03", "L03-a-21_*.zip"))):
        shp = extract_shp(z, os.path.join(RAW, "ksj", "L03", "shp"), "L03")
        d = gpd.read_file(shp, ignore_geometry=True)
        d.columns = ["メッシュ"] + cls  # .cpg の無いファイルは列名が文字化けするので列順で付け直す
        frames.append(d)
    lu = pd.concat(frames, ignore_index=True)
    lu["メッシュ"] = lu["メッシュ"].astype(str)
    lu[cls] = lu[cls].apply(pd.to_numeric, errors="coerce").fillna(0)
    lu["land"] = lu[cls].drop(columns=["河川湖沼", "海水域", "範囲外"]).sum(axis=1)
    lu = lu[lu["land"] > 0]
    j = assign_muni(to_points(lu, "メッシュ"), muni)
    agg = j.groupby("code")[["田", "他農用地", "建物用地", "land"]].sum()
    return pd.DataFrame({
        "agri_ratio": (agg["田"] + agg["他農用地"]) / agg["land"],
        "paddy_share": agg["田"] / (agg["田"] + agg["他農用地"]).replace(0, np.nan),
        "building_ratio": agg["建物用地"] / agg["land"],
    })


if __name__ == "__main__":
    muni = gpd.read_file(os.path.join(HERE, "..", "data", "geo", "muni_simplified.geojson"))
    pop = to_points(read_mesh_pop(), "KEY_CODE")
    print("mesh cells", len(pop), "pop", int(pop["pop"].sum()))
    pop = assign_muni(pop, muni)
    print("unassigned pop", int(pop.loc[pop["code"].isna(), "pop"].sum()))
    km_lat = 111.32
    pop["area_km2"] = (pop.dlat * km_lat) * (pop.dlon * km_lat * np.cos(np.radians(pop.lat)))
    pop["dens"] = pop["pop"] / pop["area_km2"]
    pop["d_station"] = nearest_km(pop, stations())
    pop["d_hospital"] = nearest_km(pop, hospitals())
    pop = pop.dropna(subset=["code"])
    w = pop["pop"]
    g = pop.assign(wd=w * pop.dens, ws=w * pop.d_station, wh=w * pop.d_hospital).groupby("code")
    f = pd.DataFrame({
        "pw_density": g.wd.sum() / g["pop"].sum(),
        "station_dist": g.ws.sum() / g["pop"].sum(),
        "hospital_dist": g.wh.sum() / g["pop"].sum(),
    }).join(landuse(muni), how="outer")
    f.index.name = "code"
    f.to_csv(OUT)
    print(f.describe().T)
