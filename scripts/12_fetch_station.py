"""駅圏分析用の追加データ取得: 国土数値情報 土地利用細分メッシュ L03-b-21（2021, 100m）。
駅から 2km 以内にかかる 1次メッシュだけを取得し、dbf（メッシュコードと土地利用種別）だけを parquet に縮めて保存する。
"""
import io, os, urllib.request, zipfile, concurrent.futures as cf
import geopandas as gpd
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw")
OUT = os.path.join(RAW, "ksj", "L03b")
URL = "https://nlftp.mlit.go.jp/ksj/gml/data/L03-b/L03-b-21/L03-b-21_{m}-jgd2011_GML.zip"


def station_first_meshes():
    s = gpd.read_file(f"zip://{RAW}/ksj/S12-23_GML.zip!UTF-8/S12-23_NumberOfPassengers.shp", encoding="utf-8")
    c = s.to_crs(6933).geometry.centroid.to_crs(4326)
    out = set()
    for dy in (-0.02, 0, 0.02):
        for dx in (-0.025, 0, 0.025):
            for la, lo in zip(c.y + dy, c.x + dx):
                out.add(f"{int(la * 1.5)}{int(lo - 100)}")
    return sorted(out)


def fetch(m):
    path = os.path.join(OUT, f"L03b_{m}.parquet")
    if os.path.exists(path):
        return m, "cached"
    try:
        req = urllib.request.Request(URL.format(m=m), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
    except Exception as e:
        return m, f"err {e}"
    if not data.startswith(b"PK"):
        return m, "not-zip"
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".dbf")][0]
        tmp = os.path.join(OUT, f"tmp_{m}.dbf")
        with open(tmp, "wb") as f:
            f.write(zf.read(name))
    d = gpd.read_file(tmp, ignore_geometry=True)
    os.remove(tmp)
    pd.DataFrame({"mesh": d["L03b_001"].astype(np.int64), "cls": d["L03b_002"].astype(int).astype(np.int16)}).to_parquet(path)
    return m, "ok"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ms = station_first_meshes()
    print(len(ms), "first meshes", flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        for m, st in ex.map(fetch, ms):
            print(m, st, flush=True)
