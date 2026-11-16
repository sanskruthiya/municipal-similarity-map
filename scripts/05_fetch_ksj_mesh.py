"""Phase 2 用データの取得。
- 令和2年国勢調査 250mメッシュ人口（e-Stat 統計GIS, T001142。ファイル名 tblT001142Q*.txt, KEY_CODE 10桁）
- 国土数値情報: 駅別乗降客数 S12-23（駅位置）、医療機関 P04-20、土地利用3次メッシュ L03-a-21（2021, 1km）
"""
import os, sys, urllib.request, concurrent.futures as cf
import geopandas as gpd
from shapely.geometry import box

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "data", "raw")
KSJ = "https://nlftp.mlit.go.jp/ksj/gml/data"


def first_meshes():
    g = gpd.read_file(os.path.join(HERE, "..", "data", "geo", "muni_simplified.geojson"))
    codes = []
    for p in range(30, 69):
        for q in range(22, 54):
            lat0, lon0 = p / 1.5, q + 100
            if len(g.sindex.query(box(lon0, lat0, lon0 + 1, lat0 + 2 / 3), predicate="intersects")):
                codes.append(f"{p}{q}")
    return codes


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path, "cached"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = r.read()
        if not data.startswith(b"PK"):
            return path, "not-zip"
        with open(path, "wb") as f:
            f.write(data)
        return path, "ok"
    except Exception as e:
        return path, f"err {e}"


if __name__ == "__main__":
    meshes = first_meshes()
    print(len(meshes), "first meshes")
    jobs = [(f"{KSJ}/S12/S12-23/S12-23_GML.zip", os.path.join(RAW, "ksj", "S12-23_GML.zip"))]
    jobs += [(f"{KSJ}/P04/P04-20/P04-20_{i:02d}_GML.zip", os.path.join(RAW, "ksj", f"P04-20_{i:02d}_GML.zip")) for i in range(1, 48)]
    for m in meshes:
        jobs.append((f"https://www.e-stat.go.jp/gis/statmap-search/data?statsId=T001142&code={m}&downloadType=2",
                     os.path.join(RAW, "mesh", f"pop250_{m}.zip")))
        jobs.append((f"{KSJ}/L03-a/L03-a-21/L03-a-21_{m}-jgd2011_GML.zip",
                     os.path.join(RAW, "ksj", "L03", f"L03-a-21_{m}.zip")))
    os.makedirs(os.path.join(RAW, "ksj", "L03"), exist_ok=True)
    stat = {}
    with cf.ThreadPoolExecutor(6) as ex:
        for path, s in ex.map(lambda j: get(*j), jobs):
            stat[s.split()[0]] = stat.get(s.split()[0], 0) + 1
            if s.startswith("err"):
                print(os.path.basename(path), s)
    print(stat)
