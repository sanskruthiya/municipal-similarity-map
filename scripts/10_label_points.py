"""地図の市区町村名ラベル・マーカー位置（最大面積のポリゴンの到達不能極 = 内部で境界から最も遠い点）を作る。

出力: web/data/labels.json  {code: [lon, lat]}
"""
import json, os
import geopandas as gpd
from shapely.ops import polylabel

HERE = os.path.dirname(__file__)
g = gpd.read_file(os.path.join(HERE, "..", "data", "geo", "muni_simplified.geojson")).to_crs(6933)
pts = {}
for code, geom in zip(g["code"], g.geometry):
    part = max(getattr(geom, "geoms", [geom]), key=lambda p: p.area)
    pts[code] = polylabel(part, tolerance=50)
ll = gpd.GeoSeries(list(pts.values()), crs=6933).to_crs(4326)
out = {c: [round(p.x, 4), round(p.y, 4)] for c, p in zip(pts, ll)}
with open(os.path.join(HERE, "..", "web", "data", "labels.json"), "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(len(out), "label points")
