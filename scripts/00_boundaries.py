"""国土数値情報 N03（行政区域, 2024-01-01）から市区町村（政令市は区）単位のポリゴンを作成する。

- N03_007（行政区域コード）でディゾルブ。所属未定地などコードの無いものは除外
- 浜松市は2024年1月に区再編（7区→3区）されたが、統計（2020年基準）は旧区のため市単位(22130)に統合
- 統計側のエリア一覧（data/processed/areas.csv）に存在するコードのみ残す
"""
import os, sys
import geopandas as gpd
import pandas as pd

HERE = os.path.dirname(__file__)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "geo", "src", "N03-20240101.shp")
GEO = os.path.join(HERE, "..", "data", "geo")
HAMAMATSU_NEW_WARDS = {"22138", "22139", "22140"}

g = gpd.read_file(SRC)
g = g[g["N03_007"].notna()].rename(columns={"N03_007": "code"})
g.loc[g["code"].isin(HAMAMATSU_NEW_WARDS), "code"] = "22130"
areas = pd.read_csv(os.path.join(HERE, "..", "data", "processed", "areas.csv"), dtype=str)
g = g[g["code"].isin(set(areas["code"]))]
d = g[["code", "geometry"]].dissolve(by="code").reset_index().merge(areas, on="code")
d.to_file(os.path.join(GEO, "muni_full.shp"), encoding="utf-8")

s = d.to_crs(6933)
s["geometry"] = s.geometry.simplify(50).buffer(0)
s = s.to_crs(4326)
s.to_file(os.path.join(GEO, "muni_simplified.geojson"), driver="GeoJSON")
print(len(d), "municipalities")
