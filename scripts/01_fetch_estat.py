"""e-Stat 社会・人口統計体系（市区町村データ 基礎データ・廃置分合処理済）から必要な項目を取得する。"""
import json, os, sys, time, urllib.parse, urllib.request
import pandas as pd

APPID = os.environ["ESTAT_APPID"]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "ssds_municipal.csv")

ITEMS = {
    "0000020201": ["A1101", "A1231", "A1301", "A1303", "A1419", "A1405", "A141402", "A161001", "A1700", "A1801",
                   "A191001", "A191007", "A4101", "A4200", "A5103", "A5104", "A6108", "A710101", "A710201",
                   "A810102", "A810105", "A8301", "A9101", "A9201"],
    "0000020202": ["B1101", "B1103", "B1105"],
    "0000020203": ["C120110", "C120120", "C2207", "C220702", "C220709", "C220711", "C220713", "C220719", "C220722",
                   "C220716", "C220718"],
    "0000020204": ["D2201", "D2203"],
    "0000020205": ["E9101", "E9106", "E9102"],
    "0000020206": ["F1101", "F110102", "F1102", "F1107", "F2116", "F2201", "F2211", "F2221", "F2401", "F2409",
                   "F2705", "F2803"],
    "0000020208": ["H1100", "H1101", "H110202", "H1310", "H1401", "H1403", "H2130", "H6130", "H6131",
                   "H7301", "H730102", "H730105", "H730111"],
    "0000020209": ["I5211", "I5102", "I6100"],
    "0000020210": ["J2506"],
    "0000020211": ["K4201"],
}


def fetch(stats_id, codes):
    rows, start = [], 1
    while True:
        q = {"appId": APPID, "statsDataId": stats_id, "cdCat01": ",".join(codes), "limit": 100000,
             "startPosition": start, "metaGetFlg": "N", "cntGetFlg": "N", "sectionHeaderFlg": "2"}
        url = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData?" + urllib.parse.urlencode(q)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=300) as r:
                    d = json.load(r)["GET_STATS_DATA"]
                break
            except Exception as e:
                print("retry", stats_id, e, file=sys.stderr); time.sleep(5)
        st = d["STATISTICAL_DATA"]
        vals = st["DATA_INF"]["VALUE"]
        for v in vals:
            rows.append((v["@cat01"].lstrip("#"), v["@area"], v["@time"][:4], v["$"], v.get("@unit", "")))
        nxt = st["RESULT_INF"].get("NEXT_KEY")
        print(stats_id, start, len(vals), file=sys.stderr)
        if not nxt:
            return rows
        start = nxt


def fetch_census_housing():
    """令和2年国勢調査 住宅の所有の関係・建て方別一般世帯数（全市区町村をカバー）。"""
    q = {"appId": APPID, "statsDataId": "0003445129", "cdCat01": "1,111,113", "cdCat02": "1,11,1304,1305",
         "limit": 100000, "metaGetFlg": "N", "cntGetFlg": "N", "sectionHeaderFlg": "2"}
    url = "https://api.e-stat.go.jp/rest/3.0/app/json/getStatsData?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=300) as r:
        vals = json.load(r)["GET_STATS_DATA"]["STATISTICAL_DATA"]["DATA_INF"]["VALUE"]
    df = pd.DataFrame([(v["@cat01"], v["@cat02"], v["@area"], v["$"]) for v in vals],
                      columns=["tenure", "building", "area", "value"])
    df.to_csv(os.path.join(os.path.dirname(OUT), "census2020_housing.csv"), index=False)
    print("census housing", df.shape)


if __name__ == "__main__":
    fetch_census_housing()
    allrows = []
    for sid, codes in ITEMS.items():
        allrows += fetch(sid, codes)
    df = pd.DataFrame(allrows, columns=["item", "area", "year", "value", "unit"])
    df.to_csv(OUT, index=False)
    print(df.shape, df.item.nunique(), df.area.nunique())

