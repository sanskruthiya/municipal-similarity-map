"""SSDS 生データから市区町村 × 指標の行列を作る（比率化・変換前の値）。"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "data", "raw", "ssds_municipal.csv")
GEO = os.path.join(HERE, "..", "data", "geo", "muni_simplified.geojson")
OUT = os.path.join(HERE, "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(RAW, dtype=str)
df["v"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["v"])


def latest(item, year=None):
    s = df[df.item == item]
    if year is None:
        year = s.year.max()
    s = s[s.year == str(year)]
    return s.set_index("area")["v"], str(year)


geo = json.load(open(GEO))
areas = pd.DataFrame([f["properties"] for f in geo["features"]]).set_index("code")
idx = areas.index

V, YEARS = {}, {}
for it in df.item.unique():
    s, y = latest(it)
    V[it] = s.reindex(idx)
    YEARS[it] = y
hc = pd.read_csv(os.path.join(HERE, "..", "data", "raw", "census2020_housing.csv"), dtype=str)
hc["v"] = pd.to_numeric(hc["value"], errors="coerce")
for (t, b), s_ in hc.groupby(["tenure", "building"]):
    V[f"census_h_{t}_{b}"] = s_.set_index("area")["v"].reindex(idx)
    YEARS[f"census_h_{t}_{b}"] = "2020"
pop15, _ = latest("A1101", 2015)
V["A1101_2015"] = pop15.reindex(idx)
YEARS["A1101_2015"] = "2015"
P = V["A1101"]

# (key, 日本語名, 分野, 式, 変換, 出典項目)
INDICATORS = [
    ("pop_density", "可住地人口密度(人/km²)", "人口・世帯", lambda: P / (V["B1103"] / 100), "log", ["A1101", "B1103"]),
    ("child_ratio", "15歳未満人口割合", "人口・世帯", lambda: V["A1301"] / P, None, ["A1301"]),
    ("elderly_ratio", "65歳以上人口割合", "人口・世帯", lambda: V["A1303"] / P, None, ["A1303"]),
    ("pop_change", "人口増減率(2015→2020)", "人口・世帯", lambda: P / V["A1101_2015"] - 1, None, ["A1101"]),
    ("proj_change_2050", "将来推計人口変化率(2020→2050)", "人口・世帯", lambda: V["A191007"] / P - 1, None, ["A191007"]),
    ("foreign_ratio", "外国人人口割合", "人口・世帯", lambda: V["A1700"] / P, "log", ["A1700"]),
    ("natural_change", "自然増減率(出生−死亡)/人口", "人口・世帯", lambda: (V["A4101"] - V["A4200"]) / P, None, ["A4101", "A4200"]),
    ("net_migration", "社会増減率(転入−転出)/人口", "人口・世帯", lambda: (V["A5103"] - V["A5104"]) / P, None, ["A5103", "A5104"]),
    ("mobility", "人口移動率(転入+転出)/人口", "人口・世帯", lambda: (V["A5103"] + V["A5104"]) / P, "log", ["A5103", "A5104"]),
    ("unmarried", "未婚者割合(15歳以上)", "人口・世帯", lambda: V["A161001"] / 100, None, ["A161001"]),
    ("single_hh", "単独世帯割合", "人口・世帯", lambda: V["A810105"] / V["A710101"], None, ["A810105"]),
    ("elderly_single_hh", "高齢単身世帯割合", "人口・世帯", lambda: V["A8301"] / V["A710101"], None, ["A8301"]),
    ("hh_size", "平均世帯人員", "人口・世帯", lambda: V["A710201"] / V["A710101"], None, ["A710201"]),

    ("day_night", "昼夜間人口比率", "都市構造・通勤", lambda: V["A6108"] / 100, "log", ["A6108"]),
    ("did_ratio", "DID(人口集中地区)人口割合", "都市構造・通勤", lambda: (V["A1801"].fillna(0)) / P, None, ["A1801"]),
    ("out_commute", "他市区町村への通勤者比率", "都市構造・通勤", lambda: V["F2705"] / V["F1102"], None, ["F2705"]),
    ("rail_commute", "通勤通学の鉄道利用割合", "都市構造・通勤", lambda: V["H730102"] / V["H7301"], "sqrt", ["H730102"]),
    ("car_commute", "通勤通学の自家用車利用割合", "都市構造・通勤", lambda: V["H730105"] / V["H7301"], None, ["H730105"]),
    ("bike_commute", "通勤通学の自転車利用割合", "都市構造・通勤", lambda: V["H730111"] / V["H7301"], "sqrt", ["H730111"]),
    ("forest_ratio", "林野率", "都市構造・通勤", lambda: V["B1105"] / V["B1101"], None, ["B1105", "B1101"]),

    ("primary_ind", "第1次産業就業者比率", "産業・労働", lambda: V["F2201"] / (V["F2201"] + V["F2211"] + V["F2221"]), "sqrt", ["F2201"]),
    ("secondary_ind", "第2次産業就業者比率", "産業・労働", lambda: V["F2211"] / (V["F2201"] + V["F2211"] + V["F2221"]), None, ["F2211"]),
    ("unemployment", "完全失業率", "産業・労働", lambda: V["F1107"] / V["F1101"], None, ["F1107"]),
    ("female_lfpr", "女性労働力率", "産業・労働", lambda: V["F110102"] / V["A141402"], None, ["F110102"]),
    ("elderly_workers", "就業者に占める65歳以上割合", "産業・労働", lambda: V["F2116"] / V["F1102"], None, ["F2116"]),
    ("regular_emp", "雇用者に占める正規雇用割合", "産業・労働", lambda: V["F2409"] / V["F2401"], None, ["F2409"]),
    ("jobs_per_capita", "従業者数/常住人口", "産業・労働", lambda: V["C2207"] / P, "log", ["C2207"]),
    ("emp_manufacturing", "従業者に占める製造業割合", "産業・労働", lambda: V["C220709"] / V["C2207"], "sqrt", ["C220709"]),
    ("emp_info_finance_prof", "従業者に占める情報通信・金融・専門サービス割合", "産業・労働", lambda: (V["C220711"] + V["C220716"] + V["C220718"]) / V["C2207"], "log", ["C220711", "C220716", "C220718"]),
    ("emp_hospitality", "従業者に占める宿泊・飲食サービス割合", "産業・労働", lambda: V["C220719"] / V["C2207"], "log", ["C220719"]),
    ("emp_health_welfare", "従業者に占める医療・福祉割合", "産業・労働", lambda: V["C220722"] / V["C2207"], None, ["C220722"]),

    ("income", "納税義務者1人当たり課税対象所得(千円)", "所得・教育", lambda: V["C120110"] / V["C120120"], "log", ["C120110", "C120120"]),
    ("univ_grad", "最終学歴が大学・大学院の割合", "所得・教育", lambda: V["E9106"] / V["E9101"], None, ["E9106"]),
    ("low_edu", "最終学歴が小学校・中学校の割合", "所得・教育", lambda: V["E9102"] / V["E9101"], None, ["E9102"]),

    ("owner_ratio", "持ち家世帯割合", "住宅", lambda: V["census_h_111_1"] / V["census_h_1_1"], None, ["census_h_111_1", "census_h_1_1"]),
    ("private_rent_ratio", "民営借家世帯割合", "住宅", lambda: V["census_h_113_1"].fillna(0) / V["census_h_1_1"], None, ["census_h_113_1"]),
    ("detached_ratio", "一戸建て居住世帯割合", "住宅", lambda: V["census_h_1_11"] / V["census_h_1_1"], None, ["census_h_1_11"]),
    ("highrise_ratio", "11階建以上の共同住宅居住世帯割合", "住宅", lambda: (V["census_h_1_1304"].fillna(0) + V["census_h_1_1305"].fillna(0)) / V["census_h_1_1"], "sqrt", ["census_h_1_1304", "census_h_1_1305"]),

    ("retail_pc", "小売店数(人口千人当たり)", "生活利便・医療", lambda: V["H6130"] / P * 1000, "log", ["H6130"]),
    ("restaurant_pc", "飲食店数(人口千人当たり)", "生活利便・医療", lambda: V["H6131"] / P * 1000, "log", ["H6131"]),
    ("doctors_pc", "医師数(人口10万人当たり)", "生活利便・医療", lambda: V["I6100"] / P * 1e5, "sqrt", ["I6100"]),
    ("clinics_pc", "一般診療所数(人口10万人当たり)", "生活利便・医療", lambda: V["I5102"] / P * 1e5, "log1p", ["I5102"]),
    ("beds_pc", "病院病床数(人口10万人当たり)", "生活利便・医療", lambda: V["I5211"] / P * 1e5, "sqrt", ["I5211"]),
]

feat = pd.DataFrame(index=idx)
meta = []
for key, label, group, fn, tr, src in INDICATORS:
    s = fn().replace([np.inf, -np.inf], np.nan)
    feat[key] = s
    meta.append({"key": key, "label": label, "group": group, "transform": tr or "none",
                 "years": sorted({YEARS[x] for x in src}), "source_items": src,
                 "missing": int(s.isna().sum())})

# --- Phase 2: 国土数値情報・メッシュ人口からの指標（06_ksj_features.py の出力）
KSJ = os.path.join(OUT, "ksj_features.csv")
KSJ_INDICATORS = [
    ("pw_density", "人口加重人口密度(人/km², 250mメッシュ)", "log", "2020", "国勢調査250mメッシュ"),
    ("station_dist", "最寄り駅までの人口加重平均距離(km)", "log", "2020/2023", "S12-23 駅別乗降客数"),
    ("hospital_dist", "最寄り病院までの人口加重平均距離(km)", "log", "2020", "P04-20 医療機関"),
    ("agri_ratio", "農用地率(陸域に占める田・畑等)", "sqrt", "2021", "L03-a-21 土地利用3次メッシュ"),
    ("paddy_share", "農用地に占める田の割合", None, "2021", "L03-a-21 土地利用3次メッシュ"),
    ("building_ratio", "建物用地率(陸域に占める割合)", "sqrt", "2021", "L03-a-21 土地利用3次メッシュ"),
]
if os.path.exists(KSJ):
    ksj = pd.read_csv(KSJ, dtype={"code": str}).set_index("code").reindex(idx)
    for key, label, tr, year, src in KSJ_INDICATORS:
        feat[key] = ksj[key]
        meta.append({"key": key, "label": label, "group": "立地・土地利用", "transform": tr or "none",
                     "years": [year], "source_items": [src], "missing": int(ksj[key].isna().sum()), "phase2": True})

feat.index.name = "code"
feat.to_csv(os.path.join(OUT, "indicators_raw.csv"))
json.dump(meta, open(os.path.join(OUT, "indicators_meta.json"), "w"), ensure_ascii=False, indent=1)
areas.to_csv(os.path.join(OUT, "areas.csv"))
print(feat.shape)
print(pd.DataFrame(meta)[["key", "group", "years", "missing"]].to_string())
print(feat.describe().T[["min", "50%", "max"]].to_string())
