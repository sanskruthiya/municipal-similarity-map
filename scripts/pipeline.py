"""特徴量空間の構築（03_embed.py と 07_sensitivity.py で共有）。"""
import json, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed")
RAW = os.path.join(HERE, "..", "data", "raw")


def load():
    raw = pd.read_csv(os.path.join(PROC, "indicators_raw.csv"), dtype={"code": str}).set_index("code")
    areas = pd.read_csv(os.path.join(PROC, "areas.csv"), dtype={"code": str}).set_index("code")
    meta = json.load(open(os.path.join(PROC, "indicators_meta.json")))
    return raw, areas, meta


def disrupted(raw, threshold=0.5):
    """原発事故の避難指示等で2020年の統計が平常時と大きく異なる自治体（2020年人口が2010年比 threshold 未満）。"""
    ss = pd.read_csv(os.path.join(RAW, "ssds_municipal.csv"), dtype=str)
    pop = ss[ss.item == "A1101"].assign(v=lambda d: pd.to_numeric(d.value, errors="coerce")).pivot(index="area", columns="year", values="v")
    ratio = (pop["2020"] / pop["2010"].fillna(pop["2015"])).reindex(raw.index)
    return ratio.index[(ratio < threshold) | ~(pop["2020"].reindex(raw.index) > 0)]


def transform(df, meta):
    X = df.copy()
    for m in meta:
        k, tr = m["key"], m["transform"]
        x = X[k]
        if tr == "log":
            pos = x[x > 0]
            X[k] = np.log(x.clip(lower=0) + pos.quantile(0.01) / 2)
        elif tr == "log1p":
            X[k] = np.log1p(x)
        elif tr == "sqrt":
            X[k] = np.sqrt(x.clip(lower=0))
    return X


def build_space(raw, meta, excluded=(), weighting="domain", imputer="knn", keys=None):
    """変換 → ロバスト標準化(中央値・IQR, ±4クリップ) → 欠損補完 → 分野重み。
    戻り値: (Xw 重み付き空間, Z 標準化値, 補完マスク)"""
    meta = [m for m in meta if keys is None or m["key"] in keys]
    cols = [m["key"] for m in meta]
    X = transform(raw.drop(index=list(excluded))[cols], meta)
    med = X.median()
    scale = (X.quantile(0.75) - X.quantile(0.25)) / 1.349
    scale = scale.where(scale > 0, X.std())
    Z = ((X - med) / scale).clip(-4, 4)
    mask = Z.isna()
    imp = KNNImputer(n_neighbors=10, weights="distance") if imputer == "knn" else SimpleImputer(strategy="median")
    Z = pd.DataFrame(imp.fit_transform(Z), index=Z.index, columns=Z.columns)
    if weighting == "domain":
        groups = pd.Series({m["key"]: m["group"] for m in meta})
        w = 1 / np.sqrt(groups.map(groups.value_counts()))
    else:
        w = pd.Series(1.0, index=cols)
    return Z * w, Z, mask
