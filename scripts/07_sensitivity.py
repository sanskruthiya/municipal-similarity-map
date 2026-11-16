"""感度分析: 前処理・指標選択・除外ルール・UMAPパラメータを変えたとき、類似Top10と配色がどれだけ変わるか。

- 類似Top10の一致率 = ベースラインとの共通近傍数 / 10（両方に含まれる市区町村で平均）
- 市区町村ごとの安定度 = 指標の80%をランダムに選んだ再計算（N_BOOT回）でのTop10一致率の平均
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
import umap
from pipeline import load, disrupted, build_space, PROC

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web", "data")
K, SEED, N_BOOT = 10, 42, 30

raw, areas, meta = load()
excl = disrupted(raw)
keys = [m["key"] for m in meta]
groups = {m["key"]: m["group"] for m in meta}


def topk(Xw):
    idx = NearestNeighbors(n_neighbors=K + 1).fit(Xw).kneighbors(Xw)[1][:, 1:]
    codes = Xw.index.to_numpy()
    return pd.Series([set(codes[r]) for r in idx], index=Xw.index)


def overlap(a, b):
    common = a.index.intersection(b.index)
    return pd.Series([len(a[c] & b[c]) / K for c in common], index=common)


base_X, _, _ = build_space(raw, meta, excl)
base = topk(base_X)

variants = {
    "分野重みなし（全指標均等）": dict(weighting="none"),
    "欠損を中央値で補完": dict(imputer="median"),
    "除外なし（福島7町村も含める）": dict(excluded=()),
}
for g in dict.fromkeys(groups.values()):
    variants[f"分野「{g}」を除く"] = dict(keys=[k for k in keys if groups[k] != g])
phase1 = [m["key"] for m in meta if not m.get("phase2")]
if len(phase1) < len(keys) and not any(set(v.get("keys", ())) == set(phase1) for v in variants.values()):
    variants["Phase 1 の指標のみ"] = dict(keys=phase1)

rows = []
for name, kw in variants.items():
    kw = {"excluded": excl, **kw}
    ov = overlap(base, topk(build_space(raw, meta, **kw)[0]))
    rows.append({"variant": name, "mean_overlap": round(float(ov.mean()), 3), "median_overlap": round(float(ov.median()), 3),
                 "share_ge_half": round(float((ov >= 0.5).mean()), 3)})
    print(rows[-1])

rng = np.random.default_rng(SEED)
boot = []
for b in range(N_BOOT):
    sub = list(rng.choice(keys, size=int(len(keys) * 0.8), replace=False))
    boot.append(overlap(base, topk(build_space(raw, meta, excl, keys=sub)[0])))
stab = pd.concat(boot, axis=1).mean(axis=1)
rows.append({"variant": f"指標の80%をランダム抽出（{N_BOOT}回平均）", "mean_overlap": round(float(stab.mean()), 3),
             "median_overlap": round(float(stab.median()), 3), "share_ge_half": round(float((stab >= 0.5).mean()), 3)})
print(rows[-1])

# --- UMAP パラメータ・乱数シード
Xn = base_X.to_numpy()
pairs = rng.integers(0, len(Xn), size=(40000, 2))
def pdist_s(E):
    return np.linalg.norm(E[pairs[:, 0]] - E[pairs[:, 1]], axis=1)
def knn_idx(E):
    return NearestNeighbors(n_neighbors=K + 1).fit(E).kneighbors(E)[1][:, 1:]
ref = umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.3, random_state=SEED).fit_transform(Xn)
ref_knn, ref_d = knn_idx(ref), pdist_s(ref)
X_knn = knn_idx(Xn)
urows = []
for nn_ in (10, 20, 50):
    for md in (0.1, 0.3, 0.6):
        for seed in (SEED, 1, 2):
            if seed != SEED and (nn_, md) != (20, 0.3):
                continue
            E = umap.UMAP(n_components=3, n_neighbors=nn_, min_dist=md, random_state=seed).fit_transform(Xn)
            ek = knn_idx(E)
            urows.append({
                "n_neighbors": nn_, "min_dist": md, "seed": seed,
                "trustworthiness": round(float(trustworthiness(Xn, E, n_neighbors=K)), 4),
                "knn_overlap_vs_original": round(float(np.mean([len(set(a) & set(b)) / K for a, b in zip(X_knn, ek)])), 4),
                "knn_overlap_vs_default_umap": round(float(np.mean([len(set(a) & set(b)) / K for a, b in zip(ref_knn, ek)])), 4),
                "dist_spearman_vs_default_umap": round(float(spearmanr(ref_d, pdist_s(E))[0]), 4),
            })
            print(urows[-1])

out = {"neighbor_variants": rows, "umap_variants": urows, "n_boot": N_BOOT, "k": K}
json.dump(out, open(os.path.join(PROC, "sensitivity.json"), "w"), ensure_ascii=False, indent=1)
json.dump({"summary": out, "stability": {c: round(float(v), 3) for c, v in stab.items()}},
          open(os.path.join(WEB, "sensitivity.json"), "w"), ensure_ascii=False, separators=(",", ":"))
