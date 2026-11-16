"""指標の変換・標準化 → 類似度(top-k) → 3次元埋め込み(PCA / UMAP) → OKLab 色 → 評価指標。"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
import umap

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "..", "data", "processed")
WEB = os.path.join(HERE, "..", "web", "data")
os.makedirs(WEB, exist_ok=True)
K = 10
SEED = 42

raw = pd.read_csv(os.path.join(PROC, "indicators_raw.csv"), dtype={"code": str}).set_index("code")
areas = pd.read_csv(os.path.join(PROC, "areas.csv"), dtype={"code": str}).set_index("code")
meta = json.load(open(os.path.join(PROC, "indicators_meta.json")))
keys = [m["key"] for m in meta]

# --- 除外: 原発事故の避難指示等で2020年の統計が平常時と大きく異なる自治体（人口が2010年比で半分未満）
ss = pd.read_csv(os.path.join(HERE, "..", "data", "raw", "ssds_municipal.csv"), dtype=str)
pop = ss[ss.item == "A1101"].assign(v=lambda d: pd.to_numeric(d.value, errors="coerce")).pivot(index="area", columns="year", values="v")
ratio = (pop["2020"] / pop["2010"].fillna(pop["2015"])).reindex(raw.index)
excluded = ratio.index[(ratio < 0.5) | ~(pop["2020"].reindex(raw.index) > 0)]
print("excluded:", list(areas.loc[excluded, "name"]))
use = raw.drop(index=excluded)

# --- 変換
X = use.copy()
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

# --- ロバスト標準化（中央値・IQR）と外れ値の切り詰め
med = X.median()
scale = (X.quantile(0.75) - X.quantile(0.25)) / 1.349
scale = scale.where(scale > 0, X.std())
Z = ((X - med) / scale).clip(-4, 4)

# --- 欠損補完（所得: 政令市の区、住宅: 住宅・土地統計調査の非対象町村）
imputed_mask = Z.isna()
Z = pd.DataFrame(KNNImputer(n_neighbors=10, weights="distance").fit_transform(Z), index=Z.index, columns=Z.columns)

# --- 分野ごとに寄与を揃える重み（各分野の重み二乗和 = 1）
groups = pd.Series({m["key"]: m["group"] for m in meta})
w = 1 / np.sqrt(groups.map(groups.value_counts()))
Xw = Z * w

# --- 主成分分析（診断用）
pca_full = PCA().fit(Xw)
evr = pca_full.explained_variance_ratio_
print("PCA explained (cum):", np.round(np.cumsum(evr)[:10], 3))
loadings = pd.DataFrame(pca_full.components_[:5].T, index=keys, columns=[f"PC{i+1}" for i in range(5)])

# --- 類似度: 重み付き標準化空間でのユークリッド距離
nn = NearestNeighbors(n_neighbors=K + 1).fit(Xw)
dist, ind = nn.kneighbors(Xw)
codes = Xw.index.to_numpy()
all_d = np.sort(nn.kneighbors(Xw, n_neighbors=len(Xw))[0][:, 1:].ravel())
def sim_score(d):  # 全ペア距離中の順位（100 = 最も近い）
    return float(np.round(100 * (1 - np.searchsorted(all_d, d) / len(all_d)), 2))

# --- 3次元埋め込み
emb = {
    "pca": PCA(n_components=3, random_state=SEED).fit_transform(Xw),
    "umap": umap.UMAP(n_components=3, n_neighbors=20, min_dist=0.3, random_state=SEED).fit_transform(Xw),
}

# --- OKLab → sRGB
M2 = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
               [-1.2684380046, 2.6097574011, -0.3413193965],
               [-0.0041960863, -0.7034186147, 1.7076147010]])
def oklab_to_linrgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    lms = np.stack([l_ ** 3, m_ ** 3, s_ ** 3], -1)
    return lms @ M2.T
def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)
def gamut_clip(lab):
    """色相と明度を保ったまま彩度を下げて sRGB 内に収める。"""
    lab = lab.copy()
    for i in range(len(lab)):
        lo, hi = 0.0, 1.0
        if np.all((oklab_to_linrgb(lab[i]) >= 0) & (oklab_to_linrgb(lab[i]) <= 1)):
            continue
        for _ in range(25):
            mid = (lo + hi) / 2
            t = lab[i].copy(); t[1:] *= mid
            ok = np.all((oklab_to_linrgb(t) >= -1e-6) & (oklab_to_linrgb(t) <= 1 + 1e-6))
            lo, hi = (mid, hi) if ok else (lo, mid)
        lab[i, 1:] *= lo
    return lab
def to_hex(rgb):
    return ["#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c) for c in rgb]

L_RANGE = (0.42, 0.92)
AB_MAX = 0.20
def color_oklab(E):
    """等方スケーリング: 3軸を同じ倍率で OKLab に写すので、埋め込み空間の距離比が色差に保たれる。
    埋め込みを主軸回転し、分散最大の2軸を色相平面(a,b)、第3軸を明度 L に割り当てる。"""
    E = E - np.median(E, axis=0)
    E = PCA(n_components=3).fit_transform(E)
    span = np.percentile(E, 98, axis=0) - np.percentile(E, 2, axis=0)
    s = min(2 * AB_MAX / span[0], (L_RANGE[1] - L_RANGE[0]) / span[2])
    lab = np.stack([np.mean(L_RANGE) + s * E[:, 2], s * E[:, 0], s * E[:, 1]], -1)
    lab[:, 0] = lab[:, 0].clip(*L_RANGE)
    lab = gamut_clip(lab)
    return lab, to_hex(lin_to_srgb(oklab_to_linrgb(lab)))
def color_rgb_naive(E):
    """比較用: 各軸を独立に min-max して RGB に直接割り当てる素朴な方法。"""
    lo, hi = np.percentile(E, 1, axis=0), np.percentile(E, 99, axis=0)
    rgb = ((E - lo) / (hi - lo)).clip(0, 1)
    return rgb, to_hex(rgb)

def knn_overlap(A, B, k=K):
    ia = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)[1][:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(B)[1][:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ia, ib)]))
rng = np.random.default_rng(SEED)
pairs = rng.integers(0, len(Xw), size=(40000, 2))
def dist_corr(A, B):
    da = np.linalg.norm(A[pairs[:, 0]] - A[pairs[:, 1]], axis=1)
    db = np.linalg.norm(B[pairs[:, 0]] - B[pairs[:, 1]], axis=1)
    return float(spearmanr(da, db)[0])

X_np = Xw.to_numpy()
colors, metrics = {}, {}
for name, E in emb.items():
    lab, hx = color_oklab(E)
    colors[f"{name}_oklab"] = hx
    rgb, hx2 = color_rgb_naive(E)
    colors[f"{name}_rgb"] = hx2
    metrics[name] = {
        "trustworthiness_k10": round(float(trustworthiness(X_np, E, n_neighbors=K)), 4),
        "knn_overlap_k10_embedding": round(knn_overlap(X_np, E), 4),
        "knn_overlap_k10_oklab_color": round(knn_overlap(X_np, lab), 4),
        "knn_overlap_k10_rgb_color(perceptual)": None,
        "dist_spearman_embedding": round(dist_corr(X_np, E), 4),
        "dist_spearman_oklab_color": round(dist_corr(X_np, lab), 4),
    }
    # 素朴RGBの色を OKLab に変換して知覚的な色差での保存度を測る
    def srgb_to_oklab(rgb):
        c = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                       [0.2119034982, 0.6806995451, 0.1073969566],
                       [0.0883024619, 0.2817188376, 0.6299787005]])
        lms = np.cbrt(c @ M1.T)
        M = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                      [1.9779984951, -2.4285922050, 0.4505937099],
                      [0.0259040371, 0.7827717662, -0.8086757660]])
        return lms @ M.T
    lab_naive = srgb_to_oklab(rgb)
    metrics[name]["knn_overlap_k10_rgb_color(perceptual)"] = round(knn_overlap(X_np, lab_naive), 4)
    metrics[name]["dist_spearman_rgb_color(perceptual)"] = round(dist_corr(X_np, lab_naive), 4)
print(json.dumps(metrics, indent=1))

clusters = KMeans(n_clusters=12, n_init=20, random_state=SEED).fit_predict(Xw)

# --- 出力
recs = []
pos = {c: i for i, c in enumerate(codes)}
for i, c in enumerate(codes):
    recs.append({
        "code": c, "name": areas.loc[c, "name"], "pref": areas.loc[c, "pref"],
        "colors": {m: colors[m][i] for m in colors},
        "cluster": int(clusters[i]),
        "neighbors": [{"code": codes[j], "d": round(float(d), 3), "sim": sim_score(d)} for j, d in zip(ind[i, 1:], dist[i, 1:])],
        "z": [round(float(v), 2) for v in Z.loc[c, keys]],
        "raw": [None if pd.isna(v) else round(float(v), 5) for v in raw.loc[c, keys]],
        "imputed": [k for k in keys if imputed_mask.loc[c, k]],
        "emb": {n: [round(float(v), 3) for v in emb[n][i]] for n in emb},
    })
for c in excluded:
    recs.append({"code": c, "name": areas.loc[c, "name"], "pref": areas.loc[c, "pref"], "excluded": True,
                 "raw": [None if pd.isna(v) else round(float(v), 5) for v in raw.loc[c, keys]]})

out_meta = {
    "indicators": meta, "k": K, "metrics": metrics,
    "pca_explained": [round(float(v), 4) for v in evr[:10]],
    "pca_loadings": {pc: {k: round(float(v), 3) for k, v in loadings[pc].items()} for pc in loadings},
    "excluded_note": "東日本大震災・原発事故の影響で2020年人口が2010年比50%未満の自治体は類似度計算から除外",
}
json.dump({"meta": out_meta, "areas": recs}, open(os.path.join(WEB, "attributes.json"), "w"), ensure_ascii=False, separators=(",", ":"))
Z.assign(**{f"col_{m}": colors[m] for m in colors}).to_csv(os.path.join(PROC, "zscores_colors.csv"))
pd.DataFrame([{"code": c, "rank": r + 1, "neighbor": codes[j], "distance": float(d)}
              for i, c in enumerate(codes) for r, (j, d) in enumerate(zip(ind[i, 1:], dist[i, 1:]))]
             ).to_csv(os.path.join(PROC, "neighbors_top10.csv"), index=False)
for pc in ["PC1", "PC2", "PC3"]:
    s = loadings[pc].sort_values()
    print(pc, "−:", list(s.index[:4]), "+:", list(s.index[-4:]))
