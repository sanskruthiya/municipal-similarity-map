"""類型（大分類 K1・小分類 K2 の2階層）を作る。

- 大分類: 重み付き標準化空間（類似度と同じ空間）で k-means（K1）
- 小分類: 大分類ごとに k-means。各大分類の小分類数は、分類内のばらつき（偏差平方和）に比例して K2 を配分
  → 小分類は必ずひとつの大分類に含まれる（入れ子）
- 安定度: 指標の80%をランダムに選んで再計算（N_BOOT回）し、各市区町村について
  「同じ類型に入った市区町村の集合」の一致度（Jaccard）を平均

出力: web/data/typology.json, data/processed/typology_profiles.json（名前付け用の要約）
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pipeline import load, disrupted, build_space, RAW, PROC

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web", "data")
K1, K2, SEED, N_BOOT = 8, 30, 42, 20


def typology(X, seed=SEED):
    maj = KMeans(K1, n_init=20, random_state=seed).fit_predict(X)
    sse = np.array([((X[maj == m] - X[maj == m].mean(0)) ** 2).sum() for m in range(K1)])
    alloc = np.maximum(1, np.floor(sse / sse.sum() * K2)).astype(int)
    while alloc.sum() < K2:
        alloc[np.argmax(sse / alloc)] += 1
    while alloc.sum() > K2:
        alloc[np.argmax(np.where(alloc > 1, alloc, 0))] -= 1
    sub, off = np.zeros(len(X), int), 0
    for m in range(K1):
        idx = np.where(maj == m)[0]
        sub[idx] = off + (KMeans(alloc[m], n_init=20, random_state=seed).fit_predict(X[idx]) if alloc[m] > 1 else 0)
        off += alloc[m]
    return maj, sub


def members(lab):
    return {c: frozenset(np.where(lab == c)[0]) for c in np.unique(lab)}


if __name__ == "__main__":
    raw, areas, meta = load()
    keys = [m["key"] for m in meta]
    excl = disrupted(raw)
    Xw, Z, mask = build_space(raw, meta, excl)
    X, codes = Xw.to_numpy(), Xw.index.to_numpy()
    maj, sub = typology(X)

    # 大分類の番号を「都市度の目安」の高い順に並べ替え、小分類も大分類内で同様に並べる
    urb = (Z["pw_density"] + Z["did_ratio"] + Z["building_ratio"] - Z["forest_ratio"] - Z["car_commute"]).to_numpy()
    mo = np.argsort([-urb[maj == m].mean() for m in range(K1)])
    maj_new = np.empty(K1, int); maj_new[mo] = np.arange(K1)
    subs = sorted(np.unique(sub), key=lambda s: (maj_new[maj[sub == s][0]], -urb[sub == s].mean()))
    sub_new = {s: i for i, s in enumerate(subs)}
    maj = maj_new[maj]; sub = np.array([sub_new[s] for s in sub])

    # 安定度
    rng = np.random.default_rng(SEED)
    mb_maj, mb_sub = members(maj), members(sub)
    st_maj, st_sub, ari_maj, ari_sub = np.zeros(len(X)), np.zeros(len(X)), [], []
    for b in range(N_BOOT):
        ks = list(rng.choice(keys, size=int(len(keys) * 0.8), replace=False))
        m2, s2 = typology(build_space(raw, meta, excl, keys=ks)[0].to_numpy(), seed=SEED + b + 1)
        ari_maj.append(adjusted_rand_score(maj, m2)); ari_sub.append(adjusted_rand_score(sub, s2))
        for base, alt, acc, mb in ((maj, m2, st_maj, mb_maj), (sub, s2, st_sub, mb_sub)):
            ma = members(alt)
            for i in range(len(X)):
                A, B = mb[base[i]], ma[alt[i]]
                acc[i] += len(A & B) / len(A | B) / N_BOOT
    print("ARI major", round(np.mean(ari_maj), 3), "minor", round(np.mean(ari_sub), 3))

    # 名前付け用の要約
    ss = pd.read_csv(os.path.join(RAW, "ssds_municipal.csv"), dtype=str)
    pop = ss[(ss.item == "A1101") & (ss.year == "2020")].set_index("area").value.pipe(pd.to_numeric, errors="coerce").reindex(codes)
    zc = Z.loc[codes, keys]
    prof = []
    for lvl, lab in (("major", maj), ("minor", sub)):
        for c in np.unique(lab):
            mk = lab == c
            med = zc[mk].median()
            top = med.abs().sort_values(ascending=False).index[:8]
            cen = X[mk].mean(0)
            near = codes[mk][np.argsort(((X[mk] - cen) ** 2).sum(1))[:6]]
            big = pop[mk].sort_values(ascending=False).index[:6]
            prof.append({"level": lvl, "id": int(c), "major": int(maj[mk][0]), "n": int(mk.sum()),
                         "stability": round(float((st_maj if lvl == "major" else st_sub)[mk].mean()), 3),
                         "pop_total": int(pop[mk].sum()),
                         "features": {k: round(float(med[k]), 2) for k in top},
                         "central": [f'{areas.loc[x, "pref"]}{areas.loc[x, "name"]}' for x in near],
                         "largest": [f'{areas.loc[x, "pref"]}{areas.loc[x, "name"]}' for x in big],
                         "prefs": areas.loc[codes[mk], "pref"].value_counts().head(4).to_dict()})
    json.dump(prof, open(os.path.join(PROC, "typology_profiles.json"), "w"), ensure_ascii=False, indent=1)

    # 色: 大分類 = 色相（都市的な赤橙 → 農村的な青。遠隔の町村は紫）、小分類 = 大分類の色相のまま明るさを変える
    names = json.load(open(os.path.join(HERE, "typology_names.json")))
    HUES = [28, 52, 80, 125, 158, 200, 305, 252]
    def oklch(L, C, h):
        while True:
            a, b = C * np.cos(np.radians(h)), C * np.sin(np.radians(h))
            l_, m_, s_ = L + .3963377774 * a + .2158037573 * b, L - .1055613458 * a - .0638541728 * b, L - .0894841775 * a - 1.2914855480 * b
            rgb = np.array([[4.0767416621, -3.3077115913, .2309699292], [-1.2684380046, 2.6097574011, -.3413193965],
                            [-.0041960863, -.7034186147, 1.7076147010]]) @ np.array([l_ ** 3, m_ ** 3, s_ ** 3])
            if rgb.min() >= 0 and rgb.max() <= 1 or C < .01:
                break
            C -= .005
        rgb = np.clip(rgb, 0, 1)
        rgb = np.where(rgb <= .0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - .055)
        return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)
    P = {(p["level"], p["id"]): p for p in prof}
    major_out, minor_out = [], []
    for m in range(K1):
        major_out.append({"name": names["major"][m], "color": oklch(.70, .14, HUES[m]), "n": P["major", m]["n"], "stability": P["major", m]["stability"]})
        ms = sorted(set(sub[maj == m]))
        Ls = np.linspace(.84, .54, len(ms)) if len(ms) > 1 else [.70]
        hs = np.linspace(-8, 8, len(ms)) if len(ms) > 1 else [0]
        for s_, L, dh in zip(ms, Ls, hs):
            minor_out.append({"name": names["minor"][s_], "major": m, "color": oklch(float(L), .13, HUES[m] + dh),
                              "n": P["minor", s_]["n"], "stability": P["minor", s_]["stability"]})
    out = {"meta": {"k_major": K1, "k_minor": K2, "n_boot": N_BOOT,
                    "ari_major": round(float(np.mean(ari_maj)), 3), "ari_minor": round(float(np.mean(ari_sub)), 3)},
           "major": major_out, "minor": minor_out,
           "areas": {c: [int(maj[i]), int(sub[i]), round(float(st_maj[i]), 2), round(float(st_sub[i]), 2)] for i, c in enumerate(codes)}}
    json.dump(out, open(os.path.join(WEB, "typology.json"), "w"), ensure_ascii=False, separators=(",", ":"))
    print(len(out["areas"]), "areas")
