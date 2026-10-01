"""
kNN (cosseno) sobre TF-IDF palavra 1-2gram, voto ponderado por similaridade.

Hipótese: quase-duplicatas (mesmo template, nome/número trocado) atravessam
treino/validação, pois os folds só agrupam textos IDÊNTICOS. Memorização
explícita captura esse sinal de forma diferente dos modelos parametrizados.

Tudo fixo, sem grid: k=25, peso = sim^4, votos normalizados por prior de classe.

Uso: python src/experiments/knn_tfidf.py
Saídas (results/): knn_tfidf_cv.csv | _oof.csv | _summary.csv | _vs_v3.csv
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT / "data" / "splits" / "folds.csv"
RESULTS = ROOT / "results"
V3_OOF = RESULTS / "ensemble_majority_5_v3_norberto_oof.csv"

LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
K = 25
POWER = 4
CHUNK = 500
BUCKETS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0001)]  # pré-declaradas


def load_data() -> pd.DataFrame:
    df = pd.read_excel(TRAIN_PATH).reset_index(drop=True)
    folds = pd.read_csv(FOLDS_PATH)
    if len(df) != len(folds):
        raise ValueError("train.xlsx e folds.csv incompatíveis.")
    if not np.array_equal(folds["row_index"].to_numpy(), np.arange(len(df))):
        raise ValueError("folds.csv não corresponde ao índice de train.xlsx.")
    df["resp_text"] = df["resp_text"].fillna("").astype(str)
    df["clarity"] = df["clarity"].astype(str).str.strip()
    df["y"] = df["clarity"].map(LABEL2ID)
    if df["y"].isna().any():
        raise ValueError("Labels desconhecidos.")
    df["row_id"] = np.arange(len(df))
    df["fold"] = folds["fold"].astype(int).to_numpy()
    for f in sorted(df["fold"].unique()):
        if set(df.loc[df["fold"] != f, "resp_text"]) & set(df.loc[df["fold"] == f, "resp_text"]):
            raise ValueError(f"Leakage de resp_text no fold {f}.")
    return df


def duplicate_ceiling(df: pd.DataFrame) -> None:
    """Teto de acurácia nas linhas com texto repetido (ruído de anotação)."""
    g = df.groupby("resp_text")["y"]
    size = g.transform("size")
    dup = size > 1
    best = df[dup].groupby("resp_text")["y"].agg(lambda s: s.value_counts().iloc[0]).sum()
    n = int(dup.sum())
    print(f"Linhas em grupos duplicados: {n} ({n / len(df):.2%}) | "
          f"teto de acurácia nelas: {best / n:.2%}")


def knn_fold(tr, va, texts, y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.995,
                          sublinear_tf=True, dtype=np.float32)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xva = vec.transform([texts[i] for i in va])
    ytr = y[tr]
    prior = np.bincount(ytr, minlength=3) / len(ytr)

    probs = np.zeros((len(va), 3))
    max_sim = np.zeros(len(va))
    XtrT = Xtr.T.tocsr()

    for s in range(0, len(va), CHUNK):
        sim = (Xva[s:s + CHUNK] @ XtrT).toarray()          # cosseno (L2-normalizado)
        top = np.argpartition(-sim, K, axis=1)[:, :K]
        ts = np.take_along_axis(sim, top, axis=1)
        w = np.clip(ts, 0, None) ** POWER
        votes = np.zeros((sim.shape[0], 3))
        for c in range(3):
            votes[:, c] = (w * (ytr[top] == c)).sum(1)
        votes = votes / prior                               # compensa prior de classe
        probs[s:s + CHUNK] = votes / np.maximum(votes.sum(1, keepdims=True), 1e-12)
        max_sim[s:s + CHUNK] = ts.max(1)
    return probs, max_sim


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_data()
    duplicate_ceiling(df)

    y = df["y"].to_numpy(int)
    folds = df["fold"].to_numpy(int)
    texts = df["resp_text"].tolist()
    probs = np.zeros((len(df), 3))
    max_sim = np.zeros(len(df))
    rows = []

    for f in sorted(np.unique(folds)):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        t0 = time.perf_counter()
        probs[va], max_sim[va] = knn_fold(tr, va, texts, y)
        pred = probs[va].argmax(1)
        rows.append({"fold": int(f), "k": K, "power": POWER,
                     "accuracy": accuracy_score(y[va], pred),
                     "macro_f1": f1_score(y[va], pred, average="macro"),
                     "runtime_seconds": time.perf_counter() - t0})
        print(f"Fold {f}: acc={rows[-1]['accuracy']:.6f} f1={rows[-1]['macro_f1']:.6f}")

    pred = probs.argmax(1)
    cv = pd.DataFrame(rows)
    oof = pd.DataFrame({
        "row_id": df["row_id"], "fold": folds, "y_true_id": y, "y_pred_id": pred,
        "y_true": [ID2LABEL[v] for v in y], "y_pred": [ID2LABEL[int(v)] for v in pred],
        "prob_c1": probs[:, 0], "prob_c234": probs[:, 1], "prob_c5": probs[:, 2],
        "max_sim": max_sim, "correct": pred == y, "resp_text": df["resp_text"],
    })
    f1s = f1_score(y, pred, labels=[0, 1, 2], average=None)
    summary = pd.DataFrame([{
        "model": "knn_tfidf", "k": K, "power": POWER,
        "oof_accuracy": accuracy_score(y, pred),
        "oof_macro_f1": f1_score(y, pred, average="macro"),
        "f1_c1": f1s[0], "f1_c234": f1s[1], "f1_c5": f1s[2],
        "cv_accuracy_mean": cv["accuracy"].mean(),
        "cv_accuracy_std": cv["accuracy"].std(ddof=1),
    }])
    cv.to_csv(RESULTS / "knn_tfidf_cv.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(RESULTS / "knn_tfidf_oof.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "knn_tfidf_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"OOF Accuracy = {summary.at[0, 'oof_accuracy']:.6f} | "
          f"Macro-F1 = {summary.at[0, 'oof_macro_f1']:.6f}")
    print(f"F1 c1={f1s[0]:.4f} | c234={f1s[1]:.4f} | c5={f1s[2]:.4f}")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))

    # Teste direto da hipótese: acurácia por faixa de similaridade ao vizinho mais próximo.
    base = None
    if V3_OOF.exists():
        v3 = pd.read_csv(V3_OOF).set_index("row_id")
        base = v3.loc[df["row_id"], "y_pred_id_v3"].to_numpy(int)

    print("\nPOR FAIXA DE SIMILARIDADE (max cosseno ao vizinho mais próximo)")
    print(f"{'faixa':>12} {'n':>6} {'kNN acc':>9} {'V3 acc':>9}")
    for lo, hi in BUCKETS:
        m = (max_sim >= lo) & (max_sim < hi)
        if m.sum() == 0:
            continue
        v3a = f"{accuracy_score(y[m], base[m]):.4f}" if base is not None else "  n/a"
        print(f"[{lo:.2f},{min(hi, 1):.2f}) {int(m.sum()):>6} "
              f"{accuracy_score(y[m], pred[m]):>9.4f} {v3a:>9}")

    if base is not None:
        print("\nDIAGNÓSTICO vs V3 (oracle só diagnóstico)")
        print(f"Disagreement          : {np.mean(pred != base):.4%}")
        print(f"Novo certo / V3 errado: {int(((pred == y) & (base != y)).sum())}")
        print(f"V3 certo / Novo errado: {int(((pred != y) & (base == y)).sum())}")
        for c, name in ID2LABEL.items():
            s = y == c
            g = int(((pred == y) & (base != y) & s).sum())
            l = int(((pred != y) & (base == y) & s).sum())
            print(f"  true {name:5s}: novo corrige {g:4d} | V3 corrige {l:4d} | saldo {g - l:+d}")
        pd.DataFrame([{
            "new_accuracy": accuracy_score(y, pred), "v3_accuracy": accuracy_score(y, base),
            "disagreement": float(np.mean(pred != base)),
            "new_right_v3_wrong": int(((pred == y) & (base != y)).sum()),
            "v3_right_new_wrong": int(((pred != y) & (base == y)).sum()),
        }]).to_csv(RESULTS / "knn_tfidf_vs_v3.csv", index=False)


if __name__ == "__main__":
    main()