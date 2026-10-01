"""
Atributos estilísticos/estruturais + SVD(TF-IDF) + HistGradientBoosting.

Hipótese: clareza depende de estrutura/extensão/jargão (sinal que embeddings
semânticos e Transformers truncados capturam mal). GBM sobre atributos
interpretáveis comete erros diferentes do V3.

Uso:
    python src/experiments/style_features_hgb.py
    python src/experiments/style_features_hgb.py \
        --e5-cache results/cache/frozen_e5_chunks_intfloat_multilingual-e5-large_c254_m6.npy

Saídas (results/):
    style_hgb_cv.csv | style_hgb_oof.csv | style_hgb_summary.csv | style_hgb_vs_v3.csv
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT / "data" / "splits" / "folds.csv"
RESULTS = ROOT / "results"
V3_OOF = RESULTS / "ensemble_majority_5_v3_norberto_oof.csv"

LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SEED = 42
EXPECTED_N = 20_092


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    df = pd.read_excel(TRAIN_PATH).reset_index(drop=True)
    folds = pd.read_csv(FOLDS_PATH)
    if len(df) != len(folds) or len(df) != EXPECTED_N:
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


# --------------------------------------------------------------------------- #
# Atributos manuais (determinísticos, sem ajuste => sem leakage)
# --------------------------------------------------------------------------- #
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")
VOWEL_RE = re.compile(r"[aeiouáéíóúâêôãõà]+")
SENT_RE = re.compile(r"[.!?;]+(?:\s|$)")
PARA_RE = re.compile(r"\s{2,}|\n+")

KEYWORDS = {
    "kw_anexo": r"\banexo",
    "kw_lei": r"\blei\b|\bleis\b|\bl\.\s?\d",
    "kw_art": r"\bart\.?\s?\d|\bartigo",
    "kw_decreto": r"\bdecreto|\bportaria|\binstrução normativa|\bresolução|\bresolucao",
    "kw_inciso": r"\binciso|\bparágrafo|§",
    "kw_processo": r"\bprocesso|\bnup\b|\bsei\b",
    "kw_nao": r"\bnão\b|\bnao\b",
    "kw_informamos": r"\binformamos|\besclarecemos|\bcomunicamos",
    "kw_conforme": r"\bconforme|\bem conformidade|\bnos termos",
    "kw_cortesia": r"\batenciosamente|\bcordialmente|\bagradecemos|\bcolocamo-nos",
    "kw_recurso": r"\brecurso|\binstância|\binstancia|\bprazo",
    "kw_encaminh": r"\bencaminh|\bredirecion|\bremet",
    "kw_site": r"\bsite\b|\bportal\b|\bsítio|\bacess[ea]r?\b|\bdispon[ií]vel",
    "kw_sigilo": r"\bsigil|\brestrit|\bclassific|\bsigiloso|\bpessoal\b",
}
KW_RE = {k: re.compile(v, re.IGNORECASE) for k, v in KEYWORDS.items()}
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
MAIL_RE = re.compile(r"\S+@\S+\.\S+")
ENUM_RE = re.compile(r"(?:^|\s)(?:\d{1,2}[.)]|[a-z][.)]|[-•])\s", re.IGNORECASE)
NUM_RE = re.compile(r"\d+")


def text_features(t: str) -> dict[str, float]:
    t = t.replace("\xa0", " ")
    words = WORD_RE.findall(t)
    n_words = max(len(words), 1)
    sents = [s for s in SENT_RE.split(t) if s.strip()]
    n_sents = max(len(sents), 1)
    paras = [p for p in PARA_RE.split(t) if p.strip()]
    n_chars = max(len(t), 1)

    wl = np.fromiter((len(w) for w in words), dtype=np.float32) if words else np.zeros(1)
    syll = sum(max(len(VOWEL_RE.findall(w.lower())), 1) for w in words)
    lower_words = [w.lower() for w in words]
    sent_len = np.array([max(len(WORD_RE.findall(s)), 1) for s in sents]) if sents else np.ones(1)

    f: dict[str, float] = {
        "n_chars_log": np.log1p(len(t)),
        "n_words_log": np.log1p(len(words)),
        "n_sents_log": np.log1p(len(sents)),
        "n_paras_log": np.log1p(len(paras)),
        "avg_word_len": float(wl.mean()),
        "std_word_len": float(wl.std()),
        "long_word_ratio": float((wl >= 9).mean()),
        "avg_sent_len": n_words / n_sents,
        "max_sent_len": float(sent_len.max()),
        "std_sent_len": float(sent_len.std()),
        "long_sent_ratio": float((sent_len >= 30).mean()),
        "words_per_para": n_words / max(len(paras), 1),
        "syll_per_word": syll / n_words,
        # Flesch adaptado ao português (Martins et al., 1996)
        "flesch_pt": 248.835 - 1.015 * (n_words / n_sents) - 84.6 * (syll / n_words),
        "ttr": len(set(lower_words)) / n_words,
        "upper_ratio": sum(c.isupper() for c in t) / n_chars,
        "digit_ratio": sum(c.isdigit() for c in t) / n_chars,
        "punct_ratio": sum((not c.isalnum()) and (not c.isspace()) for c in t) / n_chars,
        "space_run_ratio": len(re.findall(r"\s{2,}", t)) / n_words,
        "comma_per_word": t.count(",") / n_words,
        "colon_per_word": t.count(":") / n_words,
        "semicolon_per_word": t.count(";") / n_words,
        "paren_per_word": (t.count("(") + t.count(")")) / n_words,
        "quote_per_word": (t.count('"') + t.count("“") + t.count("”")) / n_words,
        "n_urls": float(len(URL_RE.findall(t))),
        "n_emails": float(len(MAIL_RE.findall(t))),
        "n_enum": float(len(ENUM_RE.findall(t))),
        "n_numbers_log": np.log1p(len(NUM_RE.findall(t))),
        "starts_prezad": float(t.strip()[:10].lower().startswith("prezad")),
        "starts_senhor": float(t.strip()[:8].lower().startswith("senhor")),
        "is_very_short": float(len(words) <= 15),
    }
    for k, rx in KW_RE.items():
        f[k] = float(len(rx.findall(t))) / n_words * 100.0  # por 100 palavras
        f[k + "_has"] = float(f[k] > 0)
    return f


def build_handcrafted(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    feats = pd.DataFrame([text_features(t) for t in df["resp_text"]])
    print(f"Atributos manuais: {feats.shape[1]} | {time.perf_counter() - t0:.0f}s")
    return feats


# --------------------------------------------------------------------------- #
# CV
# --------------------------------------------------------------------------- #
def run_cv(df: pd.DataFrame, hand: np.ndarray, e5: np.ndarray | None, args):
    y = df["y"].to_numpy(int)
    folds = df["fold"].to_numpy(int)
    texts = df["resp_text"].tolist()
    probs = np.zeros((len(df), 3))
    rows = []

    for f in sorted(np.unique(folds)):
        tr, va = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        t0 = time.perf_counter()

        # TF-IDF + SVD ajustados SÓ no treino do fold.
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.995, max_features=100_000,
            sublinear_tf=True, dtype=np.float32,
        )
        Xt_tr = tfidf.fit_transform([texts[i] for i in tr])
        Xt_va = tfidf.transform([texts[i] for i in va])
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=SEED)
        S_tr, S_va = svd.fit_transform(Xt_tr), svd.transform(Xt_va)

        parts_tr, parts_va = [hand[tr], S_tr], [hand[va], S_va]
        if e5 is not None:
            pca = PCA(n_components=64, random_state=SEED).fit(e5[tr])
            parts_tr.append(pca.transform(e5[tr]))
            parts_va.append(pca.transform(e5[va]))

        X_tr, X_va = np.hstack(parts_tr), np.hstack(parts_va)

        clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=600, max_leaf_nodes=15,
            min_samples_leaf=40, l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
            random_state=SEED,
        ).fit(X_tr, y[tr])

        p = clf.predict_proba(X_va)
        probs[va] = p
        pred = p.argmax(1)
        rows.append({
            "fold": int(f), "n_features": X_tr.shape[1], "n_iter": int(clf.n_iter_),
            "accuracy": accuracy_score(y[va], pred),
            "macro_f1": f1_score(y[va], pred, average="macro"),
            "runtime_seconds": time.perf_counter() - t0,
        })
        print(f"Fold {f}: acc={rows[-1]['accuracy']:.6f} f1={rows[-1]['macro_f1']:.6f} "
              f"iters={rows[-1]['n_iter']} ({rows[-1]['runtime_seconds']:.0f}s)")

    pred = probs.argmax(1)
    oof = pd.DataFrame({
        "row_id": df["row_id"], "fold": folds,
        "y_true_id": y, "y_pred_id": pred,
        "y_true": [ID2LABEL[v] for v in y], "y_pred": [ID2LABEL[int(v)] for v in pred],
        "prob_c1": probs[:, 0], "prob_c234": probs[:, 1], "prob_c5": probs[:, 2],
        "correct": pred == y, "resp_text": df["resp_text"],
    })
    return pd.DataFrame(rows), oof


# --------------------------------------------------------------------------- #
# Diagnóstico vs V3 (oracle apenas diagnóstico)
# --------------------------------------------------------------------------- #
def compare_with_v3(oof: pd.DataFrame) -> pd.DataFrame | None:
    if not V3_OOF.exists():
        print(f"\n[aviso] {V3_OOF.name} não encontrado; comparação pulada.")
        return None

    v3 = pd.read_csv(V3_OOF)
    m = oof.merge(v3, on="row_id", suffixes=("_new", "_v3"), validate="one_to_one")
    assert (m["y_true_id_new"] == m["y_true_id_v3"]).all(), "y_true inconsistente"

    y = m["y_true_id_new"].to_numpy(int)
    new, base = m["y_pred_id"].to_numpy(int), m["y_pred_id_v3"].to_numpy(int)

    print("\n" + "=" * 78)
    print("DIAGNÓSTICO vs ENSEMBLE V3 (oracle é só diagnóstico)")
    print("=" * 78)
    print(f"Novo acc={accuracy_score(y, new):.6f} | V3 acc={accuracy_score(y, base):.6f}")
    print(f"Disagreement          : {np.mean(new != base):.4%}")
    print(f"Novo certo / V3 errado: {int(((new == y) & (base != y)).sum())}")
    print(f"V3 certo / Novo errado: {int(((new != y) & (base == y)).sum())}")
    print(f"Oracle (diagnóstico)  : {np.mean((new == y) | (base == y)):.4%}")
    for c, name in ID2LABEL.items():
        s = y == c
        g = int(((new == y) & (base != y) & s).sum())
        l = int(((new != y) & (base == y) & s).sum())
        print(f"  true {name:5s}: novo corrige {g:4d} | V3 corrige {l:4d} | saldo {g - l:+d}")

    print("\nGo/no-go: só vale integrar ao ensemble se (a) disagreement alto, "
          "(b) saldo favorável em c234 e (c) o ganho for consistente nos 5 folds. "
          "Caso contrário, mantenha o V3 congelado.")

    return pd.DataFrame([{
        "new_accuracy": accuracy_score(y, new),
        "v3_accuracy": accuracy_score(y, base),
        "disagreement": float(np.mean(new != base)),
        "new_right_v3_wrong": int(((new == y) & (base != y)).sum()),
        "v3_right_new_wrong": int(((new != y) & (base == y)).sum()),
        "oracle_diagnostic": float(np.mean((new == y) | (base == y))),
    }])


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--svd-dim", type=int, default=128)
    p.add_argument("--e5-cache", type=Path, default=None,
                   help="Opcional: .npy de embeddings (N x D) gerado pelo script anterior.")
    args = p.parse_args()

    np.random.seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    hand = build_handcrafted(df).to_numpy(dtype=np.float32)

    e5 = None
    if args.e5_cache is not None:
        e5 = np.load(args.e5_cache)
        if e5.shape[0] != len(df):
            raise ValueError("Cache de embeddings com número de linhas incorreto.")
        print(f"Embeddings E5: {e5.shape}")

    cv, oof = run_cv(df, hand, e5, args)

    y, pred = oof["y_true_id"].to_numpy(), oof["y_pred_id"].to_numpy()
    f1s = f1_score(y, pred, labels=[0, 1, 2], average=None)
    summary = pd.DataFrame([{
        "model": "style_features+svd_tfidf+HGB" + ("+e5pca" if e5 is not None else ""),
        "svd_dim": args.svd_dim,
        "oof_accuracy": accuracy_score(y, pred),
        "oof_macro_f1": f1_score(y, pred, average="macro"),
        "f1_c1": f1s[0], "f1_c234": f1s[1], "f1_c5": f1s[2],
        "cv_accuracy_mean": cv["accuracy"].mean(),
        "cv_accuracy_std": cv["accuracy"].std(ddof=1),
    }])

    cv.to_csv(RESULTS / "style_hgb_cv.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(RESULTS / "style_hgb_oof.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "style_hgb_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"OOF Accuracy = {summary.at[0, 'oof_accuracy']:.6f} | "
          f"Macro-F1 = {summary.at[0, 'oof_macro_f1']:.6f}")
    print(f"F1 c1={f1s[0]:.4f} | c234={f1s[1]:.4f} | c5={f1s[2]:.4f}")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))

    cmp_df = compare_with_v3(oof)
    if cmp_df is not None:
        cmp_df.to_csv(RESULTS / "style_hgb_vs_v3.csv", index=False)


if __name__ == "__main__":
    main()