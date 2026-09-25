"""
Experimento CPU rápido:
TF-IDF de palavras + TF-IDF de caracteres + LinearSVC.

Motivação operacional:
- usa os mesmos 5 folds congelados;
- é muito mais barato que um Transformer;
- roda na CPU enquanto a GPU executa outro experimento;
- testa uma representação bastante diferente do BERT, útil também como
  candidato futuro a ensemble.

A inovação principal do trabalho continua podendo ser o BERT/ensemble.
Este experimento serve para procurar Accuracy de forma eficiente.

Uso:
    python src/experiments/tfidf_word_char_linearsvc.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT / "data" / "splits" / "folds.csv"
RESULTS = ROOT / "results"

CV_PATH = RESULTS / "tfidf_word_char_linearsvc_cv.csv"
OOF_PATH = RESULTS / "tfidf_word_char_linearsvc_oof.csv"
SUMMARY_PATH = RESULTS / "tfidf_word_char_linearsvc_summary.csv"

TEXT_COL = "resp_text"
LABEL_COL = "clarity"
LABELS = ["c1", "c234", "c5"]
LABEL2ID = {x: i for i, x in enumerate(LABELS)}

# Grid pequeno. A vetorização é feita UMA vez por fold e reutilizada nos C.
C_VALUES = [0.5, 1.0, 2.0]

SEED = 42


def load_data() -> pd.DataFrame:
    train = pd.read_excel(TRAIN_PATH).reset_index(drop=True)
    folds = pd.read_csv(FOLDS_PATH)

    if len(train) != len(folds):
        raise ValueError("train.xlsx e folds.csv têm tamanhos diferentes.")

    if "fold" not in folds.columns:
        raise ValueError("folds.csv não contém coluna fold.")

    train["row_id"] = np.arange(len(train))
    train["fold"] = folds["fold"].astype(int).to_numpy()
    train[TEXT_COL] = train[TEXT_COL].fillna("").astype(str)
    train["y"] = train[LABEL_COL].map(LABEL2ID)

    if train["y"].isna().any():
        raise ValueError("Há labels desconhecidos.")

    # Defesa contra leakage de textos repetidos.
    for fold in sorted(train["fold"].unique()):
        a = set(train.loc[train["fold"] != fold, TEXT_COL])
        b = set(train.loc[train["fold"] == fold, TEXT_COL])
        if a.intersection(b):
            raise ValueError(f"Leakage de resp_text detectado no fold {fold}.")

    return train


def make_vectorizers():
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        max_features=120_000,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )

    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=180_000,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )

    return word, char


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print("=" * 78)
    print("TF-IDF WORD + CHAR + LinearSVC")
    print("=" * 78)
    print(f"Amostras: {len(df)}")
    print(f"C grid: {C_VALUES}")

    cv_rows = []
    oof_by_c = {
        C: np.full(len(df), -1, dtype=int)
        for C in C_VALUES
    }

    for fold in sorted(df["fold"].unique()):
        train_mask = df["fold"].to_numpy() != fold
        val_mask = ~train_mask

        train_text = df.loc[train_mask, TEXT_COL].tolist()
        val_text = df.loc[val_mask, TEXT_COL].tolist()
        y_train = df.loc[train_mask, "y"].to_numpy(int)
        y_val = df.loc[val_mask, "y"].to_numpy(int)

        print(f"\nFold {fold}: vetorizando...")
        t0 = time.perf_counter()

        word, char = make_vectorizers()

        Xw_train = word.fit_transform(train_text)
        Xw_val = word.transform(val_text)

        Xc_train = char.fit_transform(train_text)
        Xc_val = char.transform(val_text)

        X_train = hstack([Xw_train, Xc_train], format="csr")
        X_val = hstack([Xw_val, Xc_val], format="csr")

        vectorize_seconds = time.perf_counter() - t0

        print(
            f"  shape={X_train.shape} | vetorização={vectorize_seconds:.1f}s"
        )

        for C in C_VALUES:
            t1 = time.perf_counter()

            clf = LinearSVC(
                C=C,
                class_weight="balanced",
                random_state=SEED,
                max_iter=5000,
            )
            clf.fit(X_train, y_train)
            pred = clf.predict(X_val)

            elapsed = time.perf_counter() - t1
            acc = accuracy_score(y_val, pred)
            mf1 = f1_score(y_val, pred, average="macro", zero_division=0)

            oof_by_c[C][val_mask] = pred

            row = {
                "fold": int(fold),
                "C": float(C),
                "accuracy": float(acc),
                "macro_f1": float(mf1),
                "fit_predict_seconds": float(elapsed),
                "vectorize_seconds": float(vectorize_seconds),
                "n_word_features": int(Xw_train.shape[1]),
                "n_char_features": int(Xc_train.shape[1]),
                "n_total_features": int(X_train.shape[1]),
            }
            cv_rows.append(row)

            print(
                f"  C={C:<3} Accuracy={acc:.6f} | "
                f"Macro-F1={mf1:.6f} | treino={elapsed:.1f}s"
            )

        # libera as matrizes grandes antes do próximo fold
        del Xw_train, Xw_val, Xc_train, Xc_val, X_train, X_val

        pd.DataFrame(cv_rows).to_csv(CV_PATH, index=False)

    cv = pd.DataFrame(cv_rows)

    summary = (
        cv.groupby("C")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
        .sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\n" + "=" * 78)
    print("RESUMO")
    print("=" * 78)
    print(summary.to_string(index=False))

    best_C = float(summary.iloc[0]["C"])
    best_pred = oof_by_c[best_C]

    if (best_pred < 0).any():
        raise RuntimeError("OOF do melhor C incompleto.")

    y = df["y"].to_numpy(int)
    oof_acc = accuracy_score(y, best_pred)
    oof_f1 = f1_score(y, best_pred, average="macro", zero_division=0)

    oof = pd.DataFrame(
        {
            "row_id": df["row_id"].astype(int),
            "fold": df["fold"].astype(int),
            "y_true": y,
            "y_pred": best_pred,
            "resp_text": df[TEXT_COL],
        }
    )
    oof.to_csv(OOF_PATH, index=False)

    print(f"\nMelhor C = {best_C}")
    print(f"OOF Accuracy = {oof_acc:.6f}")
    print(f"OOF Macro-F1 = {oof_f1:.6f}")

    print("\nReferências atuais:")
    print("  BERT 256 padrão : Accuracy 0.462922 | Macro-F1 0.458239")
    print("  BERT 512 padrão : Accuracy 0.464962 | Macro-F1 0.460948")


if __name__ == "__main__":
    main()
