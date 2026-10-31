"""
Gera probabilidades/scores OOF (e de teste) dos dois componentes clássicos do
ensemble V3, que até agora só tinham o rótulo previsto salvo:

    - baseline TF-IDF + LogisticRegression  -> prob_c1/prob_c234/prob_c5
    - TF-IDF word+char + LinearSVC (C=0.5)  -> score_c1/score_c234/score_c5
                                               (decision_function)

Configurações idênticas às oficiais (baseline.py e tfidf_word_char_linearsvc.py);
só muda o que é salvo. Usa exclusivamente data/splits/folds.csv.
Apenas CPU; leva minutos.

Saídas (results/):
    baseline_probs_oof.csv   svc_scores_oof.csv
    baseline_probs_<split>.csv   svc_scores_<split>.csv     (split = test1, test2)

Os arquivos de teste vêm de um ajuste em 100% de train.xlsx, sem tocar nos
rótulos de teste.

Uso (a partir da raiz do projeto):
    python src/experiments/prob_components_cv.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.folds import carregar_folds  # noqa: E402
from src.evaluation.stacking import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    PROB_COLS,
    SCORE_COLS,
    summarize,
)
from src.models.baseline import criar_baseline_oficial  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42

TEST_FILES = {
    "test1": DATA_DIR / "test1-rotulado-para-teste-estatistico.xlsx",
    "test2": DATA_DIR / "test2.xlsx",
}


def make_svc() -> Pipeline:
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
    return Pipeline(
        [
            ("tfidf", FeatureUnion([("word", word), ("char", char)])),
            (
                "clf",
                LinearSVC(
                    C=0.5,
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )


MODELS = {
    "baseline_probs": {
        "factory": criar_baseline_oficial,
        "cols": PROB_COLS,
        "output": lambda m, X: m.predict_proba(X),
    },
    "svc_scores": {
        "factory": make_svc,
        "cols": SCORE_COLS,
        "output": lambda m, X: m.decision_function(X),
    },
}


def load_train(train_path: Path, folds_path: Path) -> pd.DataFrame:
    df = pd.read_excel(train_path)
    df["resp_text"] = df["resp_text"].fillna("").astype(str)
    df["clarity"] = df["clarity"].astype(str).str.strip()
    df = carregar_folds(df, folds_path)
    df["row_id"] = np.arange(len(df), dtype=int)
    df["y"] = df["clarity"].map(LABEL2ID)
    if df["y"].isna().any():
        raise ValueError("Classes inesperadas em train.xlsx.")
    df["y"] = df["y"].astype(int)
    return df


def to_frame(df: pd.DataFrame, values: np.ndarray, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(values, columns=cols)
    out.insert(0, "row_id", df["row_id"].to_numpy())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.xlsx")
    parser.add_argument(
        "--folds-file", type=Path, default=PROJECT_ROOT / "data" / "splits" / "folds.csv"
    )
    args = parser.parse_args()

    df = load_train(args.train, args.folds_file)
    texts = df["resp_text"].tolist()
    y = df["y"].to_numpy()

    for name, spec in MODELS.items():
        print(f"\n=== {name} ===")
        oof = np.zeros((len(df), 3), dtype=float)
        t0 = time.perf_counter()

        for fold in sorted(df["fold"].unique()):
            val = (df["fold"] == fold).to_numpy()
            model = spec["factory"]()
            model.fit([t for t, m in zip(texts, ~val) if m], y[~val])
            oof[val] = spec["output"](model, [t for t, m in zip(texts, val) if m])
            acc = (oof[val].argmax(axis=1) == y[val]).mean()
            print(f"  fold {fold}: accuracy={acc:.6f}")

        pred = oof.argmax(axis=1)
        print(f"  OOF: {summarize(y, pred)}  ({time.perf_counter() - t0:.0f}s)")

        out = to_frame(df, oof, spec["cols"])
        out["fold"] = df["fold"].to_numpy()
        out["y_true_id"] = y
        out["y_pred_id"] = pred
        RESULTS_DIR.mkdir(exist_ok=True)
        out.to_csv(RESULTS_DIR / f"{name}_oof.csv", index=False)

        # Ajuste em 100% do treino -> previsões de teste (rótulos nunca usados).
        final = spec["factory"]()
        final.fit(texts, y)
        for split, path in TEST_FILES.items():
            if not path.exists():
                print(f"  [{split}] {path.name} não encontrado; pulando.")
                continue
            test = pd.read_excel(path)
            test_texts = test["resp_text"].fillna("").astype(str).tolist()
            values = spec["output"](final, test_texts)
            test_out = pd.DataFrame(values, columns=spec["cols"])
            test_out.insert(0, "row_id", np.arange(len(test), dtype=int))
            test_out["y_pred_id"] = values.argmax(axis=1)
            test_out["y_pred"] = test_out["y_pred_id"].map(ID2LABEL)
            test_out.to_csv(RESULTS_DIR / f"{name}_{split}.csv", index=False)
            print(f"  [{split}] {len(test)} linhas salvas")


if __name__ == "__main__":
    main()
