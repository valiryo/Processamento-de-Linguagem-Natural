"""
Ensemble OOF fixo e reproduzível:
BERTimbau 512 + duplicate-aware soft labels 256 + TF-IDF word/char LinearSVC
+ baseline TF-IDF/LogisticRegression + BERTimbau ordinal corrigido.

Não faz busca de combinações nem otimiza pesos.
A regra é maioria simples entre 5 classificadores (sem empate possível).

Arquivos esperados em results/:
- bertimbau_512_oof.csv
- bertimbau_duplicate_soft_labels_256_oof.csv
- tfidf_word_char_linearsvc_oof.csv
- baseline_oof.csv
- bertimbau_ordinal_256_oof.csv

Saídas:
- ensemble_majority_5_cv.csv
- ensemble_majority_5_oof.csv
- ensemble_majority_5_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

BERT512_PATH = RESULTS / "bertimbau_512_oof.csv"
SOFT_PATH = RESULTS / "bertimbau_duplicate_soft_labels_256_oof.csv"
SVC_PATH = RESULTS / "tfidf_word_char_linearsvc_oof.csv"
BASELINE_PATH = RESULTS / "baseline_oof.csv"
ORDINAL_PATH = RESULTS / "bertimbau_ordinal_256_oof.csv"

CV_OUT = RESULTS / "ensemble_majority_5_cv.csv"
OOF_OUT = RESULTS / "ensemble_majority_5_oof.csv"
SUMMARY_OUT = RESULTS / "ensemble_majority_5_summary.csv"

EXPECTED_N = 20_092
EXPECTED_FOLDS = {0, 1, 2, 3, 4}

LABELS = ["c1", "c234", "c5"]
LABEL2ID = {
    "c1": 0,
    "c234": 1,
    "c5": 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def as_int_labels(series: pd.Series, source: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        values = numeric.astype(int).to_numpy()

        if set(np.unique(values)).issubset({0, 1, 2}):
            return values

    mapped = series.astype(str).str.strip().map(LABEL2ID)

    if mapped.isna().any():
        bad = (
            series[mapped.isna()]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{source}: labels não reconhecidos: {bad}"
        )

    return mapped.astype(int).to_numpy()


def validate_row_id_and_fold(
    row_id: np.ndarray,
    fold: np.ndarray,
    source: str,
) -> None:
    if len(row_id) != EXPECTED_N:
        raise ValueError(
            f"{source}: {len(row_id)} linhas; esperado {EXPECTED_N}."
        )

    if len(np.unique(row_id)) != EXPECTED_N:
        raise ValueError(
            f"{source}: row_id duplicado."
        )

    found_folds = set(np.unique(fold).tolist())

    if found_folds != EXPECTED_FOLDS:
        raise ValueError(
            f"{source}: folds encontrados={sorted(found_folds)}; "
            f"esperado={sorted(EXPECTED_FOLDS)}."
        )


def load_standard_oof(
    path: Path,
    source: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{source}: arquivo não encontrado: {path}"
        )

    df = pd.read_csv(path)

    if "row_id" not in df.columns:
        raise ValueError(
            f"{source}: coluna row_id não encontrada."
        )

    if "fold" not in df.columns:
        raise ValueError(
            f"{source}: coluna fold não encontrada."
        )

    if "y_true_id" in df.columns:
        y_true = as_int_labels(
            df["y_true_id"],
            source,
        )
    elif "y_true" in df.columns:
        y_true = as_int_labels(
            df["y_true"],
            source,
        )
    else:
        raise ValueError(
            f"{source}: y_true_id/y_true ausente."
        )

    if "y_pred_id" in df.columns:
        y_pred = as_int_labels(
            df["y_pred_id"],
            source,
        )
    elif "y_pred" in df.columns:
        y_pred = as_int_labels(
            df["y_pred"],
            source,
        )
    else:
        raise ValueError(
            f"{source}: y_pred_id/y_pred ausente."
        )

    row_id = pd.to_numeric(
        df["row_id"],
        errors="raise",
    ).astype(int).to_numpy()

    fold = pd.to_numeric(
        df["fold"],
        errors="raise",
    ).astype(int).to_numpy()

    validate_row_id_and_fold(
        row_id=row_id,
        fold=fold,
        source=source,
    )

    out = pd.DataFrame(
        {
            "row_id": row_id,
            "fold": fold,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    if "resp_text" in df.columns:
        out["resp_text"] = (
            df["resp_text"]
            .fillna("")
            .astype(str)
            .to_numpy()
        )

    return (
        out.sort_values("row_id")
        .reset_index(drop=True)
    )


def load_baseline(path: Path) -> pd.DataFrame:
    source = "baseline"

    if not path.exists():
        raise FileNotFoundError(
            f"{source}: arquivo não encontrado: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "row_index",
        "classe_real",
        "predicao",
        "fold",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{source}: colunas ausentes: {sorted(missing)}"
        )

    row_id = pd.to_numeric(
        df["row_index"],
        errors="raise",
    ).astype(int).to_numpy()

    fold = pd.to_numeric(
        df["fold"],
        errors="raise",
    ).astype(int).to_numpy()

    y_true = as_int_labels(
        df["classe_real"],
        source,
    )

    y_pred = as_int_labels(
        df["predicao"],
        source,
    )

    validate_row_id_and_fold(
        row_id=row_id,
        fold=fold,
        source=source,
    )

    return (
        pd.DataFrame(
            {
                "row_id": row_id,
                "fold": fold,
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        .sort_values("row_id")
        .reset_index(drop=True)
    )


def load_ordinal_corrected(path: Path) -> pd.DataFrame:
    """
    O CSV ordinal antigo contém y_pred da regra incorreta por argmax.

    A regra oficial corrigida do experimento é:
        classe = número de thresholds cuja probabilidade > 0.5

    As probabilidades cumulativas podem ser recuperadas das colunas
    de classe salvas:
        P(y > c1)   = 1 - prob_c1
        P(y > c234) = prob_c5

    Logo:
        pred = (1 - prob_c1 > 0.5) + (prob_c5 > 0.5)

    Essa reconstrução reproduz o resultado ordinal corrigido oficial.
    """
    source = "ordinal256"

    if not path.exists():
        raise FileNotFoundError(
            f"{source}: arquivo não encontrado: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "row_id",
        "fold",
        "prob_c1",
        "prob_c5",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{source}: colunas ausentes: {sorted(missing)}"
        )

    if "y_true_id" in df.columns:
        y_true = as_int_labels(
            df["y_true_id"],
            source,
        )
    elif "y_true" in df.columns:
        y_true = as_int_labels(
            df["y_true"],
            source,
        )
    else:
        raise ValueError(
            f"{source}: y_true_id/y_true ausente."
        )

    row_id = pd.to_numeric(
        df["row_id"],
        errors="raise",
    ).astype(int).to_numpy()

    fold = pd.to_numeric(
        df["fold"],
        errors="raise",
    ).astype(int).to_numpy()

    p_c1 = pd.to_numeric(
        df["prob_c1"],
        errors="raise",
    ).to_numpy(float)

    p_c5 = pd.to_numeric(
        df["prob_c5"],
        errors="raise",
    ).to_numpy(float)

    y_pred = (
        ((1.0 - p_c1) > 0.5).astype(int)
        + (p_c5 > 0.5).astype(int)
    )

    validate_row_id_and_fold(
        row_id=row_id,
        fold=fold,
        source=source,
    )

    out = pd.DataFrame(
        {
            "row_id": row_id,
            "fold": fold,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    return (
        out.sort_values("row_id")
        .reset_index(drop=True)
    )


def validate_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    name: str,
) -> None:
    if not np.array_equal(
        reference["row_id"].to_numpy(),
        candidate["row_id"].to_numpy(),
    ):
        raise ValueError(
            f"{name}: row_id não coincide com a referência."
        )

    if not np.array_equal(
        reference["fold"].to_numpy(),
        candidate["fold"].to_numpy(),
    ):
        raise ValueError(
            f"{name}: folds não coincidem com a referência."
        )

    if not np.array_equal(
        reference["y_true"].to_numpy(),
        candidate["y_true"].to_numpy(),
    ):
        raise ValueError(
            f"{name}: y_true não coincide com a referência."
        )


def majority_vote(predictions: np.ndarray) -> np.ndarray:
    """
    predictions: shape [N, 5]

    Com 5 modelos e 3 classes sempre existe uma classe com maioria/pluralidade
    única ou, em um caso 2-2-1, existe empate. Para preservar uma regra totalmente
    determinística, o raro empate 2-2-1 é resolvido pelo BERT512 (coluna 0).
    """
    out = np.empty(
        predictions.shape[0],
        dtype=int,
    )

    for i, row in enumerate(predictions):
        counts = np.bincount(
            row,
            minlength=3,
        )

        winners = np.flatnonzero(
            counts == counts.max()
        )

        if len(winners) == 1:
            out[i] = int(winners[0])
        else:
            # Tie-break fixo e pré-definido:
            # previsão do melhor modelo individual, BERT512.
            out[i] = int(row[0])

    return out


def metric_row(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | str]:
    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    return {
        "model": name,
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_c1": float(class_f1[0]),
        "f1_c234": float(class_f1[1]),
        "f1_c5": float(class_f1[2]),
    }


def main() -> None:
    print("=" * 79)
    print("ENSEMBLE MAJORITY-5 — OOF")
    print("=" * 79)

    bert512 = load_standard_oof(
        BERT512_PATH,
        "bert512",
    )

    soft = load_standard_oof(
        SOFT_PATH,
        "soft256",
    )

    svc = load_standard_oof(
        SVC_PATH,
        "svc_word_char",
    )

    baseline = load_baseline(
        BASELINE_PATH,
    )

    ordinal = load_ordinal_corrected(
        ORDINAL_PATH,
    )

    reference = bert512

    for name, df in [
        ("soft256", soft),
        ("svc_word_char", svc),
        ("baseline", baseline),
        ("ordinal256_corrected", ordinal),
    ]:
        validate_alignment(
            reference,
            df,
            name,
        )

    y_true = reference["y_true"].to_numpy(int)
    folds = reference["fold"].to_numpy(int)

    component_predictions = {
        "bert512": bert512["y_pred"].to_numpy(int),
        "soft256": soft["y_pred"].to_numpy(int),
        "svc_word_char": svc["y_pred"].to_numpy(int),
        "baseline": baseline["y_pred"].to_numpy(int),
        "ordinal256_corrected": ordinal["y_pred"].to_numpy(int),
    }

    prediction_matrix = np.column_stack(
        [
            component_predictions["bert512"],
            component_predictions["soft256"],
            component_predictions["svc_word_char"],
            component_predictions["baseline"],
            component_predictions["ordinal256_corrected"],
        ]
    )

    ensemble_pred = majority_vote(
        prediction_matrix
    )

    print("\nComponentes:")
    summary_rows = []

    for name, pred in component_predictions.items():
        row = metric_row(
            name,
            y_true,
            pred,
        )
        summary_rows.append(row)

        print(
            f"  {name:22s} | "
            f"Acc={row['accuracy']:.6f} | "
            f"Macro-F1={row['macro_f1']:.6f}"
        )

    ensemble_metrics = metric_row(
        "ensemble_majority_5",
        y_true,
        ensemble_pred,
    )

    summary_rows.append(
        ensemble_metrics
    )

    print("\nEnsemble:")
    print(
        f"  Accuracy = {ensemble_metrics['accuracy']:.6f}"
    )
    print(
        f"  Macro-F1 = {ensemble_metrics['macro_f1']:.6f}"
    )
    print(
        f"  F1 c1    = {ensemble_metrics['f1_c1']:.6f}"
    )
    print(
        f"  F1 c234  = {ensemble_metrics['f1_c234']:.6f}"
    )
    print(
        f"  F1 c5    = {ensemble_metrics['f1_c5']:.6f}"
    )

    print("\nMatriz de confusão:")
    print(
        confusion_matrix(
            y_true,
            ensemble_pred,
            labels=[0, 1, 2],
        )
    )

    # ============================================================
    # Resultado por fold
    # ============================================================

    cv_rows = []

    for fold in sorted(np.unique(folds)):
        mask = folds == fold

        row = {
            "fold": int(fold),
            **metric_row(
                "ensemble_majority_5",
                y_true[mask],
                ensemble_pred[mask],
            ),
            "n_val": int(mask.sum()),
        }

        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows)

    cv_df.to_csv(
        CV_OUT,
        index=False,
    )

    acc_mean = float(
        cv_df["accuracy"].mean()
    )

    acc_std = float(
        cv_df["accuracy"].std(ddof=1)
    )

    f1_mean = float(
        cv_df["macro_f1"].mean()
    )

    f1_std = float(
        cv_df["macro_f1"].std(ddof=1)
    )

    print("\nPor fold:")
    print(
        cv_df[
            [
                "fold",
                "accuracy",
                "macro_f1",
                "f1_c1",
                "f1_c234",
                "f1_c5",
            ]
        ].to_string(index=False)
    )

    print("\nCV média ± desvio:")
    print(
        f"  Accuracy = {acc_mean:.6f} ± {acc_std:.6f}"
    )
    print(
        f"  Macro-F1 = {f1_mean:.6f} ± {f1_std:.6f}"
    )

    # ============================================================
    # OOF final
    # ============================================================

    oof = pd.DataFrame(
        {
            "row_id": reference["row_id"].astype(int),
            "fold": folds,
            "y_true_id": y_true,
            "y_pred_id": ensemble_pred,
            "y_true": [
                ID2LABEL[int(x)]
                for x in y_true
            ],
            "y_pred": [
                ID2LABEL[int(x)]
                for x in ensemble_pred
            ],
            "pred_bert512": component_predictions["bert512"],
            "pred_soft256": component_predictions["soft256"],
            "pred_svc_word_char": component_predictions["svc_word_char"],
            "pred_baseline": component_predictions["baseline"],
            "pred_ordinal256_corrected": component_predictions[
                "ordinal256_corrected"
            ],
            "correct": ensemble_pred == y_true,
        }
    )

    if "resp_text" in bert512.columns:
        oof["resp_text"] = bert512["resp_text"]

    oof.to_csv(
        OOF_OUT,
        index=False,
    )

    # ============================================================
    # Summary
    # ============================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df["cv_accuracy_mean"] = np.nan
    summary_df["cv_accuracy_std"] = np.nan
    summary_df["cv_macro_f1_mean"] = np.nan
    summary_df["cv_macro_f1_std"] = np.nan

    ensemble_mask = (
        summary_df["model"]
        == "ensemble_majority_5"
    )

    summary_df.loc[
        ensemble_mask,
        "cv_accuracy_mean",
    ] = acc_mean

    summary_df.loc[
        ensemble_mask,
        "cv_accuracy_std",
    ] = acc_std

    summary_df.loc[
        ensemble_mask,
        "cv_macro_f1_mean",
    ] = f1_mean

    summary_df.loc[
        ensemble_mask,
        "cv_macro_f1_std",
    ] = f1_std

    summary_df.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    print("\nArquivos salvos:")
    print(f"  {CV_OUT}")
    print(f"  {OOF_OUT}")
    print(f"  {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
