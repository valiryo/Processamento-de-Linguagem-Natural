"""
Ensemble OOF robusto para os arquivos reais do projeto.

- Não treina nenhum modelo.
- Usa apenas OOFs já gerados.
- Aceita y_true_id numérico OU y_true em {c1,c234,c5}.
- Procura os nomes atuais dos arquivos do projeto.
- Modelos sem prob_c1/prob_c234/prob_c5 são ignorados com aviso,
  em vez de derrubar a execução.
- Inclui baseline automaticamente se ele possuir probabilidades.
- Produz:
    1) resultado individual;
    2) média uniforme de todos os modelos compatíveis;
    3) busca rápida de todos os subconjuntos com média uniforme;
    4) busca exploratória de pesos por cross-fold.

IMPORTANTE:
A média uniforme possui avaliação OOF direta sem hiperparâmetro aprendido.
A busca de subconjuntos/pesos usa os OOFs para seleção e deve ser tratada
como seleção de hiperparâmetro via CV, não como um "novo teste independente".
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MODEL_FILES = {
    # Arquivos que aparecem na pasta atual do projeto.
    "baseline": RESULTS / "baseline_oof.csv",
    "bert512": RESULTS / "bertimbau_512_oof.csv",
    "soft256": RESULTS / "bertimbau_duplicate_soft_labels_256_oof.csv",
    "headtail256": RESULTS / "bertimbau_head_tail_256_oof.csv",

    # Nome ATUAL após o rename mostrado pelo usuário.
    "ordinal256": RESULTS / "bertimbau_ordinal_256_oof.csv",

    # Passará a ser incluído automaticamente assim que o experimento terminar.
    "smooth256": RESULTS / "bertimbau_label_smoothing_256_oof.csv",
}

PROB_COLS = ["prob_c1", "prob_c234", "prob_c5"]

LABELS = ["c1", "c234", "c5"]
LABEL2ID = {
    "c1": 0,
    "c234": 1,
    "c5": 2,
}

EXPECTED_N = 20_092
EXPECTED_FOLDS = {0, 1, 2, 3, 4}

INDIVIDUAL_OUT = RESULTS / "oof_ensemble_individual_models.csv"
SUBSETS_OUT = RESULTS / "oof_ensemble_equal_weight_subsets.csv"
UNIFORM_OOF_OUT = RESULTS / "oof_ensemble_uniform_oof.csv"
WEIGHTED_FOLDS_OUT = RESULTS / "oof_ensemble_weighted_crossfold.csv"
WEIGHTED_OOF_OUT = RESULTS / "oof_ensemble_weighted_crossfold_oof.csv"

WEIGHT_STEP = 0.10


def labels_to_ids(series: pd.Series, source_name: str) -> np.ndarray:
    """
    Aceita:
    - inteiros 0/1/2;
    - floats equivalentes a 0/1/2;
    - strings c1/c234/c5.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        arr = numeric.astype(int).to_numpy()
        if set(np.unique(arr)).issubset({0, 1, 2}):
            return arr

    text = series.astype(str).str.strip()
    mapped = text.map(LABEL2ID)

    if mapped.isna().any():
        bad = sorted(text[mapped.isna()].unique().tolist())[:10]
        raise ValueError(
            f"{source_name}: não consegui interpretar os labels. "
            f"Valores problemáticos: {bad}"
        )

    return mapped.astype(int).to_numpy()


def choose_true_label_column(df: pd.DataFrame, source_name: str) -> np.ndarray:
    """
    Prioriza explicitamente y_true_id quando existe, porque vários scripts
    antigos guardam y_true como string e y_true_id como inteiro.
    """
    if "y_true_id" in df.columns:
        return labels_to_ids(df["y_true_id"], source_name)

    if "y_true" in df.columns:
        return labels_to_ids(df["y_true"], source_name)

    raise ValueError(
        f"{source_name}: não existe y_true_id nem y_true."
    )


def load_model(name: str, path: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not path.exists():
        return None, f"arquivo não encontrado: {path.name}"

    df = pd.read_csv(path)

    mandatory = {"row_id", "fold"}
    missing_mandatory = mandatory - set(df.columns)
    if missing_mandatory:
        return None, (
            f"faltam colunas obrigatórias: "
            f"{sorted(missing_mandatory)}"
        )

    missing_probs = set(PROB_COLS) - set(df.columns)
    if missing_probs:
        return None, (
            "não possui probabilidades de classe "
            f"{sorted(missing_probs)}; será ignorado no probability ensemble"
        )

    try:
        y_true = choose_true_label_column(df, name)
    except ValueError as exc:
        return None, str(exc)

    if df["row_id"].duplicated().any():
        return None, "row_id duplicado"

    fold = pd.to_numeric(df["fold"], errors="coerce")
    if fold.isna().any():
        return None, "fold contém valores não numéricos"

    fold = fold.astype(int).to_numpy()

    if not set(np.unique(fold)).issubset(EXPECTED_FOLDS):
        return None, f"folds inesperados: {sorted(np.unique(fold).tolist())}"

    probs = df[PROB_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)

    if not np.isfinite(probs).all():
        return None, "probabilidades contêm NaN/inf"

    if (probs < -1e-8).any():
        return None, "probabilidades negativas"

    sums = probs.sum(axis=1)

    if not np.allclose(sums, 1.0, atol=2e-3):
        return None, (
            "prob_c1/prob_c234/prob_c5 não somam aproximadamente 1 "
            f"(min={sums.min():.6f}, max={sums.max():.6f})"
        )

    out = pd.DataFrame(
        {
            "row_id": pd.to_numeric(df["row_id"], errors="raise").astype(int),
            "fold": fold,
            "y_true": y_true,
            f"{name}__prob_c1": probs[:, 0],
            f"{name}__prob_c234": probs[:, 1],
            f"{name}__prob_c5": probs[:, 2],
        }
    ).sort_values("row_id").reset_index(drop=True)

    return out, None


def align_models(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = list(models)

    first = names[0]
    base = models[first].copy()

    base = base.rename(
        columns={
            "fold": f"{first}__fold",
            "y_true": f"{first}__y_true",
        }
    )

    reference_ids = set(base["row_id"])

    for name in names[1:]:
        current = models[name]

        if set(current["row_id"]) != reference_ids:
            missing_here = len(reference_ids - set(current["row_id"]))
            extra_here = len(set(current["row_id"]) - reference_ids)
            raise ValueError(
                f"{name}: conjunto de row_id diferente do modelo {first}. "
                f"faltando={missing_here}, extras={extra_here}"
            )

        current = current.rename(
            columns={
                "fold": f"{name}__fold",
                "y_true": f"{name}__y_true",
            }
        )

        base = base.merge(
            current,
            on="row_id",
            how="inner",
            validate="one_to_one",
        )

    ref_fold = base[f"{first}__fold"].to_numpy(int)
    ref_y = base[f"{first}__y_true"].to_numpy(int)

    for name in names[1:]:
        other_fold = base[f"{name}__fold"].to_numpy(int)
        other_y = base[f"{name}__y_true"].to_numpy(int)

        if not np.array_equal(ref_fold, other_fold):
            raise ValueError(
                f"{name}: atribuição de folds não coincide com {first}."
            )

        if not np.array_equal(ref_y, other_y):
            raise ValueError(
                f"{name}: y_true não coincide com {first}."
            )

    drop_cols = []
    for name in names:
        drop_cols.extend(
            [
                f"{name}__fold",
                f"{name}__y_true",
            ]
        )

    base["fold"] = ref_fold
    base["y_true"] = ref_y

    base = (
        base.drop(columns=drop_cols)
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    return base


def probs_for_model(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[
        [
            f"{name}__prob_c1",
            f"{name}__prob_c234",
            f"{name}__prob_c5",
        ]
    ].to_numpy(float)


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, np.ndarray]:
    pred = probs.argmax(axis=1)

    return (
        float(accuracy_score(y_true, pred)),
        float(
            f1_score(
                y_true,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
        pred,
    )


def mean_probs(
    aligned: pd.DataFrame,
    model_names: list[str] | tuple[str, ...],
) -> np.ndarray:
    tensors = [
        probs_for_model(aligned, name)
        for name in model_names
    ]

    return np.mean(
        np.stack(tensors, axis=0),
        axis=0,
    )


def simplex_weights(n_models: int, step: float) -> np.ndarray:
    units = int(round(1.0 / step))

    if not np.isclose(units * step, 1.0):
        raise ValueError("WEIGHT_STEP precisa dividir 1.0.")

    rows: list[list[int]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append(prefix + [remaining])
            return

        for x in range(remaining + 1):
            rec(
                prefix + [x],
                remaining - x,
                slots - 1,
            )

    rec([], units, n_models)

    return np.asarray(rows, dtype=float) / units


def crossfold_weight_search(
    aligned: pd.DataFrame,
    names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Busca exploratória:
    para cada fold, escolhe pesos usando os outros 4 folds OOF e aplica
    no fold deixado de fora.

    É útil para verificar complementaridade, mas não substitui um nested CV
    completo com retreinamento dos modelos-base.
    """
    y = aligned["y_true"].to_numpy(int)
    folds = aligned["fold"].to_numpy(int)

    tensor = np.stack(
        [probs_for_model(aligned, name) for name in names],
        axis=1,
    )
    # [N, M, 3]

    grid = simplex_weights(len(names), WEIGHT_STEP)

    final_probs = np.full((len(aligned), 3), np.nan)
    fold_rows = []

    for outer_fold in sorted(np.unique(folds)):
        train_mask = folds != outer_fold
        val_mask = folds == outer_fold

        best = None

        for weights in grid:
            # Exclui soluções degeneradas de um único modelo.
            if np.count_nonzero(weights > 0) < 2:
                continue

            train_probs = np.einsum(
                "nmc,m->nc",
                tensor[train_mask],
                weights,
                optimize=True,
            )

            train_pred = train_probs.argmax(axis=1)

            acc = accuracy_score(
                y[train_mask],
                train_pred,
            )

            mf1 = f1_score(
                y[train_mask],
                train_pred,
                average="macro",
                zero_division=0,
            )

            score = (float(acc), float(mf1))

            if best is None or score > best[0]:
                best = (
                    score,
                    weights.copy(),
                )

        if best is None:
            raise RuntimeError(
                "Nenhum conjunto de pesos válido encontrado."
            )

        (inner_acc, inner_f1), weights = best

        val_probs = np.einsum(
            "nmc,m->nc",
            tensor[val_mask],
            weights,
            optimize=True,
        )

        val_acc, val_f1, _ = evaluate(
            y[val_mask],
            val_probs,
        )

        final_probs[val_mask] = val_probs

        row = {
            "fold": int(outer_fold),
            "inner_accuracy": inner_acc,
            "inner_macro_f1": inner_f1,
            "outer_accuracy": val_acc,
            "outer_macro_f1": val_f1,
        }

        for name, weight in zip(names, weights):
            row[f"weight_{name}"] = float(weight)

        fold_rows.append(row)

    final_acc, final_f1, final_pred = evaluate(
        y,
        final_probs,
    )

    fold_df = pd.DataFrame(fold_rows)

    oof_df = pd.DataFrame(
        {
            "row_id": aligned["row_id"].astype(int),
            "fold": aligned["fold"].astype(int),
            "y_true": y,
            "y_pred": final_pred,
            "prob_c1": final_probs[:, 0],
            "prob_c234": final_probs[:, 1],
            "prob_c5": final_probs[:, 2],
        }
    )

    print("\nBusca exploratória de pesos por cross-fold:")
    print(
        f"  Accuracy={final_acc:.6f} | "
        f"Macro-F1={final_f1:.6f}"
    )

    return fold_df, oof_df


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("=" * 79)
    print("ENSEMBLE OOF — VERIFICAÇÃO E COMBINAÇÃO")
    print("=" * 79)

    models: dict[str, pd.DataFrame] = {}

    for name, path in MODEL_FILES.items():
        model_df, reason = load_model(name, path)

        if model_df is None:
            print(
                f"[skip] {name:12s} | {path.name} | {reason}"
            )
            continue

        models[name] = model_df

        print(
            f"[ok]   {name:12s} | {path.name} | "
            f"{len(model_df)} linhas"
        )

    if len(models) < 2:
        raise RuntimeError(
            "Menos de dois OOFs compatíveis com probability ensemble."
        )

    aligned = align_models(models)
    names = list(models)

    print("\nModelos compatíveis:")
    for name in names:
        print(f"  - {name}")

    print(f"\nLinhas alinhadas: {len(aligned)}")

    if len(aligned) != EXPECTED_N:
        print(
            f"[aviso] Esperávamos {EXPECTED_N} linhas, "
            f"mas foram alinhadas {len(aligned)}."
        )

    y = aligned["y_true"].to_numpy(int)

    # ==============================================================
    # 1. INDIVIDUAIS
    # ==============================================================

    individual_rows = []

    print("\nResultados individuais recalculados a partir dos OOFs:")

    for name in names:
        probs = probs_for_model(aligned, name)

        acc, mf1, _ = evaluate(
            y,
            probs,
        )

        individual_rows.append(
            {
                "model": name,
                "accuracy": acc,
                "macro_f1": mf1,
            }
        )

        print(
            f"  {name:12s} | "
            f"Accuracy={acc:.6f} | "
            f"Macro-F1={mf1:.6f}"
        )

    individual_df = pd.DataFrame(individual_rows).sort_values(
        ["accuracy", "macro_f1"],
        ascending=False,
    )

    individual_df.to_csv(
        INDIVIDUAL_OUT,
        index=False,
    )

    # ==============================================================
    # 2. MÉDIA UNIFORME DE TODOS
    # ==============================================================

    uniform_probs = mean_probs(
        aligned,
        names,
    )

    uniform_acc, uniform_f1, uniform_pred = evaluate(
        y,
        uniform_probs,
    )

    print("\nMédia uniforme de TODOS os OOFs compatíveis:")
    print(
        f"  Accuracy={uniform_acc:.6f} | "
        f"Macro-F1={uniform_f1:.6f}"
    )

    uniform_oof = pd.DataFrame(
        {
            "row_id": aligned["row_id"].astype(int),
            "fold": aligned["fold"].astype(int),
            "y_true": y,
            "y_pred": uniform_pred,
            "prob_c1": uniform_probs[:, 0],
            "prob_c234": uniform_probs[:, 1],
            "prob_c5": uniform_probs[:, 2],
        }
    )

    uniform_oof.to_csv(
        UNIFORM_OOF_OUT,
        index=False,
    )

    # ==============================================================
    # 3. TODOS OS SUBCONJUNTOS COM PESO IGUAL
    # ==============================================================

    subset_rows = []

    for size in range(2, len(names) + 1):
        for subset in combinations(names, size):
            probs = mean_probs(
                aligned,
                subset,
            )

            acc, mf1, _ = evaluate(
                y,
                probs,
            )

            subset_rows.append(
                {
                    "models": "+".join(subset),
                    "n_models": size,
                    "accuracy": acc,
                    "macro_f1": mf1,
                }
            )

    subset_df = pd.DataFrame(subset_rows).sort_values(
        ["accuracy", "macro_f1"],
        ascending=False,
    )

    subset_df.to_csv(
        SUBSETS_OUT,
        index=False,
    )

    print("\nMelhores médias uniformes entre subconjuntos:")
    print(
        subset_df.head(10).to_string(
            index=False
        )
    )

    # ==============================================================
    # 4. PESOS POR CROSS-FOLD — EXPLORATÓRIO
    # ==============================================================

    fold_df, weighted_oof = crossfold_weight_search(
        aligned,
        names,
    )

    fold_df.to_csv(
        WEIGHTED_FOLDS_OUT,
        index=False,
    )

    weighted_oof.to_csv(
        WEIGHTED_OOF_OUT,
        index=False,
    )

    weighted_acc = accuracy_score(
        weighted_oof["y_true"],
        weighted_oof["y_pred"],
    )
    weighted_f1 = f1_score(
        weighted_oof["y_true"],
        weighted_oof["y_pred"],
        average="macro",
        zero_division=0,
    )

    print("\n" + "=" * 79)
    print("RESUMO")
    print("=" * 79)

    best_individual = individual_df.iloc[0]
    best_subset = subset_df.iloc[0]

    print(
        f"Melhor individual: {best_individual['model']} | "
        f"Acc={best_individual['accuracy']:.6f} | "
        f"F1={best_individual['macro_f1']:.6f}"
    )

    print(
        f"Média uniforme todos: "
        f"Acc={uniform_acc:.6f} | F1={uniform_f1:.6f}"
    )

    print(
        f"Melhor subconjunto uniforme: {best_subset['models']} | "
        f"Acc={best_subset['accuracy']:.6f} | "
        f"F1={best_subset['macro_f1']:.6f}"
    )

    print(
        f"Pesos cross-fold (exploratório): "
        f"Acc={weighted_acc:.6f} | F1={weighted_f1:.6f}"
    )

    print("\nClassification report do melhor ensemble de pesos cross-fold:")
    print(
        classification_report(
            weighted_oof["y_true"],
            weighted_oof["y_pred"],
            labels=[0, 1, 2],
            target_names=LABELS,
            digits=6,
            zero_division=0,
        )
    )

    print("Matriz de confusão:")
    print(
        confusion_matrix(
            weighted_oof["y_true"],
            weighted_oof["y_pred"],
            labels=[0, 1, 2],
        )
    )

    print("\nArquivos gravados:")
    for path in [
        INDIVIDUAL_OUT,
        SUBSETS_OUT,
        UNIFORM_OOF_OUT,
        WEIGHTED_FOLDS_OUT,
        WEIGHTED_OOF_OUT,
    ]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
