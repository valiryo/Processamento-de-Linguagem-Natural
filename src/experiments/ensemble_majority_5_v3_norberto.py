from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


LABELS = (0, 1, 2)
ID2LABEL = {0: "c1", 1: "c234", 2: "c5"}
C234_ID = 1

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENSEMBLE_OOF = (
    PROJECT_ROOT / "results" / "ensemble_majority_5_v2_oof.csv"
)
DEFAULT_NORBERTO_OOF = (
    PROJECT_ROOT / "results" / "norberto_base_1024_oof.csv"
)

DEFAULT_OOF_OUTPUT = (
    PROJECT_ROOT / "results" / "ensemble_majority_5_v3_norberto_oof.csv"
)
DEFAULT_CV_OUTPUT = (
    PROJECT_ROOT / "results" / "ensemble_majority_5_v3_norberto_cv.csv"
)
DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT / "results" / "ensemble_majority_5_v3_norberto_summary.csv"
)

COMPONENT_COLUMNS = [
    "pred_bert512",
    "pred_soft256",
    "pred_svc_word_char",
    "pred_baseline",
    "pred_ordinal256_corrected",
]

NORBERTO_PROB_COLUMNS = ["prob_c1", "prob_c234", "prob_c5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensemble v3: preserva o ensemble majority-5 v2 e usa as "
            "probabilidades do NorBERTo somente nos empates 2-2-1 em que "
            "c234 e uma das duas classes lideres."
        )
    )

    parser.add_argument(
        "--ensemble-oof",
        type=Path,
        default=DEFAULT_ENSEMBLE_OOF,
    )
    parser.add_argument(
        "--norberto-oof",
        type=Path,
        default=DEFAULT_NORBERTO_OOF,
    )
    parser.add_argument(
        "--oof-output",
        type=Path,
        default=DEFAULT_OOF_OUTPUT,
    )
    parser.add_argument(
        "--cv-output",
        type=Path,
        default=DEFAULT_CV_OUTPUT,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )

    return parser.parse_args()


def validate_inputs(ensemble: pd.DataFrame, norberto: pd.DataFrame) -> None:
    required_ensemble = {
        "row_id",
        "fold",
        "y_true_id",
        "y_pred_id",
        *COMPONENT_COLUMNS,
    }
    required_norberto = {
        "row_id",
        "fold",
        "y_true_id",
        "y_pred_id",
        *NORBERTO_PROB_COLUMNS,
    }

    missing_ensemble = required_ensemble - set(ensemble.columns)
    missing_norberto = required_norberto - set(norberto.columns)

    if missing_ensemble:
        raise ValueError(
            "Colunas ausentes no OOF do ensemble v2: "
            + ", ".join(sorted(missing_ensemble))
        )

    if missing_norberto:
        raise ValueError(
            "Colunas ausentes no OOF do NorBERTo: "
            + ", ".join(sorted(missing_norberto))
        )

    if ensemble["row_id"].duplicated().any():
        raise ValueError("OOF do ensemble possui row_id duplicado.")

    if norberto["row_id"].duplicated().any():
        raise ValueError("OOF do NorBERTo possui row_id duplicado.")


def merge_oofs(
    ensemble: pd.DataFrame,
    norberto: pd.DataFrame,
) -> pd.DataFrame:
    nor_cols = [
        "row_id",
        "fold",
        "y_true_id",
        "y_pred_id",
        *NORBERTO_PROB_COLUMNS,
    ]

    merged = ensemble.merge(
        norberto[nor_cols],
        on="row_id",
        suffixes=("_ensemble", "_norberto"),
        validate="one_to_one",
    )

    if len(merged) != len(ensemble) or len(merged) != len(norberto):
        raise ValueError(
            "Os OOFs nao possuem exatamente o mesmo conjunto de row_id."
        )

    if not np.array_equal(
        merged["fold_ensemble"].astype(int).to_numpy(),
        merged["fold_norberto"].astype(int).to_numpy(),
    ):
        raise ValueError("Fold inconsistente entre ensemble e NorBERTo.")

    if not np.array_equal(
        merged["y_true_id_ensemble"].astype(int).to_numpy(),
        merged["y_true_id_norberto"].astype(int).to_numpy(),
    ):
        raise ValueError("y_true_id inconsistente entre ensemble e NorBERTo.")

    return merged


def apply_v3_rule(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()

    votes = out[COMPONENT_COLUMNS].astype(int).to_numpy()
    probs = out[NORBERTO_PROB_COLUMNS].to_numpy(dtype=float)

    original_pred = out["y_pred_id_ensemble"].astype(int).to_numpy()
    final_pred = original_pred.copy()

    targeted_tie = np.zeros(len(out), dtype=bool)
    changed = np.zeros(len(out), dtype=bool)

    vote_pattern = []
    leader_a = np.full(len(out), -1, dtype=int)
    leader_b = np.full(len(out), -1, dtype=int)

    for i in range(len(out)):
        counts = np.bincount(votes[i], minlength=3)
        sorted_counts = np.sort(counts)[::-1]
        vote_pattern.append("-".join(map(str, sorted_counts.tolist())))

        # A unica situacao sem vencedor unico em 5 votos e 2-2-1.
        leaders = np.flatnonzero(counts == counts.max())

        if len(leaders) != 2 or sorted_counts.tolist() != [2, 2, 1]:
            continue

        leader_a[i] = int(leaders[0])
        leader_b[i] = int(leaders[1])

        # Regra v3:
        # so intervem se c234 esta entre as duas classes empatadas.
        if C234_ID not in leaders:
            continue

        targeted_tie[i] = True

        chosen = int(leaders[np.argmax(probs[i, leaders])])
        final_pred[i] = chosen
        changed[i] = chosen != original_pred[i]

    out["vote_pattern"] = vote_pattern
    out["tie_leader_a"] = leader_a
    out["tie_leader_b"] = leader_b
    out["v3_targeted_tie"] = targeted_tie
    out["v3_changed_prediction"] = changed
    out["y_pred_id_v3"] = final_pred
    out["y_pred_v3"] = [ID2LABEL[int(v)] for v in final_pred]

    y_true = out["y_true_id_ensemble"].astype(int).to_numpy()

    out["correct_v2"] = original_pred == y_true
    out["correct_v3"] = final_pred == y_true

    return out


def metrics_for(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    per_class = f1_score(
        y_true,
        y_pred,
        labels=list(LABELS),
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_c1": float(per_class[0]),
        "f1_c234": float(per_class[1]),
        "f1_c5": float(per_class[2]),
    }


def build_cv(out: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for fold in sorted(out["fold_ensemble"].astype(int).unique()):
        subset = out[out["fold_ensemble"].astype(int) == fold]

        y_true = subset["y_true_id_ensemble"].astype(int).to_numpy()
        pred_v2 = subset["y_pred_id_ensemble"].astype(int).to_numpy()
        pred_v3 = subset["y_pred_id_v3"].astype(int).to_numpy()

        v2 = metrics_for(y_true, pred_v2)
        v3 = metrics_for(y_true, pred_v3)

        targeted = subset["v3_targeted_tie"].astype(bool).to_numpy()
        changed = subset["v3_changed_prediction"].astype(bool).to_numpy()

        fixes = int(
            np.sum(
                changed
                & (pred_v3 == y_true)
                & (pred_v2 != y_true)
            )
        )
        breaks = int(
            np.sum(
                changed
                & (pred_v3 != y_true)
                & (pred_v2 == y_true)
            )
        )

        rows.append(
            {
                "fold": fold,
                "n": len(subset),
                "targeted_ties": int(targeted.sum()),
                "changed_predictions": int(changed.sum()),
                "fixes": fixes,
                "breaks": breaks,
                "net_correct": fixes - breaks,
                "v2_accuracy": v2["accuracy"],
                "v3_accuracy": v3["accuracy"],
                "accuracy_delta": v3["accuracy"] - v2["accuracy"],
                "v2_macro_f1": v2["macro_f1"],
                "v3_macro_f1": v3["macro_f1"],
                "macro_f1_delta": v3["macro_f1"] - v2["macro_f1"],
                "v3_f1_c1": v3["f1_c1"],
                "v3_f1_c234": v3["f1_c234"],
                "v3_f1_c5": v3["f1_c5"],
            }
        )

    return pd.DataFrame(rows)


def build_summary(out: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    y_true = out["y_true_id_ensemble"].astype(int).to_numpy()
    pred_v2 = out["y_pred_id_ensemble"].astype(int).to_numpy()
    pred_v3 = out["y_pred_id_v3"].astype(int).to_numpy()

    v2 = metrics_for(y_true, pred_v2)
    v3 = metrics_for(y_true, pred_v3)

    targeted = out["v3_targeted_tie"].astype(bool).to_numpy()
    changed = out["v3_changed_prediction"].astype(bool).to_numpy()

    fixes = int(
        np.sum(
            changed
            & (pred_v3 == y_true)
            & (pred_v2 != y_true)
        )
    )
    breaks = int(
        np.sum(
            changed
            & (pred_v3 != y_true)
            & (pred_v2 == y_true)
        )
    )
    changed_both_wrong = int(
        np.sum(
            changed
            & (pred_v3 != y_true)
            & (pred_v2 != y_true)
        )
    )

    cm = confusion_matrix(
        y_true,
        pred_v3,
        labels=list(LABELS),
    )

    row = {
        "rule_name": "majority5_v2_plus_norberto_c234_tie_probability",
        "n": len(out),
        "targeted_ties": int(targeted.sum()),
        "changed_predictions": int(changed.sum()),
        "fixes": fixes,
        "breaks": breaks,
        "changed_both_wrong": changed_both_wrong,
        "net_correct": fixes - breaks,

        "v2_accuracy": v2["accuracy"],
        "v3_accuracy": v3["accuracy"],
        "accuracy_delta": v3["accuracy"] - v2["accuracy"],

        "v2_macro_f1": v2["macro_f1"],
        "v3_macro_f1": v3["macro_f1"],
        "macro_f1_delta": v3["macro_f1"] - v2["macro_f1"],

        "v3_f1_c1": v3["f1_c1"],
        "v3_f1_c234": v3["f1_c234"],
        "v3_f1_c5": v3["f1_c5"],

        "cv_accuracy_mean": cv["v3_accuracy"].mean(),
        "cv_accuracy_std": cv["v3_accuracy"].std(ddof=0),
        "cv_macro_f1_mean": cv["v3_macro_f1"].mean(),
        "cv_macro_f1_std": cv["v3_macro_f1"].std(ddof=0),

        "cm_c1_to_c1": int(cm[0, 0]),
        "cm_c1_to_c234": int(cm[0, 1]),
        "cm_c1_to_c5": int(cm[0, 2]),
        "cm_c234_to_c1": int(cm[1, 0]),
        "cm_c234_to_c234": int(cm[1, 1]),
        "cm_c234_to_c5": int(cm[1, 2]),
        "cm_c5_to_c1": int(cm[2, 0]),
        "cm_c5_to_c234": int(cm[2, 1]),
        "cm_c5_to_c5": int(cm[2, 2]),
    }

    return pd.DataFrame([row])


def main() -> int:
    args = parse_args()

    ensemble = pd.read_csv(args.ensemble_oof)
    norberto = pd.read_csv(args.norberto_oof)

    validate_inputs(ensemble, norberto)
    merged = merge_oofs(ensemble, norberto)
    out = apply_v3_rule(merged)

    cv = build_cv(out)
    summary = build_summary(out, cv)

    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    args.cv_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    # OOF final enxuto, preservando tambem informacao necessaria para auditoria.
    oof_columns = [
        "row_id",
        "fold_ensemble",
        "y_true_id_ensemble",
        "y_pred_id_ensemble",
        "y_pred_id_norberto",
        *COMPONENT_COLUMNS,
        *NORBERTO_PROB_COLUMNS,
        "vote_pattern",
        "tie_leader_a",
        "tie_leader_b",
        "v3_targeted_tie",
        "v3_changed_prediction",
        "y_pred_id_v3",
        "y_pred_v3",
        "correct_v2",
        "correct_v3",
    ]

    final_oof = out[oof_columns].rename(
        columns={
            "fold_ensemble": "fold",
            "y_true_id_ensemble": "y_true_id",
            "y_pred_id_ensemble": "y_pred_id_v2",
            "y_pred_id_norberto": "y_pred_id_norberto",
        }
    )

    final_oof.to_csv(
        args.oof_output,
        index=False,
        encoding="utf-8-sig",
    )
    cv.to_csv(
        args.cv_output,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    s = summary.iloc[0]

    print("=" * 78)
    print("ENSEMBLE MAJORITY-5 V3 + NORBERTO")
    print("=" * 78)
    print(
        "Regra: alterar somente empates 2-2-1 em que c234 esta entre "
        "as duas classes lideres."
    )
    print(
        "Desempate: maior probabilidade do NorBERTo entre as duas "
        "classes lideres."
    )
    print()
    print(f"Empates alvo             : {int(s['targeted_ties'])}")
    print(f"Predicoes alteradas      : {int(s['changed_predictions'])}")
    print(f"Erros corrigidos         : {int(s['fixes'])}")
    print(f"Acertos quebrados        : {int(s['breaks'])}")
    print(f"Saldo liquido de acertos : {int(s['net_correct'])}")
    print()
    print(f"V2 Accuracy              : {s['v2_accuracy']:.9f}")
    print(f"V3 Accuracy              : {s['v3_accuracy']:.9f}")
    print(f"Delta Accuracy           : {s['accuracy_delta']:+.9f}")
    print()
    print(f"V2 Macro-F1              : {s['v2_macro_f1']:.9f}")
    print(f"V3 Macro-F1              : {s['v3_macro_f1']:.9f}")
    print(f"Delta Macro-F1           : {s['macro_f1_delta']:+.9f}")
    print()
    print(f"V3 F1 c1                 : {s['v3_f1_c1']:.9f}")
    print(f"V3 F1 c234               : {s['v3_f1_c234']:.9f}")
    print(f"V3 F1 c5                 : {s['v3_f1_c5']:.9f}")
    print()
    print("Resultados por fold:")
    print(
        cv[
            [
                "fold",
                "v2_accuracy",
                "v3_accuracy",
                "accuracy_delta",
                "v3_macro_f1",
                "net_correct",
            ]
        ].to_string(index=False)
    )
    print()
    print("Arquivos salvos:")
    print(f"  {args.oof_output}")
    print(f"  {args.cv_output}")
    print(f"  {args.summary_output}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
