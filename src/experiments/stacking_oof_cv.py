"""
Combinação dos componentes ao nível de PROBABILIDADE, com avaliação aninhada
por fold (leave-one-fold-out sobre os OOFs congelados).

Hipótese: o V3 usa só votos duros (e probabilidades apenas do NorBERTo, em
empates). Um meta-modelo pequeno sobre as probabilidades de todos os
componentes pode aproveitar confiança e complementaridade que o voto descarta.

Protocolo (sem vazamento do meta-modelo):
    para cada fold k:
        - o meta-modelo (regressão logística, ~18 features) é treinado só nas
          linhas OOF dos folds != k;
        - o hiperparâmetro C é escolhido por CV interna só nesses folds;
        - o fold k é previsto com esse meta-modelo.
    Nenhum rótulo do fold k chega ao meta-modelo que o prevê.

Limitação a registrar no relatório: cada OOF do fold j foi gerado por um
modelo base treinado com os folds != j, que inclui as linhas do fold k. É o
resíduo padrão de empilhamento com OOF; é pequeno com um meta-modelo de poucos
parâmetros, mas não é zero. Por outro lado, as regras do V3 foram escolhidas
olhando este mesmo OOF, então a comparação NÃO favorece o stacking.

Critério de adoção, declarado ANTES de rodar (não alterar depois de ver):
    stack_lr é adotado se
        accuracy(stack_lr) - accuracy(V3) >= +0,005   e
        stack_lr >= V3 em pelo menos 4 dos 5 folds.

Apenas CPU; segundos de execução.

Uso:
    python src/experiments/stacking_oof_cv.py
    python src/experiments/stacking_oof_cv.py --extra bert512=results/bert512_seed7_oof.csv
    python src/experiments/stacking_oof_cv.py --ablate        # diagnóstico, não seleciona nada
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stacking import (  # noqa: E402
    ID2LABEL,
    PROB_COLS,
    build_features,
    paired_vs_reference,
    read_component,
    read_csv_bom,
    soft_vote,
    summarize,
)

RESULTS_DIR = PROJECT_ROOT / "results"

# nome -> (tipo, [arquivos OOF]); "extra" acrescenta seeds que são promediadas.
COMPONENTS: dict[str, tuple[str, list[str]]] = {
    "bert512": ("prob", ["bertimbau_512_oof.csv"]),
    "soft256": ("prob", ["bertimbau_duplicate_soft_labels_256_oof.csv"]),
    "ordinal256": ("prob", ["bertimbau_ordinal_256_oof.csv"]),
    "norberto": ("prob", ["norberto_base_1024_oof.csv"]),
    "baseline": ("prob", ["baseline_probs_oof.csv"]),
    "svc": ("score", ["svc_scores_oof.csv"]),
}

V3_OOF = "ensemble_majority_5_v3_norberto_oof.csv"

C_GRID = (0.01, 0.1, 1.0)

# Critério pré-declarado.
MIN_DELTA = 0.005
MIN_FOLDS_NOT_WORSE = 4

SEED = 42


def make_meta(C: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=5000, random_state=SEED)),
        ]
    )


def choose_C(X: np.ndarray, y: np.ndarray, folds: np.ndarray, train_mask: np.ndarray) -> float:
    """Escolhe C por leave-one-fold-out DENTRO dos folds de treino."""
    inner = sorted(set(folds[train_mask]))
    best_C, best_correct = C_GRID[0], -1
    for C in C_GRID:
        correct = 0
        for j in inner:
            tr = train_mask & (folds != j)
            va = folds == j
            model = make_meta(C).fit(X[tr], y[tr])
            correct += int((model.predict(X[va]) == y[va]).sum())
        if correct > best_correct:
            best_C, best_correct = C, correct
    return best_C


def nested_meta_oof(X: np.ndarray, y: np.ndarray, folds: np.ndarray):
    proba = np.zeros((len(y), 3))
    chosen = {}
    for k in sorted(set(folds)):
        train_mask = folds != k
        C = choose_C(X, y, folds, train_mask)
        model = make_meta(C).fit(X[train_mask], y[train_mask])
        proba[folds == k] = model.predict_proba(X[folds == k])
        chosen[int(k)] = C
    return proba, chosen


def per_fold_accuracy(y, pred, folds) -> dict[int, float]:
    return {int(k): float((pred[folds == k] == y[folds == k]).mean()) for k in sorted(set(folds))}


def parse_extra(items: list[str]) -> dict[str, list[str]]:
    extra: dict[str, list[str]] = {}
    for item in items:
        name, _, path = item.partition("=")
        if name not in COMPONENTS or not path:
            raise ValueError(f"--extra inválido: {item!r}. Use nome=caminho com nome em {list(COMPONENTS)}")
        extra.setdefault(name, []).append(path)
    return extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra", nargs="*", default=[], help="seeds adicionais: nome=caminho_oof.csv")
    parser.add_argument("--ablate", action="store_true", help="remove um componente por vez (diagnóstico)")
    args = parser.parse_args()
    extra = parse_extra(args.extra)

    # --- referência V3 e metadados -------------------------------------
    v3 = read_csv_bom(RESULTS_DIR / V3_OOF).sort_values("row_id").reset_index(drop=True)
    row_ids = v3["row_id"].to_numpy()
    y = v3["y_true_id"].to_numpy()
    folds = v3["fold"].to_numpy()
    pred_v3 = v3["y_pred_id_v3"].to_numpy()

    texts = read_csv_bom(RESULTS_DIR / "bertimbau_512_oof.csv").set_index("row_id").reindex(row_ids)["resp_text"]
    groups = texts.fillna("").astype(str).to_numpy()

    # --- componentes ----------------------------------------------------
    kinds, comps = {}, {}
    for name, (kind, files) in COMPONENTS.items():
        paths = [RESULTS_DIR / f for f in files] + [Path(p) for p in extra.get(name, [])]
        kinds[name] = kind
        comps[name] = read_component(paths, kind)
        print(f"{name:<11} {kind:<5} arquivos={len(paths)}")

    # y_true de todos os OOFs com probabilidades deve coincidir com o do V3.
    for name, (kind, files) in COMPONENTS.items():
        raw = read_csv_bom(RESULTS_DIR / files[0]).set_index("row_id")
        if "y_true_id" in raw.columns:
            assert (raw.reindex(row_ids)["y_true_id"].to_numpy() == y).all(), f"{name}: y_true diverge do V3"

    order = list(COMPONENTS)

    # --- métodos --------------------------------------------------------
    results: dict[str, np.ndarray] = {"v3": pred_v3}
    probas: dict[str, np.ndarray] = {}

    vote = soft_vote(comps, kinds, row_ids)
    results["soft_vote"] = vote.argmax(axis=1)

    X = build_features(comps, kinds, order, row_ids)
    proba, chosen = nested_meta_oof(X, y, folds)
    results["stack_lr"] = proba.argmax(axis=1)
    probas["stack_lr"] = proba
    print(f"\nC escolhido por fold externo: {chosen}")

    if args.ablate:
        for drop in order:
            keep = [n for n in order if n != drop]
            Xa = build_features(comps, kinds, keep, row_ids)
            pa, _ = nested_meta_oof(Xa, y, folds)
            results[f"stack_lr_sem_{drop}"] = pa.argmax(axis=1)

    # --- relatório ------------------------------------------------------
    rows = []
    for name, pred in results.items():
        row = {"method": name, **summarize(y, pred)}
        fold_acc = per_fold_accuracy(y, pred, folds)
        row.update({f"fold{k}": v for k, v in fold_acc.items()})
        if name != "v3":
            cmp_ = paired_vs_reference(y, pred, pred_v3, groups=groups)
            fold_v3 = per_fold_accuracy(y, pred_v3, folds)
            row.update(cmp_)
            row["folds_not_worse_than_v3"] = int(sum(fold_acc[k] >= fold_v3[k] for k in fold_acc))
        rows.append(row)

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print("\n" + summary.round(4).to_string(index=False))

    # --- veredito pré-declarado ----------------------------------------
    s = summary.set_index("method").loc["stack_lr"]
    adopt = (s["delta_accuracy"] >= MIN_DELTA) and (s["folds_not_worse_than_v3"] >= MIN_FOLDS_NOT_WORSE)
    print(
        f"\nVEREDITO stack_lr: {'ADOTAR' if adopt else 'NÃO ADOTAR'} "
        f"(delta={s['delta_accuracy']:+.4f}, mínimo {MIN_DELTA:+.4f}; "
        f"folds não piores={int(s['folds_not_worse_than_v3'])}/5, mínimo {MIN_FOLDS_NOT_WORSE}; "
        f"McNemar p={s['mcnemar_p']:.3f}; IC95% delta=[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}])"
    )

    # --- saídas ---------------------------------------------------------
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "stacking_summary.csv", index=False)

    pred = results["stack_lr"]
    oof = pd.DataFrame(
        {
            "row_id": row_ids,
            "fold": folds,
            "y_true_id": y,
            "y_pred_id": pred,
            "y_true": [ID2LABEL[i] for i in y],
            "y_pred": [ID2LABEL[i] for i in pred],
            PROB_COLS[0]: probas["stack_lr"][:, 0],
            PROB_COLS[1]: probas["stack_lr"][:, 1],
            PROB_COLS[2]: probas["stack_lr"][:, 2],
            "correct": pred == y,
            "resp_text": texts.to_numpy(),
        }
    )
    oof.to_csv(RESULTS_DIR / "stacking_lr_oof.csv", index=False)

    # Meta-modelo final (todas as linhas OOF) para aplicar ao teste.
    C_final = max(set(chosen.values()), key=list(chosen.values()).count)
    final = make_meta(C_final).fit(X, y)
    joblib.dump(
        {"model": final, "order": order, "kinds": kinds, "C": C_final},
        RESULTS_DIR / "stacking_lr_meta.joblib",
    )
    print(f"\nSalvos: stacking_summary.csv, stacking_lr_oof.csv, stacking_lr_meta.joblib (C={C_final})")


if __name__ == "__main__":
    main()
