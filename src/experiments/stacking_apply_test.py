"""
Aplica o meta-modelo congelado (results/stacking_lr_meta.joblib, gerado por
stacking_oof_cv.py) às probabilidades dos componentes treinados em 100% de
train.xlsx.

    --split test1 : mede stacking vs V3 com os rótulos de test1 (comparação
                    pareada). Rode UMA vez, e só se o veredito do OOF foi ADOTAR.
    --split test2 : gera as previsões para o test2 (sem rótulos).

Pré-requisitos:
    python src/experiments/prob_components_cv.py      (baseline/svc para test1 e test2)
    results/*_final.csv                               (transformers, test1: já existem)
    results/*_final_test2.csv                         (transformers, test2: ver nota)

Nota test2: os scripts *_final.py atuais têm test1.xlsx e 900 linhas fixos no
código e não deixaram checkpoints. Para prever test2 é preciso retreinar os
transformers uma vez (ver plano) e salvar as saídas com o sufixo _test2.

Não recalibra nada com o teste.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stacking import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    PROB_COLS,
    build_features,
    paired_vs_reference,
    read_component,
    read_csv_bom,
    summarize,
)

RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TEST_XLSX = {
    "test1": DATA_DIR / "test1-rotulado-para-teste-estatistico.xlsx",
    "test2": DATA_DIR / "test2.xlsx",
}

STEMS = {
    "test1": {
        "bert512": "bertimbau_512_final.csv",
        "soft256": "bertimbau_duplicate_soft_labels_256_final.csv",
        "ordinal256": "bertimbau_ordinal_256_final.csv",
        "norberto": "norberto_base_1024_final.csv",
        "baseline": "baseline_probs_test1.csv",
        "svc": "svc_scores_test1.csv",
    },
    "test2": {
        "bert512": "bertimbau_512_final_test2.csv",
        "soft256": "bertimbau_duplicate_soft_labels_256_final_test2.csv",
        "ordinal256": "bertimbau_ordinal_256_final_test2.csv",
        "norberto": "norberto_base_1024_final_test2.csv",
        "baseline": "baseline_probs_test2.csv",
        "svc": "svc_scores_test2.csv",
    },
}

V3_TEST1 = "ensemble_majority_5_v3_norberto_final.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["test1", "test2"], required=True)
    args = parser.parse_args()

    bundle = joblib.load(RESULTS_DIR / "stacking_lr_meta.joblib")
    model, order, kinds = bundle["model"], bundle["order"], bundle["kinds"]

    test = pd.read_excel(TEST_XLSX[args.split])
    n = len(test)
    row_ids = np.arange(n)

    comps = {}
    for name in order:
        path = RESULTS_DIR / STEMS[args.split][name]
        comps[name] = read_component([path], kinds[name])
        if len(comps[name]) != n:
            raise ValueError(f"{path.name}: {len(comps[name])} linhas, {args.split} tem {n}.")

    X = build_features(comps, kinds, order, row_ids)
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)

    out = pd.DataFrame(
        {
            "row_id": row_ids,
            "y_pred_id": pred,
            "y_pred": [ID2LABEL[i] for i in pred],
            PROB_COLS[0]: proba[:, 0],
            PROB_COLS[1]: proba[:, 1],
            PROB_COLS[2]: proba[:, 2],
        }
    )
    out_path = RESULTS_DIR / f"stacking_lr_{args.split}.csv"
    out.to_csv(out_path, index=False)
    print(f"{args.split}: {n} previsões -> {out_path.name}; distribuição {np.bincount(pred, minlength=3).tolist()}")

    if args.split == "test1":
        y = test["clarity"].astype(str).str.strip().map(LABEL2ID).astype(int).to_numpy()
        v3 = read_csv_bom(RESULTS_DIR / V3_TEST1).sort_values("row_id")
        pred_v3 = v3["y_pred_id"].astype(int).to_numpy()
        groups = test["resp_text"].fillna("").astype(str).to_numpy()

        print("\nstack_lr:", {k: round(v, 4) for k, v in summarize(y, pred).items()})
        print("v3      :", {k: round(v, 4) for k, v in summarize(y, pred_v3).items()})
        print("pareado :", {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in paired_vs_reference(y, pred, pred_v3, groups=groups).items()})


if __name__ == "__main__":
    main()
