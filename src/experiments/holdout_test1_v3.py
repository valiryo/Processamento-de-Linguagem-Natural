"""
Mede o V3 congelado (e cada componente) no test1 rotulado.

test1-rotulado-para-teste-estatistico.xlsx tem 900 linhas (300/300/300) e é o
único holdout independente que o projeto possui: nenhum modelo, regra ou
hiperparâmetro foi escolhido com ele. Serve para duas coisas:

    1. dar o número real de generalização do V3 (o 47,57% OOF foi selecionado
       sobre o próprio OOF e tende a ser otimista);
    2. servir de referência para testar qualquer candidato novo.

NÃO use este script para escolher entre variantes. Cada olhada no test1 para
decidir consome o holdout.

Não treina nada; só lê results/*_final.csv e o xlsx.

Uso:
    python src/experiments/holdout_test1_v3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.stacking import (  # noqa: E402
    LABEL2ID,
    read_csv_bom,
    summarize,
)

RESULTS_DIR = PROJECT_ROOT / "results"
TEST1 = PROJECT_ROOT / "data" / "raw" / "test1-rotulado-para-teste-estatistico.xlsx"

FINAL_FILES = {
    "baseline_lr": "baseline_final.csv",
    "linearsvc": "tfidf_word_char_linearsvc_final.csv",
    "bert512": "bertimbau_512_final.csv",
    "soft256": "bertimbau_duplicate_soft_labels_256_final.csv",
    "ordinal256": "bertimbau_ordinal_256_final.csv",
    "norberto": "norberto_base_1024_final.csv",
    "ensemble_v2": "ensemble_majority_5_v2_final.csv",
    "ensemble_v3": "ensemble_majority_5_v3_norberto_final.csv",
}

N_BOOT = 2000
SEED = 42


def pred_column(df: pd.DataFrame) -> np.ndarray:
    for col in ("y_pred_id", "y_pred_id_v3"):
        if col in df.columns:
            return df[col].astype(int).to_numpy()
    return df["y_pred"].map(LABEL2ID).astype(int).to_numpy()


def main() -> None:
    test = pd.read_excel(TEST1)
    y = test["clarity"].astype(str).str.strip().map(LABEL2ID)
    if y.isna().any():
        raise ValueError("test1 possui rótulos fora de c1/c234/c5.")
    y = y.astype(int).to_numpy()
    print(f"test1: {len(y)} linhas, distribuição {np.bincount(y).tolist()}")

    rng = np.random.default_rng(SEED)
    boot_idx = rng.integers(0, len(y), size=(N_BOOT, len(y)))

    rows = []
    for name, filename in FINAL_FILES.items():
        df = read_csv_bom(RESULTS_DIR / filename).sort_values("row_id").reset_index(drop=True)
        if len(df) != len(y) or not np.array_equal(df["row_id"].to_numpy(), np.arange(len(y))):
            raise ValueError(f"{filename}: row_id/tamanho não bate com o test1.")
        pred = pred_column(df)
        m = summarize(y, pred)
        boots = (pred[boot_idx] == y[boot_idx]).mean(axis=1)
        rows.append(
            {
                "model": name,
                **m,
                "acc_ci_low": float(np.percentile(boots, 2.5)),
                "acc_ci_high": float(np.percentile(boots, 97.5)),
            }
        )

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.round(4).to_string(index=False))

    v3_acc = out.set_index("model").loc["ensemble_v3", "accuracy"]
    if v3_acc < 0.38:
        print(
            "\nATENÇÃO: accuracy do V3 próxima do acaso (33%). Provável desalinhamento "
            "entre as previsões salvas e este test1 (textos/ordem diferentes). "
            "Não use estes números até confirmar."
        )
    else:
        n = len(y)
        se = np.sqrt(v3_acc * (1 - v3_acc) / n)
        print(
            f"\nV3 no test1: {v3_acc:.4f} (erro-padrão ≈ {se:.4f}). Com n={n}, só diferenças "
            f"acima de ~3 p.p. são distinguíveis do acaso em comparação independente; "
            f"para candidatos use o teste pareado (McNemar) de stacking_apply_test.py."
        )

    out.to_csv(RESULTS_DIR / "holdout_test1_summary.csv", index=False)


if __name__ == "__main__":
    main()
