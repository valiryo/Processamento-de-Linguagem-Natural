from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# RAIZ DO PROJETO / IMPORT DO BASELINE HISTÓRICO
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.baseline import criar_baseline_oficial


# ============================================================
# CAMINHOS
# ============================================================


TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
TEST_PATH = ROOT_DIR / "data" / "raw" / "test1.xlsx"

RESULTS_DIR = ROOT_DIR / "results"

OUTPUT_PATH = (
    RESULTS_DIR
    / "baseline_final.csv"
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TEXT_COL = "resp_text"
LABEL_COL = "clarity"

LABELS = [
    "c1",
    "c234",
    "c5",
]

LABEL2ID = {
    "c1": 0,
    "c234": 1,
    "c5": 2,
}

EXPECTED_TRAIN_ROWS = 20_092
EXPECTED_TEST_ROWS = 900


# ============================================================
# CARREGAMENTO E VALIDAÇÃO
# ============================================================

def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("=" * 70)
    print("BASELINE FINAL — CARREGAMENTO DOS DADOS")
    print("=" * 70)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.xlsx não encontrado em:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test1.xlsx não encontrado em:\n{TEST_PATH}"
        )

    train_df = (
        pd.read_excel(TRAIN_PATH)
        .reset_index(drop=True)
    )

    test_df = (
        pd.read_excel(TEST_PATH)
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Colunas obrigatórias
    # --------------------------------------------------------

    if TEXT_COL not in train_df.columns:
        raise ValueError(
            f"Coluna '{TEXT_COL}' não encontrada em train.xlsx."
        )

    if LABEL_COL not in train_df.columns:
        raise ValueError(
            f"Coluna '{LABEL_COL}' não encontrada em train.xlsx."
        )

    if TEXT_COL not in test_df.columns:
        raise ValueError(
            f"Coluna '{TEXT_COL}' não encontrada em test1.xlsx."
        )

    # --------------------------------------------------------
    # Quantidades esperadas
    # --------------------------------------------------------

    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            "Número inesperado de linhas em train.xlsx: "
            f"{len(train_df)}. "
            f"Esperado: {EXPECTED_TRAIN_ROWS}."
        )

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            "Número inesperado de linhas em test1.xlsx: "
            f"{len(test_df)}. "
            f"Esperado: {EXPECTED_TEST_ROWS}."
        )

    # --------------------------------------------------------
    # Textos
    # --------------------------------------------------------

    train_df[TEXT_COL] = (
        train_df[TEXT_COL]
        .fillna("")
        .astype(str)
    )

    test_df[TEXT_COL] = (
        test_df[TEXT_COL]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Labels do treino
    # --------------------------------------------------------

    if train_df[LABEL_COL].isna().any():
        raise ValueError(
            "Existem labels ausentes em train.xlsx."
        )

    train_df[LABEL_COL] = (
        train_df[LABEL_COL]
        .astype(str)
        .str.strip()
    )

    labels_encontrados = set(
        train_df[LABEL_COL].unique()
    )

    labels_esperados = set(LABELS)

    if labels_encontrados != labels_esperados:
        raise ValueError(
            "Classes inesperadas em train.xlsx.\n"
            f"Encontradas: {sorted(labels_encontrados)}\n"
            f"Esperadas: {sorted(labels_esperados)}"
        )

    # --------------------------------------------------------
    # Teste não pode conter rótulos fornecidos
    # --------------------------------------------------------

    if LABEL_COL in test_df.columns:

        labels_teste = (
            test_df[LABEL_COL]
            .dropna()
        )

        if len(labels_teste) > 0:
            raise ValueError(
                "A coluna clarity de test1.xlsx contém valores "
                "não vazios. Interrompendo para evitar uso "
                "indevido de informação do teste."
            )

    print(f"Treino: {len(train_df)} linhas")
    print(f"Teste : {len(test_df)} linhas")

    print(
        "Distribuição do treino:"
    )

    print(
        train_df[LABEL_COL]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return train_df, test_df


# ============================================================
# TREINAMENTO E INFERÊNCIA
# ============================================================

def main() -> None:

    train_df, test_df = carregar_dados()

    print()
    print("=" * 70)
    print("BASELINE FINAL")
    print("=" * 70)

    print(
        "Modelo congelado:"
    )

    print(
        "TF-IDF padrão + "
        'LogisticRegression(class_weight="balanced")'
    )

    print()
    print(
        f"Treinando com 100% das {len(train_df)} amostras..."
    )

    # --------------------------------------------------------
    # IMPORTANTE:
    #
    # Esta função vem diretamente de baseline.py,
    # o script histórico do experimento.
    #
    # Não redefinimos hiperparâmetros aqui.
    # --------------------------------------------------------

    modelo = criar_baseline_oficial()

    X_train = train_df[TEXT_COL]
    y_train = train_df[LABEL_COL]

    modelo.fit(
        X_train,
        y_train,
    )

    print("Treinamento concluído.")

    # --------------------------------------------------------
    # Inferência
    # --------------------------------------------------------

    print()
    print(
        f"Gerando previsões para {len(test_df)} exemplos..."
    )

    X_test = test_df[TEXT_COL]

    y_pred = modelo.predict(
        X_test
    )

    # LogisticRegression possui predict_proba.
    probabilities = modelo.predict_proba(
        X_test
    )

    classifier = modelo.named_steps[
        "classificador"
    ]

    classes_modelo = list(
        classifier.classes_
    )

    # --------------------------------------------------------
    # Confere que as classes do modelo são exatamente
    # as esperadas antes de associar probabilidades.
    # --------------------------------------------------------

    if set(classes_modelo) != set(LABELS):
        raise RuntimeError(
            "Classes aprendidas pelo LogisticRegression "
            "não correspondem às classes esperadas.\n"
            f"Modelo: {classes_modelo}\n"
            f"Esperadas: {LABELS}"
        )

    class_to_probability_column = {
        classe: idx
        for idx, classe in enumerate(classes_modelo)
    }

    # --------------------------------------------------------
    # Converte previsão textual para IDs congelados.
    # --------------------------------------------------------

    y_pred_id = np.array(
        [
            LABEL2ID[label]
            for label in y_pred
        ],
        dtype=np.int64,
    )

    # --------------------------------------------------------
    # Arquivo intermediário
    # --------------------------------------------------------

    output = pd.DataFrame(
        {
            "row_id": np.arange(
                len(test_df),
                dtype=np.int64,
            ),

            "y_pred_id": y_pred_id,

            "y_pred": y_pred,

            "prob_c1": probabilities[
                :,
                class_to_probability_column["c1"],
            ],

            "prob_c234": probabilities[
                :,
                class_to_probability_column["c234"],
            ],

            "prob_c5": probabilities[
                :,
                class_to_probability_column["c5"],
            ],
        }
    )

    # --------------------------------------------------------
    # Validações antes de salvar
    # --------------------------------------------------------

    if len(output) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Número incorreto de previsões: "
            f"{len(output)}."
        )

    if output["row_id"].duplicated().any():
        raise RuntimeError(
            "Há row_id duplicado no resultado."
        )

    expected_row_ids = np.arange(
        EXPECTED_TEST_ROWS,
        dtype=np.int64,
    )

    if not np.array_equal(
        output["row_id"].to_numpy(),
        expected_row_ids,
    ):
        raise RuntimeError(
            "A ordem dos row_id foi alterada."
        )

    if output[
        [
            "prob_c1",
            "prob_c234",
            "prob_c5",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Existem probabilidades NaN."
        )

    prob_sum = (
        output[
            [
                "prob_c1",
                "prob_c234",
                "prob_c5",
            ]
        ]
        .sum(axis=1)
        .to_numpy()
    )

    if not np.allclose(
        prob_sum,
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "As probabilidades não somam 1."
        )

    # --------------------------------------------------------
    # Salvar
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Resumo operacional
    #
    # Isso NÃO é avaliação do modelo.
    # Não há y_true no conjunto de teste.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREVISÕES GERADAS")
    print("=" * 70)

    print(
        output["y_pred"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
        .to_string()
    )

    print()
    print(
        f"Total: {len(output)}"
    )

    print()
    print(
        "Arquivo salvo em:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Colunas:"
    )

    print(
        list(output.columns)
    )

    print()
    print("=" * 70)
    print("BASELINE FINAL CONCLUÍDO")
    print("=" * 70)


if __name__ == "__main__":
    main()