from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
TEST_PATH = ROOT_DIR / "data" / "raw" / "test1.xlsx"

RESULTS_DIR = ROOT_DIR / "results"

OUTPUT_PATH = (
    RESULTS_DIR
    / "tfidf_word_char_linearsvc_final.csv"
)


# ============================================================
# CONFIGURAÇÃO CONGELADA
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

ID2LABEL = {
    0: "c1",
    1: "c234",
    2: "c5",
}

# Selecionado anteriormente por CV.
# NÃO realizar novo grid search.
C = 0.5

SEED = 42

EXPECTED_TRAIN_ROWS = 20_092
EXPECTED_TEST_ROWS = 900


# ============================================================
# VETORIZADORES HISTÓRICOS
# ============================================================

def criar_vetorizadores():

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


# ============================================================
# CARREGAMENTO DOS DADOS
# ============================================================

def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("=" * 78)
    print("LINEARSVC FINAL — CARREGAMENTO DOS DADOS")
    print("=" * 78)

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"train.xlsx não encontrado:\n{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"test1.xlsx não encontrado:\n{TEST_PATH}"
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
    # Estrutura
    # --------------------------------------------------------

    required_train = {
        TEXT_COL,
        LABEL_COL,
    }

    missing_train = (
        required_train
        - set(train_df.columns)
    )

    if missing_train:
        raise ValueError(
            "Colunas ausentes em train.xlsx: "
            f"{sorted(missing_train)}"
        )

    if TEXT_COL not in test_df.columns:
        raise ValueError(
            f"Coluna '{TEXT_COL}' ausente em test1.xlsx."
        )

    # --------------------------------------------------------
    # Quantidade de linhas
    # --------------------------------------------------------

    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            "Número inesperado de linhas no treino: "
            f"{len(train_df)}; "
            f"esperado {EXPECTED_TRAIN_ROWS}."
        )

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            "Número inesperado de linhas no teste: "
            f"{len(test_df)}; "
            f"esperado {EXPECTED_TEST_ROWS}."
        )

    # --------------------------------------------------------
    # Texto
    #
    # Mesma preparação do experimento histórico.
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
    # Labels
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

    unknown = (
        set(train_df[LABEL_COL].unique())
        - set(LABELS)
    )

    if unknown:
        raise ValueError(
            "Labels desconhecidos no treino: "
            f"{sorted(unknown)}"
        )

    # --------------------------------------------------------
    # Segurança: test não deve ter target preenchido
    # --------------------------------------------------------

    if LABEL_COL in test_df.columns:

        non_null_test_labels = (
            test_df[LABEL_COL]
            .dropna()
        )

        if len(non_null_test_labels) > 0:
            raise ValueError(
                "test1.xlsx possui valores preenchidos em "
                "'clarity'. Interrompendo."
            )

    print(f"Treino: {len(train_df)} linhas")
    print(f"Teste : {len(test_df)} linhas")

    print()
    print("Distribuição do treino:")

    print(
        train_df[LABEL_COL]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return train_df, test_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    train_df, test_df = carregar_dados()

    print()
    print("=" * 78)
    print("TF-IDF WORD + CHAR + LinearSVC — FINAL")
    print("=" * 78)

    print("Configuração congelada:")
    print("  word ngram_range       = (1, 2)")
    print("  word min_df            = 2")
    print("  word max_df            = 0.995")
    print("  word max_features      = 120000")
    print("  char analyzer          = char_wb")
    print("  char ngram_range       = (3, 5)")
    print("  char min_df            = 2")
    print("  char max_features      = 180000")
    print(f"  LinearSVC C            = {C}")
    print('  class_weight           = "balanced"')
    print(f"  random_state           = {SEED}")
    print("  max_iter               = 5000")

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_train = (
        train_df[LABEL_COL]
        .map(LABEL2ID)
        .astype(np.int64)
        .to_numpy()
    )

    train_text = (
        train_df[TEXT_COL]
        .tolist()
    )

    test_text = (
        test_df[TEXT_COL]
        .tolist()
    )

    # ========================================================
    # VETORIZAÇÃO
    # ========================================================

    print()
    print("=" * 78)
    print("VETORIZAÇÃO")
    print("=" * 78)

    word, char = criar_vetorizadores()

    print()
    print("Ajustando TF-IDF de palavras no treino completo...")

    Xw_train = word.fit_transform(
        train_text
    )

    print("Transformando teste...")

    Xw_test = word.transform(
        test_text
    )

    print(
        f"Word features: {Xw_train.shape[1]}"
    )

    print()
    print("Ajustando TF-IDF de caracteres no treino completo...")

    Xc_train = char.fit_transform(
        train_text
    )

    print("Transformando teste...")

    Xc_test = char.transform(
        test_text
    )

    print(
        f"Char features: {Xc_train.shape[1]}"
    )

    # --------------------------------------------------------
    # Concatenação
    # --------------------------------------------------------

    X_train = hstack(
        [
            Xw_train,
            Xc_train,
        ],
        format="csr",
    )

    X_test = hstack(
        [
            Xw_test,
            Xc_test,
        ],
        format="csr",
    )

    print()
    print(
        f"Shape treino final: {X_train.shape}"
    )

    print(
        f"Shape teste final : {X_test.shape}"
    )

    # ========================================================
    # TREINAMENTO
    # ========================================================

    print()
    print("=" * 78)
    print("TREINAMENTO")
    print("=" * 78)

    clf = LinearSVC(
        C=C,
        class_weight="balanced",
        random_state=SEED,
        max_iter=5000,
    )

    print(
        f"Treinando em {len(train_df)} exemplos..."
    )

    clf.fit(
        X_train,
        y_train,
    )

    print("Treinamento concluído.")

    # ========================================================
    # INFERÊNCIA
    # ========================================================

    print()
    print("=" * 78)
    print("INFERÊNCIA")
    print("=" * 78)

    y_pred_id = (
        clf.predict(X_test)
        .astype(np.int64)
    )

    y_pred = np.array(
        [
            ID2LABEL[int(value)]
            for value in y_pred_id
        ],
        dtype=object,
    )

    # ========================================================
    # RESULTADO
    # ========================================================

    output = pd.DataFrame(
        {
            "row_id": np.arange(
                len(test_df),
                dtype=np.int64,
            ),
            "y_pred_id": y_pred_id,
            "y_pred": y_pred,
        }
    )

    # --------------------------------------------------------
    # Validações
    # --------------------------------------------------------

    if len(output) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Número incorreto de previsões: "
            f"{len(output)}."
        )

    if output["row_id"].duplicated().any():
        raise RuntimeError(
            "row_id duplicado."
        )

    if not np.array_equal(
        output["row_id"].to_numpy(),
        np.arange(
            EXPECTED_TEST_ROWS,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError(
            "A ordem das linhas foi alterada."
        )

    if output[
        [
            "y_pred_id",
            "y_pred",
        ]
    ].isna().any().any():

        raise RuntimeError(
            "Existem previsões ausentes."
        )

    if not set(
        output["y_pred_id"].unique()
    ).issubset({0, 1, 2}):

        raise RuntimeError(
            "O LinearSVC produziu classe inesperada."
        )

    expected_labels = (
        output["y_pred_id"]
        .map(ID2LABEL)
    )

    if not (
        expected_labels.to_numpy()
        == output["y_pred"].to_numpy()
    ).all():

        raise RuntimeError(
            "Inconsistência entre y_pred_id e y_pred."
        )

    # ========================================================
    # SALVAMENTO
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("PREVISÕES GERADAS")
    print("=" * 78)

    print(
        output["y_pred"]
        .value_counts()
        .reindex(
            LABELS,
            fill_value=0,
        )
        .to_string()
    )

    print()
    print(f"Total: {len(output)}")

    print()
    print("Arquivo salvo em:")
    print(OUTPUT_PATH)

    print()
    print("Colunas:")
    print(list(output.columns))

    print()
    print("=" * 78)
    print("LINEARSVC FINAL CONCLUÍDO")
    print("=" * 78)


if __name__ == "__main__":
    main()