from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def criar_folds(
    df: pd.DataFrame,
    coluna_texto: str = "resp_text",
    coluna_alvo: str = "clarity",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.Series:
    """
    Cria folds estratificados, mantendo textos idênticos no mesmo fold.

    O uso de StratifiedGroupKFold evita que uma mesma resposta textual
    apareça simultaneamente nos conjuntos de treinamento e validação,
    ao mesmo tempo em que busca preservar a distribuição das classes.

    Parameters
    ----------
    df:
        DataFrame contendo os textos e rótulos.

    coluna_texto:
        Nome da coluna contendo os textos. Também é utilizada como grupo.

    coluna_alvo:
        Nome da coluna contendo a classe alvo.

    n_splits:
        Número de folds da validação cruzada.

    random_state:
        Semente utilizada para garantir reprodutibilidade.

    Returns
    -------
    pd.Series
        Série contendo o número do fold atribuído a cada instância.
    """

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_ids = np.full(len(df), -1, dtype=int)

    textos = df[coluna_texto]
    classes = df[coluna_alvo]

    for fold, (_, validation_indices) in enumerate(
        splitter.split(
            X=textos,
            y=classes,
            groups=textos,
        )
    ):
        fold_ids[validation_indices] = fold

    if np.any(fold_ids == -1):
        raise RuntimeError(
            "Algumas instâncias não foram atribuídas a nenhum fold."
        )

    return pd.Series(
        fold_ids,
        index=df.index,
        name="fold",
    )


def carregar_folds(
    df: pd.DataFrame,
    caminho_folds: str | Path,
) -> pd.DataFrame:
    """
    Carrega uma atribuição de folds previamente salva
    e a associa ao DataFrame.

    A associação é feita pelo índice original das instâncias,
    garantindo que todos os experimentos utilizem exatamente
    a mesma divisão de dados.
    """

    folds = pd.read_csv(caminho_folds)

    if len(folds) != len(df):
        raise ValueError(
            "O número de instâncias do dataset não corresponde "
            "ao número de atribuições de folds."
        )

    if not np.array_equal(
        folds["row_index"].to_numpy(),
        df.index.to_numpy(),
    ):
        raise ValueError(
            "Os índices do dataset não correspondem aos índices "
            "utilizados na criação dos folds."
        )

    df = df.copy()
    df["fold"] = folds["fold"].to_numpy()

    return df