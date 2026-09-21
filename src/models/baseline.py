from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def criar_baseline_oficial() -> Pipeline:
    """
    Cria o baseline oficial do trabalho:

    TF-IDF + Regressão Logística com balanceamento de classes.

    O Pipeline garante que o TfidfVectorizer seja ajustado apenas
    nos dados fornecidos ao método fit(), evitando vazamento de
    informação entre treinamento e validação.
    """

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(),
        ),
        (
            "classificador",
            LogisticRegression(
                class_weight="balanced",
            ),
        ),
    ])