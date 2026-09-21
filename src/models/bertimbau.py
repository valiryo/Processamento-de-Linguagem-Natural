from collections.abc import Iterable

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

LABEL_TO_ID = {
    "c1": 0,
    "c234": 1,
    "c5": 2,
}

ID_TO_LABEL = {
    0: "c1",
    1: "c234",
    2: "c5",
}


def criar_tokenizer():
    """
    Carrega o tokenizer correspondente ao BERTimbau Base.
    """

    return AutoTokenizer.from_pretrained(MODEL_NAME)


def criar_modelo():
    """
    Cria um BERTimbau Base para classificação em três classes.

    A cabeça de classificação é inicializada especificamente para
    as classes do problema.
    """

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_TO_ID),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )


def preparar_dataset(
    textos: Iterable[str],
    rotulos: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """
    Converte textos e rótulos para um Dataset tokenizado.

    O truncamento respeita max_length, mas o padding não é realizado
    nesta etapa. O padding será feito dinamicamente em cada batch,
    evitando processamento e uso de memória desnecessários.
    """

    dataset = Dataset.from_dict({
        "text": [str(texto) for texto in textos],
        "labels": [
            LABEL_TO_ID[rotulo]
            for rotulo in rotulos
        ],
    })

    def tokenizar(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenizar,
        batched=True,
        remove_columns=["text"],
    )