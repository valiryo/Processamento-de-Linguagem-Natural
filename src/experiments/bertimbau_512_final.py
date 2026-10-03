from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# CAMINHOS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "train.xlsx"
)

TEST_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "test1.xlsx"
)

RESULTS_DIR = (
    ROOT_DIR
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "bertimbau_512_final.csv"
)

TRAIN_OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_512_final"
)


# ============================================================
# CONFIGURAÇÃO CONGELADA
# ============================================================

MODEL_NAME = (
    "neuralmind/"
    "bert-base-portuguese-cased"
)

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

MAX_LENGTH = 512

LEARNING_RATE = 3e-5
EPOCHS = 2

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

WEIGHT_DECAY = 0.01

FP16 = True
GRADIENT_CHECKPOINTING = False

SEED = 42

EXPECTED_TRAIN_ROWS = 20_092
EXPECTED_TEST_ROWS = 900


# ============================================================
# DATASETS
# ============================================================

class TrainDataset(Dataset):

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: np.ndarray,
    ):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        item = {
            key: value[idx]
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long,
        )

        return item


class TestDataset(Dataset):

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
    ):
        self.encodings = encodings

    def __len__(self) -> int:

        first_key = next(
            iter(self.encodings)
        )

        return len(
            self.encodings[first_key]
        )

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        return {
            key: value[idx]
            for key, value in self.encodings.items()
        }


# ============================================================
# SOFTMAX
# ============================================================

def softmax_numpy(
    logits: np.ndarray,
) -> np.ndarray:

    logits = logits.astype(
        np.float64
    )

    logits = (
        logits
        - logits.max(
            axis=1,
            keepdims=True,
        )
    )

    exp = np.exp(logits)

    return (
        exp
        / exp.sum(
            axis=1,
            keepdims=True,
        )
    )


# ============================================================
# DADOS
# ============================================================

def carregar_dados() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    print("=" * 78)
    print(
        "BERTIMBAU 512 FINAL — "
        "CARREGAMENTO"
    )
    print("=" * 78)

    if not TRAIN_PATH.exists():

        raise FileNotFoundError(
            f"train.xlsx não encontrado:\n"
            f"{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            f"test1.xlsx não encontrado:\n"
            f"{TEST_PATH}"
        )

    train_df = (
        pd.read_excel(
            TRAIN_PATH
        )
        .reset_index(
            drop=True
        )
    )

    test_df = (
        pd.read_excel(
            TEST_PATH
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Estrutura
    # --------------------------------------------------------

    required = {
        "resp_text",
        "clarity",
    }

    missing = (
        required
        - set(train_df.columns)
    )

    if missing:

        raise ValueError(
            "Colunas ausentes em train.xlsx: "
            f"{sorted(missing)}"
        )

    if "resp_text" not in test_df.columns:

        raise ValueError(
            "Coluna 'resp_text' ausente "
            "em test1.xlsx."
        )

    # --------------------------------------------------------
    # Tamanho
    # --------------------------------------------------------

    if len(train_df) != EXPECTED_TRAIN_ROWS:

        raise ValueError(
            "Número inesperado de linhas "
            "em train.xlsx: "
            f"{len(train_df)}"
        )

    if len(test_df) != EXPECTED_TEST_ROWS:

        raise ValueError(
            "Número inesperado de linhas "
            "em test1.xlsx: "
            f"{len(test_df)}"
        )

    # --------------------------------------------------------
    # Textos
    # --------------------------------------------------------

    train_df["resp_text"] = (
        train_df["resp_text"]
        .fillna("")
        .astype(str)
    )

    test_df["resp_text"] = (
        test_df["resp_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if train_df["clarity"].isna().any():

        raise ValueError(
            "Existem labels ausentes "
            "no conjunto de treino."
        )

    train_df["clarity"] = (
        train_df["clarity"]
        .astype(str)
        .str.strip()
    )

    unknown = (
        set(
            train_df[
                "clarity"
            ].unique()
        )
        - set(LABEL2ID)
    )

    if unknown:

        raise ValueError(
            "Labels desconhecidos: "
            f"{sorted(unknown)}"
        )

    # --------------------------------------------------------
    # O teste deve permanecer não rotulado.
    # --------------------------------------------------------

    if "clarity" in test_df.columns:

        if (
            test_df["clarity"]
            .notna()
            .any()
        ):

            raise ValueError(
                "test1.xlsx possui valores "
                "preenchidos em clarity."
            )

    print(
        f"Treino: {len(train_df)}"
    )

    print(
        f"Teste : {len(test_df)}"
    )

    return (
        train_df,
        test_df,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA não está disponível."
        )

    print("=" * 78)
    print("BERTIMBAU 512 — MODELO FINAL")
    print("=" * 78)

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print()
    print("Configuração congelada:")

    print(
        f"  modelo                = "
        f"{MODEL_NAME}"
    )

    print(
        f"  max_length            = "
        f"{MAX_LENGTH}"
    )

    print(
        f"  learning_rate         = "
        f"{LEARNING_RATE}"
    )

    print(
        f"  epochs                = "
        f"{EPOCHS}"
    )

    print(
        f"  batch físico          = "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        "  gradient accumulation = "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        "  batch efetivo         = "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"  weight_decay          = "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"  FP16                  = "
        f"{FP16}"
    )

    print(
        "  gradient checkpointing = "
        f"{GRADIENT_CHECKPOINTING}"
    )

    print(
        f"  seed                  = "
        f"{SEED}"
    )

    # ========================================================
    # DADOS
    # ========================================================

    train_df, test_df = (
        carregar_dados()
    )

    train_texts = (
        train_df[
            "resp_text"
        ]
        .tolist()
    )

    test_texts = (
        test_df[
            "resp_text"
        ]
        .tolist()
    )

    labels_series = (
        train_df[
            "clarity"
        ]
        .map(
            LABEL2ID
        )
    )

    if labels_series.isna().any():

        raise ValueError(
            "Falha ao mapear labels."
        )

    labels = (
        labels_series
        .astype(
            np.int64
        )
        .to_numpy()
    )

    # ========================================================
    # TOKENIZAÇÃO
    # ========================================================

    print()
    print("=" * 78)
    print("TOKENIZAÇÃO")
    print("=" * 78)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "Tokenizando treino..."
    )

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print(
        "Tokenizando teste..."
    )

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = TrainDataset(
        encodings=train_encodings,
        labels=labels,
    )

    test_dataset = TestDataset(
        encodings=test_encodings,
    )

    print(
        f"Train dataset: "
        f"{len(train_dataset)}"
    )

    print(
        f"Test dataset : "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # MODELO
    # ========================================================

    # Mesmo comportamento do experimento OOF:
    # seed definida antes da inicialização
    # da cabeça classificadora.
    set_seed(SEED)

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=3,
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )
    )

    if GRADIENT_CHECKPOINTING:

        model.gradient_checkpointing_enable()

    TRAIN_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = TrainingArguments(

        output_dir=str(
            TRAIN_OUTPUT_DIR
        ),

        learning_rate=(
            LEARNING_RATE
        ),

        num_train_epochs=(
            EPOCHS
        ),

        weight_decay=(
            WEIGHT_DECAY
        ),

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # Não existe validação nesta fase final.
        eval_strategy="no",

        # Mesmo protocolo histórico:
        # não selecionamos checkpoints.
        save_strategy="no",

        fp16=FP16,

        logging_strategy="steps",
        logging_steps=100,

        report_to="none",

        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # ========================================================
    # TREINAMENTO
    # ========================================================

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print()
    print("=" * 78)
    print("TREINAMENTO FINAL")
    print("=" * 78)

    print(
        "Treinando BERTimbau 512 "
        f"com {len(train_dataset)} exemplos..."
    )

    inicio = time.perf_counter()

    try:

        trainer.train()

        torch.cuda.synchronize()

        training_seconds = (
            time.perf_counter()
            - inicio
        )

        peak_vram = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )

        print()
        print(
            "Treinamento concluído."
        )

        print(
            "Tempo: "
            f"{training_seconds / 60:.2f} min"
        )

        print(
            "Pico VRAM: "
            f"{peak_vram:.2f} GB"
        )

        # ====================================================
        # INFERÊNCIA
        # ====================================================

        print()
        print("=" * 78)
        print("INFERÊNCIA NO TESTE")
        print("=" * 78)

        prediction_output = (
            trainer.predict(
                test_dataset
            )
        )

        logits = (
            prediction_output
            .predictions
        )

        if isinstance(
            logits,
            tuple,
        ):

            logits = logits[0]

        probabilities = (
            softmax_numpy(
                np.asarray(
                    logits
                )
            )
        )

        y_pred_id = np.argmax(
            probabilities,
            axis=1,
        ).astype(
            np.int64
        )

        y_pred = np.array(
            [
                ID2LABEL[int(x)]
                for x in y_pred_id
            ],
            dtype=object,
        )

        # ====================================================
        # RESULTADO
        # ====================================================

        output = pd.DataFrame(
            {
                "row_id": np.arange(
                    len(test_df),
                    dtype=np.int64,
                ),

                "y_pred_id": (
                    y_pred_id
                ),

                "y_pred": (
                    y_pred
                ),

                "prob_c1": (
                    probabilities[:, 0]
                ),

                "prob_c234": (
                    probabilities[:, 1]
                ),

                "prob_c5": (
                    probabilities[:, 2]
                ),
            }
        )

        # ====================================================
        # VALIDAÇÃO DO RESULTADO
        # ====================================================

        if len(output) != EXPECTED_TEST_ROWS:

            raise RuntimeError(
                "Número incorreto de previsões: "
                f"{len(output)}."
            )

        if (
            output[
                "row_id"
            ]
            .duplicated()
            .any()
        ):

            raise RuntimeError(
                "row_id duplicado."
            )

        expected_ids = np.arange(
            EXPECTED_TEST_ROWS,
            dtype=np.int64,
        )

        if not np.array_equal(
            output[
                "row_id"
            ].to_numpy(),
            expected_ids,
        ):

            raise RuntimeError(
                "A ordem das linhas "
                "foi alterada."
            )

        if output.isna().any().any():

            raise RuntimeError(
                "Existem valores NaN "
                "no resultado."
            )

        probability_sum = (
            output[
                [
                    "prob_c1",
                    "prob_c234",
                    "prob_c5",
                ]
            ]
            .sum(
                axis=1
            )
            .to_numpy()
        )

        if not np.allclose(
            probability_sum,
            1.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                "Probabilidades não "
                "somam 1."
            )

        argmax_check = np.argmax(
            output[
                [
                    "prob_c1",
                    "prob_c234",
                    "prob_c5",
                ]
            ].to_numpy(),
            axis=1,
        )

        if not np.array_equal(
            argmax_check,
            output[
                "y_pred_id"
            ].to_numpy(),
        ):

            raise RuntimeError(
                "y_pred_id não coincide "
                "com argmax das probabilidades."
            )

        # ====================================================
        # SALVAMENTO
        # ====================================================

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
            output[
                "y_pred"
            ]
            .value_counts()
            .reindex(
                [
                    "c1",
                    "c234",
                    "c5",
                ],
                fill_value=0,
            )
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
        print("=" * 78)
        print(
            "BERTIMBAU 512 FINAL "
            "CONCLUÍDO"
        )
        print("=" * 78)

    finally:

        del trainer
        del model

        gc.collect()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()