from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


# ============================================================
# CONFIGURAÇÃO CONGELADA
# ============================================================

MODEL_NAME = "Itau-Unibanco/NorBERTo-base"

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

LABELS = [
    "c1",
    "c234",
    "c5",
]

MAX_LENGTH = 1024

LEARNING_RATE = 2e-5
EPOCHS = 2.0

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 8

WEIGHT_DECAY = 0.01

SEED = 42

TOKENIZATION_BATCH_SIZE = 256

EXPECTED_TRAIN_ROWS = 20_092
EXPECTED_TEST_ROWS = 900


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
    / "norberto_base_1024_final.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "norberto_base_1024_final"
)


# ============================================================
# FIX HISTÓRICO DO GRADIENT ACCUMULATION
# ============================================================

class FixedGASTrainer(Trainer):
    """
    Workaround usado no experimento original do NorBERTo.

    ModernBERT aceita **kwargs, mas sua CrossEntropyLoss(mean)
    não normaliza por num_items_in_batch.

    Forçar model_accepts_loss_kwargs=False faz o Trainer
    aplicar corretamente o gradient accumulation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.model_accepts_loss_kwargs = False


# ============================================================
# PROTEÇÃO CONTRA GRAD_NORM NaN / Inf
# ============================================================

class NonFiniteGradNormCallback(
    TrainerCallback
):

    def on_log(
        self,
        args,
        state,
        control,
        logs=None,
        **kwargs,
    ):

        if (
            not logs
            or "grad_norm" not in logs
        ):
            return control

        value = logs.get(
            "grad_norm"
        )

        if value is None:
            return control

        try:

            finite = math.isfinite(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):

            finite = False

        if not finite:

            raise RuntimeError(
                "grad_norm não finito "
                f"detectado no step "
                f"{state.global_step}: "
                f"{value}"
            )

        return control


# ============================================================
# DATASETS
# ============================================================

class EncodedTrainDataset(Dataset):

    def __init__(
        self,
        encodings: dict[
            str,
            list[Any],
        ],
        labels: list[int],
    ) -> None:

        if len(labels) == 0:

            raise ValueError(
                "Dataset de treino vazio."
            )

        n = len(labels)

        for key, values in encodings.items():

            if len(values) != n:

                raise ValueError(
                    f"Campo '{key}' possui "
                    f"{len(values)} itens; "
                    f"esperado {n}."
                )

        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:

        return len(
            self.labels
        )

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, Any]:

        item = {
            key: values[idx]
            for key, values
            in self.encodings.items()
        }

        item["labels"] = (
            self.labels[idx]
        )

        return item


class EncodedTestDataset(Dataset):

    def __init__(
        self,
        encodings: dict[
            str,
            list[Any],
        ],
    ) -> None:

        if not encodings:

            raise ValueError(
                "Encodings do teste vazios."
            )

        lengths = {
            len(values)
            for values
            in encodings.values()
        }

        if len(lengths) != 1:

            raise ValueError(
                "Campos tokenizados possuem "
                "quantidades diferentes."
            )

        self.encodings = encodings
        self.n = next(
            iter(lengths)
        )

    def __len__(self) -> int:

        return self.n

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, Any]:

        return {
            key: values[idx]
            for key, values
            in self.encodings.items()
        }


# ============================================================
# TOKENIZAÇÃO HISTÓRICA
# ============================================================

def tokenize_texts(
    texts: list[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    label: str,
) -> dict[str, list[Any]]:

    outputs: dict[
        str,
        list[Any],
    ] = {}

    total = len(texts)

    for start in range(
        0,
        total,
        batch_size,
    ):

        end = min(
            start + batch_size,
            total,
        )

        encoded = tokenizer(
            texts[start:end],

            truncation=True,

            max_length=(
                max_length
            ),

            # IMPORTANTE:
            # NorBERTo histórico usa
            # padding dinâmico.
            padding=False,

            return_attention_mask=True,
        )

        for key, values in (
            encoded.items()
        ):

            outputs.setdefault(
                key,
                [],
            ).extend(
                values
            )

        print(
            f"\rTokenizando {label}: "
            f"{end:>5}/{total} "
            f"({100.0 * end / total:6.2f}%)",
            end="",
            flush=True,
        )

    print()

    return outputs


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

    exp = np.exp(
        logits
    )

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

    print("=" * 80)
    print(
        "NORBERTO BASE 1024 FINAL "
        "— CARREGAMENTO"
    )
    print("=" * 80)

    if not TRAIN_PATH.exists():

        raise FileNotFoundError(
            "train.xlsx não encontrado: "
            f"{TRAIN_PATH}"
        )

    if not TEST_PATH.exists():

        raise FileNotFoundError(
            "test1.xlsx não encontrado: "
            f"{TEST_PATH}"
        )

    train = (
        pd.read_excel(
            TRAIN_PATH
        )
        .reset_index(
            drop=True
        )
    )

    test = (
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
        - set(train.columns)
    )

    if missing:

        raise ValueError(
            "Colunas ausentes no treino: "
            f"{sorted(missing)}"
        )

    if (
        "resp_text"
        not in test.columns
    ):

        raise ValueError(
            "Coluna resp_text ausente "
            "no teste."
        )

    # --------------------------------------------------------
    # Número de linhas
    # --------------------------------------------------------

    if (
        len(train)
        != EXPECTED_TRAIN_ROWS
    ):

        raise ValueError(
            "Número inesperado de linhas "
            f"no treino: {len(train)}"
        )

    if (
        len(test)
        != EXPECTED_TEST_ROWS
    ):

        raise ValueError(
            "Número inesperado de linhas "
            f"no teste: {len(test)}"
        )

    # --------------------------------------------------------
    # Texto
    # --------------------------------------------------------

    train["resp_text"] = (
        train["resp_text"]
        .fillna("")
        .astype(str)
    )

    test["resp_text"] = (
        test["resp_text"]
        .fillna("")
        .astype(str)
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if (
        train["clarity"]
        .isna()
        .any()
    ):

        raise ValueError(
            "Há labels ausentes "
            "no treino."
        )

    train["clarity"] = (
        train["clarity"]
        .astype(str)
        .str.strip()
    )

    unexpected = sorted(
        set(
            train[
                "clarity"
            ].unique()
        )
        - set(LABEL2ID)
    )

    if unexpected:

        raise ValueError(
            "Classes inesperadas: "
            f"{unexpected}"
        )

    # Teste precisa continuar sem rótulos.
    if (
        "clarity" in test.columns
        and test[
            "clarity"
        ].notna().any()
    ):

        raise ValueError(
            "test1.xlsx contém "
            "clarity preenchido."
        )

    print(
        f"Treino : {len(train)}"
    )

    print(
        f"Teste  : {len(test)}"
    )

    return train, test


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA não está disponível."
        )

    print("=" * 80)
    print(
        "NORBERTO BASE 1024 — FINAL"
    )
    print("=" * 80)

    print(
        f"Modelo                  : "
        f"{MODEL_NAME}"
    )

    print(
        f"CUDA device             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"max_length              : "
        f"{MAX_LENGTH}"
    )

    print(
        f"learning_rate           : "
        f"{LEARNING_RATE}"
    )

    print(
        f"epochs                  : "
        f"{EPOCHS}"
    )

    print(
        f"batch físico            : "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        "gradient accumulation   : "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        "batch efetivo           : "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"weight_decay            : "
        f"{WEIGHT_DECAY}"
    )

    print(
        "scheduler               : linear"
    )

    print(
        "warmup                  : 0"
    )

    print(
        "FP16                    : True"
    )

    print(
        "BF16                    : False"
    )

    print(
        "gradient checkpointing  : False"
    )

    print(
        "max_grad_norm           : 1.0"
    )

    print(
        "GAS fix                 : ATIVO"
    )

    print("=" * 80)

    # ========================================================
    # DADOS
    # ========================================================

    train_df, test_df = (
        carregar_dados()
    )

    train_labels = (
        train_df[
            "clarity"
        ]
        .map(
            LABEL2ID
        )
        .astype(int)
        .tolist()
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
    )

    # Padding dinâmico — histórico.
    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt",
        )
    )

    # ========================================================
    # TOKENIZAÇÃO
    # ========================================================

    print()
    print("=" * 80)
    print("TOKENIZAÇÃO")
    print("=" * 80)

    train_enc = tokenize_texts(
        train_df[
            "resp_text"
        ].tolist(),

        tokenizer=tokenizer,

        max_length=MAX_LENGTH,

        batch_size=(
            TOKENIZATION_BATCH_SIZE
        ),

        label="treino completo",
    )

    test_enc = tokenize_texts(
        test_df[
            "resp_text"
        ].tolist(),

        tokenizer=tokenizer,

        max_length=MAX_LENGTH,

        batch_size=(
            TOKENIZATION_BATCH_SIZE
        ),

        label="teste",
    )

    train_dataset = (
        EncodedTrainDataset(
            train_enc,
            train_labels,
        )
    )

    test_dataset = (
        EncodedTestDataset(
            test_enc
        )
    )

    # ========================================================
    # MODELO
    # ========================================================

    # O histórico exige reset do seed
    # imediatamente antes da criação
    # da cabeça classificadora.
    set_seed(
        SEED
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,

            num_labels=len(
                LABEL2ID
            ),

            label2id=LABEL2ID,

            id2label=ID2LABEL,

            problem_type=(
                "single_label_classification"
            ),
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # TRAINING ARGUMENTS
    # ========================================================

    training_args = (
        TrainingArguments(

            output_dir=str(
                OUTPUT_DIR
            ),

            # Não há validação nesta fase.
            eval_strategy="no",

            # Não precisamos selecionar
            # ou salvar checkpoints.
            save_strategy="no",

            num_train_epochs=(
                EPOCHS
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

            learning_rate=(
                LEARNING_RATE
            ),

            weight_decay=(
                WEIGHT_DECAY
            ),

            lr_scheduler_type=(
                "linear"
            ),

            warmup_steps=0,

            fp16=True,

            bf16=False,

            gradient_checkpointing=False,

            max_grad_norm=1.0,

            logging_strategy="steps",

            logging_steps=50,

            logging_first_step=True,

            report_to=[],

            seed=SEED,

            data_seed=SEED,

            dataloader_num_workers=0,

            dataloader_pin_memory=True,

            remove_unused_columns=True,

            disable_tqdm=False,

            load_best_model_at_end=False,
        )
    )

    trainer = FixedGASTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        data_collator=data_collator,

        callbacks=[
            NonFiniteGradNormCallback()
        ],
    )

    if (
        trainer
        .model_accepts_loss_kwargs
        is not False
    ):

        raise RuntimeError(
            "GAS fix não foi "
            "aplicado corretamente."
        )

    # ========================================================
    # TREINAMENTO
    # ========================================================

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()

    print()
    print("=" * 80)
    print("TREINAMENTO FINAL")
    print("=" * 80)

    started = (
        time.perf_counter()
    )

    try:

        train_output = (
            trainer.train()
        )

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - started
        )

        peak_allocated = (
            torch.cuda
            .max_memory_allocated()
            / (1024**3)
        )

        peak_reserved = (
            torch.cuda
            .max_memory_reserved()
            / (1024**3)
        )

        print()
        print(
            "Treinamento concluído."
        )

        print(
            f"Epoch final             : "
            f"{trainer.state.epoch}"
        )

        print(
            f"Global step             : "
            f"{trainer.state.global_step}"
        )

        print(
            f"Tempo                   : "
            f"{elapsed / 60:.2f} min"
        )

        print(
            "VRAM allocated pico     : "
            f"{peak_allocated:.3f} GB"
        )

        print(
            "VRAM reserved pico      : "
            f"{peak_reserved:.3f} GB"
        )

        # ====================================================
        # INFERÊNCIA
        # ====================================================

        print()
        print("=" * 80)
        print("INFERÊNCIA NO TESTE")
        print("=" * 80)

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

        y_pred_id = (
            probabilities
            .argmax(axis=1)
            .astype(int)
        )

        y_pred = np.array(
            [
                ID2LABEL[int(x)]
                for x
                in y_pred_id
            ],
            dtype=object,
        )

        # ====================================================
        # OUTPUT
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
                    probabilities[
                        :,
                        LABEL2ID["c1"],
                    ]
                ),

                "prob_c234": (
                    probabilities[
                        :,
                        LABEL2ID["c234"],
                    ]
                ),

                "prob_c5": (
                    probabilities[
                        :,
                        LABEL2ID["c5"],
                    ]
                ),
            }
        )

        # ====================================================
        # VALIDAÇÕES
        # ====================================================

        if (
            len(output)
            != EXPECTED_TEST_ROWS
        ):

            raise RuntimeError(
                "Número incorreto "
                "de previsões."
            )

        expected_row_ids = (
            np.arange(
                EXPECTED_TEST_ROWS,
                dtype=np.int64,
            )
        )

        if not np.array_equal(
            output[
                "row_id"
            ].to_numpy(),
            expected_row_ids,
        ):

            raise RuntimeError(
                "Ordem dos row_id "
                "foi alterada."
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

        if (
            output
            .isna()
            .any()
            .any()
        ):

            raise RuntimeError(
                "Há valores NaN "
                "no resultado."
            )

        prob_cols = [
            "prob_c1",
            "prob_c234",
            "prob_c5",
        ]

        if not np.allclose(
            output[
                prob_cols
            ].sum(axis=1),
            1.0,
            atol=1e-6,
        ):

            raise RuntimeError(
                "Probabilidades "
                "não somam 1."
            )

        expected_argmax = (
            np.argmax(
                output[
                    prob_cols
                ].to_numpy(),
                axis=1,
            )
        )

        if not np.array_equal(
            expected_argmax,
            output[
                "y_pred_id"
            ].to_numpy(),
        ):

            raise RuntimeError(
                "y_pred_id não coincide "
                "com argmax."
            )

        expected_labels = (
            output[
                "y_pred_id"
            ]
            .map(
                ID2LABEL
            )
        )

        if not np.array_equal(
            expected_labels.to_numpy(),
            output[
                "y_pred"
            ].to_numpy(),
        ):

            raise RuntimeError(
                "Inconsistência entre "
                "y_pred_id e y_pred."
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
        print("=" * 80)
        print("PREVISÕES GERADAS")
        print("=" * 80)

        print(
            output[
                "y_pred"
            ]
            .value_counts()
            .reindex(
                LABELS,
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
        print("=" * 80)
        print(
            "NORBERTO FINAL CONCLUÍDO"
        )
        print("=" * 80)

    finally:

        del trainer
        del model
        del train_dataset
        del test_dataset

        gc.collect()

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()