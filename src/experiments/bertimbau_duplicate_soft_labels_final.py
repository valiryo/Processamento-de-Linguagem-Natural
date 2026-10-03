from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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

TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
TEST_PATH = ROOT_DIR / "data" / "raw" / "test1.xlsx"

RESULTS_DIR = ROOT_DIR / "results"
OUTPUT_PATH = RESULTS_DIR / "bertimbau_duplicate_soft_labels_256_final.csv"

TRAIN_OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_duplicate_soft_labels_256_final"
)


# ============================================================
# CONFIGURAÇÃO CONGELADA
# ============================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

LABELS = ["c1", "c234", "c5"]

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

MAX_LENGTH = 256

LEARNING_RATE = 3e-5
EPOCHS = 2

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2

WEIGHT_DECAY = 0.01

FP16 = True
GRADIENT_CHECKPOINTING = False

SEED = 42

EXPECTED_TRAIN_ROWS = 20_092
EXPECTED_TEST_ROWS = 900


# ============================================================
# DATASETS
# ============================================================

class SoftLabelDataset(Dataset):

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        targets: np.ndarray,
    ):
        self.encodings = encodings
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        item = {
            key: value[idx]
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.targets[idx],
            dtype=torch.float32,
        )

        return item


class TestDataset(Dataset):

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
    ):
        self.encodings = encodings

    def __len__(self) -> int:
        key = next(iter(self.encodings))
        return len(self.encodings[key])

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        return {
            key: value[idx]
            for key, value in self.encodings.items()
        }


# ============================================================
# TRAINER HISTÓRICO PARA SOFT LABELS
# ============================================================

class SoftLabelTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):

        targets = inputs.pop("labels").float()

        outputs = model(**inputs)
        logits = outputs.logits

        # Mesmo cálculo histórico:
        # cross-entropy com distribuição alvo.
        # Loss calculada em float32 por estabilidade.
        log_probs = F.log_softmax(
            logits.float(),
            dim=-1,
        )

        loss_per_example = -(
            targets * log_probs
        ).sum(dim=-1)

        loss = loss_per_example.mean()

        if return_outputs:
            return loss, outputs

        return loss


# ============================================================
# SOFTMAX
# ============================================================

def softmax_numpy(
    logits: np.ndarray,
) -> np.ndarray:

    logits = logits.astype(np.float64)

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
    print("SOFT-LABEL FINAL — CARREGAMENTO DOS DADOS")
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

    required = {
        "resp_text",
        "clarity",
    }

    missing = required - set(train_df.columns)

    if missing:
        raise ValueError(
            f"Colunas ausentes em train.xlsx: {sorted(missing)}"
        )

    if "resp_text" not in test_df.columns:
        raise ValueError(
            "Coluna 'resp_text' ausente em test1.xlsx."
        )

    if len(train_df) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"train.xlsx possui {len(train_df)} linhas; "
            f"esperado {EXPECTED_TRAIN_ROWS}."
        )

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"test1.xlsx possui {len(test_df)} linhas; "
            f"esperado {EXPECTED_TEST_ROWS}."
        )

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

    if train_df["clarity"].isna().any():
        raise ValueError(
            "Existem labels ausentes no treino."
        )

    train_df["clarity"] = (
        train_df["clarity"]
        .astype(str)
        .str.strip()
    )

    unknown = (
        set(train_df["clarity"].unique())
        - set(LABELS)
    )

    if unknown:
        raise ValueError(
            f"Labels desconhecidos: {sorted(unknown)}"
        )

    if (
        "clarity" in test_df.columns
        and test_df["clarity"].notna().any()
    ):
        raise ValueError(
            "test1.xlsx possui valores preenchidos em clarity."
        )

    print(f"Treino: {len(train_df)}")
    print(f"Teste : {len(test_df)}")

    return train_df, test_df


# ============================================================
# CONSTRUÇÃO DOS SOFT LABELS
# ============================================================

def construir_targets(
    train_df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:

    """
    Adaptação exata do experimento de CV para treinamento final.

    Para cada resp_text, calcula a distribuição empírica das
    classes usando TODAS as ocorrências no train.xlsx.

    Exemplo:

        texto X:
            c1
            c234
            c234
            c5

        target de todas as quatro ocorrências:
            [0.25, 0.50, 0.25]

    Textos únicos ou duplicatas consistentes permanecem one-hot.
    """

    df = train_df.copy()

    df["_text_key"] = (
        df["resp_text"]
        .fillna("")
        .astype(str)
    )

    counts = pd.crosstab(
        df["_text_key"],
        df["clarity"],
    ).reindex(
        columns=LABELS,
        fill_value=0,
    )

    probs = counts.div(
        counts.sum(axis=1),
        axis=0,
    )

    targets = np.zeros(
        (
            len(df),
            len(LABELS),
        ),
        dtype=np.float32,
    )

    for class_id, label in enumerate(LABELS):

        mapping = probs[label].to_dict()

        targets[:, class_id] = (
            df["_text_key"]
            .map(mapping)
            .to_numpy(dtype=np.float32)
        )

    # --------------------------------------------------------
    # Validações
    # --------------------------------------------------------

    target_sums = targets.sum(axis=1)

    if not np.allclose(
        target_sums,
        1.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Há soft targets que não somam 1."
        )

    num_labels_per_group = (
        (counts > 0)
        .sum(axis=1)
    )

    group_sizes = (
        counts.sum(axis=1)
    )

    conflicting_groups = (
        counts[
            num_labels_per_group > 1
        ]
    )

    stats = {
        "unique_text_groups": int(
            len(counts)
        ),
        "duplicate_groups": int(
            (group_sizes > 1).sum()
        ),
        "conflicting_groups": int(
            len(conflicting_groups)
        ),
        "conflicting_occurrences": int(
            conflicting_groups
            .sum(axis=1)
            .sum()
        ),
    }

    return targets, stats


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível."
        )

    print("=" * 78)
    print("BERTIMBAU DUPLICATE SOFT LABELS 256 — FINAL")
    print("=" * 78)

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print()
    print("Configuração congelada:")

    print(f"  model                 = {MODEL_NAME}")
    print(f"  max_length            = {MAX_LENGTH}")
    print(f"  learning_rate         = {LEARNING_RATE}")
    print(f"  epochs                = {EPOCHS}")
    print(f"  batch físico          = {TRAIN_BATCH_SIZE}")
    print(
        "  gradient accumulation = "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )
    print(
        "  batch efetivo         = "
        f"{TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )
    print(f"  weight_decay          = {WEIGHT_DECAY}")
    print(f"  FP16                  = {FP16}")
    print(
        "  gradient checkpointing = "
        f"{GRADIENT_CHECKPOINTING}"
    )
    print(f"  seed                  = {SEED}")

    # ========================================================
    # DADOS
    # ========================================================

    train_df, test_df = carregar_dados()

    # ========================================================
    # SOFT TARGETS
    # ========================================================

    print()
    print("=" * 78)
    print("CONSTRUÇÃO DOS SOFT LABELS")
    print("=" * 78)

    targets, stats = construir_targets(
        train_df
    )

    print(
        "Grupos de texto únicos       : "
        f"{stats['unique_text_groups']}"
    )

    print(
        "Grupos duplicados            : "
        f"{stats['duplicate_groups']}"
    )

    print(
        "Grupos com conflito          : "
        f"{stats['conflicting_groups']}"
    )

    print(
        "Ocorrências em grupos conflito: "
        f"{stats['conflicting_occurrences']}"
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

    train_texts = (
        train_df["resp_text"]
        .tolist()
    )

    test_texts = (
        test_df["resp_text"]
        .tolist()
    )

    print("Tokenizando treino...")

    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    print("Tokenizando teste...")

    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    train_dataset = SoftLabelDataset(
        encodings=train_encodings,
        targets=targets,
    )

    test_dataset = TestDataset(
        encodings=test_encodings,
    )

    # ========================================================
    # MODELO
    # ========================================================

    set_seed(SEED)

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
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

        learning_rate=LEARNING_RATE,

        num_train_epochs=EPOCHS,

        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        # Não há validação na fase final.
        eval_strategy="no",

        save_strategy="no",

        fp16=FP16,

        logging_strategy="steps",
        logging_steps=100,

        report_to="none",

        seed=SEED,
        data_seed=SEED,
    )

    trainer = SoftLabelTrainer(
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

    inicio = time.perf_counter()

    try:

        trainer.train()

        torch.cuda.synchronize()

        elapsed = (
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
            f"Tempo: {elapsed / 60:.2f} min"
        )

        print(
            f"Pico VRAM: {peak_vram:.2f} GB"
        )

        # ====================================================
        # INFERÊNCIA
        # ====================================================

        print()
        print("=" * 78)
        print("INFERÊNCIA NO TESTE")
        print("=" * 78)

        pred_output = (
            trainer.predict(
                test_dataset
            )
        )

        logits = (
            pred_output.predictions
        )

        if isinstance(
            logits,
            tuple,
        ):
            logits = logits[0]

        probabilities = (
            softmax_numpy(
                np.asarray(logits)
            )
        )

        y_pred_id = (
            np.argmax(
                probabilities,
                axis=1,
            )
            .astype(np.int64)
        )

        y_pred = np.array(
            [
                ID2LABEL[int(x)]
                for x in y_pred_id
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

                "y_pred_id": y_pred_id,

                "y_pred": y_pred,

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
        # VALIDAÇÕES
        # ====================================================

        if len(output) != EXPECTED_TEST_ROWS:
            raise RuntimeError(
                "Número incorreto de previsões."
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

        if output.isna().any().any():
            raise RuntimeError(
                "Existem NaNs no resultado."
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
                "Probabilidades não somam 1."
            )

        argmax = np.argmax(
            output[
                prob_cols
            ].to_numpy(),
            axis=1,
        )

        if not np.array_equal(
            argmax,
            output[
                "y_pred_id"
            ].to_numpy(),
        ):
            raise RuntimeError(
                "y_pred_id não coincide "
                "com argmax."
            )

        # ====================================================
        # SALVAR
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
            output["y_pred"]
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
        print("=" * 78)
        print(
            "SOFT-LABEL FINAL CONCLUÍDO"
        )
        print("=" * 78)

    finally:

        del trainer
        del model

        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()