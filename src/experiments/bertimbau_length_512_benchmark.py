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

TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT_DIR / "data" / "splits" / "folds.csv"

OUTPUT_DIR = (
    ROOT_DIR
    / "output"
    / "bertimbau_length_512_benchmark"
)


# ============================================================
# MODELO
# ============================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

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


# ============================================================
# CONFIGURAÇÃO DO BENCHMARK
#
# IMPORTANTE:
# este arquivo serve apenas para verificar custo computacional,
# VRAM e estabilidade do max_length=512.
#
# Seus resultados NÃO entram na comparação experimental.
# ============================================================

FOLD = 0

MAX_LENGTH = 512

LEARNING_RATE = 3e-5

TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4

GRADIENT_ACCUMULATION_STEPS = 4

WEIGHT_DECAY = 0.01

FP16 = True
GRADIENT_CHECKPOINTING = False

MAX_STEPS = 100

SEED = 42


# ============================================================
# DATASET
# ============================================================

class BertimbauDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: np.ndarray,
        indices: np.ndarray,
    ):
        self.encodings = encodings
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, torch.Tensor]:

        original_idx = self.indices[idx]

        item = {
            key: value[original_idx]
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            self.labels[original_idx],
            dtype=torch.long,
        )

        return item


# ============================================================
# CARREGAMENTO
# ============================================================

def carregar_dados() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    print("=" * 70)
    print("Carregando dados")
    print("=" * 70)

    train_df = pd.read_excel(
        TRAIN_PATH
    )

    folds_df = pd.read_csv(
        FOLDS_PATH
    )

    if len(train_df) != len(folds_df):
        raise ValueError(
            "train.xlsx e folds.csv possuem "
            "números diferentes de linhas: "
            f"{len(train_df)} != {len(folds_df)}"
        )

    if "resp_text" not in train_df.columns:
        raise ValueError(
            "Coluna 'resp_text' não encontrada."
        )

    if "clarity" not in train_df.columns:
        raise ValueError(
            "Coluna 'clarity' não encontrada."
        )

    if "fold" not in folds_df.columns:
        raise ValueError(
            "Coluna 'fold' não encontrada "
            "em folds.csv."
        )

    folds_encontrados = sorted(
        folds_df["fold"]
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"Instâncias: {len(train_df)}"
    )

    print(
        f"Folds encontrados: "
        f"{folds_encontrados}"
    )

    return train_df, folds_df


# ============================================================
# TOKENIZAÇÃO
# ============================================================

def tokenizar_corpus(
    textos: list[str],
) -> dict[str, torch.Tensor]:

    print()
    print("=" * 70)
    print("Tokenização")
    print("=" * 70)

    print(
        f"Modelo: {MODEL_NAME}"
    )

    print(
        f"max_length: {MAX_LENGTH}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    inicio = time.perf_counter()

    encodings = tokenizer(
        textos,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    tempo = (
        time.perf_counter()
        - inicio
    )

    print(
        "Tokenização concluída em "
        f"{tempo:.2f} s"
    )

    return encodings


# ============================================================
# BENCHMARK
# ============================================================

def executar_benchmark() -> None:

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível."
        )

    print("=" * 70)
    print(
        "BERTimbau — Benchmark max_length=512"
    )
    print("=" * 70)

    print(
        "GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    train_df, folds_df = (
        carregar_dados()
    )

    textos = (
        train_df["resp_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels_series = (
        train_df["clarity"]
        .map(LABEL2ID)
    )

    if labels_series.isna().any():
        invalidos = (
            train_df.loc[
                labels_series.isna(),
                "clarity",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Rótulos desconhecidos: "
            f"{invalidos}"
        )

    labels = (
        labels_series
        .astype(np.int64)
        .to_numpy()
    )

    encodings = tokenizar_corpus(
        textos
    )

    fold_values = (
        folds_df["fold"]
        .astype(int)
        .to_numpy()
    )

    train_indices = np.where(
        fold_values != FOLD
    )[0]

    eval_indices = np.where(
        fold_values == FOLD
    )[0]

    train_dataset = BertimbauDataset(
        encodings=encodings,
        labels=labels,
        indices=train_indices,
    )

    eval_dataset = BertimbauDataset(
        encodings=encodings,
        labels=labels,
        indices=eval_indices,
    )

    effective_batch_size = (
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print()
    print("=" * 70)
    print("Configuração")
    print("=" * 70)

    print(
        f"Fold: {FOLD}"
    )

    print(
        f"Treino: {len(train_dataset)}"
    )

    print(
        f"Validação: {len(eval_dataset)}"
    )

    print()

    print(
        f"max_length: {MAX_LENGTH}"
    )

    print(
        f"learning_rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"train batch físico: "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"eval batch físico: "
        f"{EVAL_BATCH_SIZE}"
    )

    print(
        "gradient_accumulation_steps: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"batch efetivo: "
        f"{effective_batch_size}"
    )

    print(
        f"weight_decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"FP16: {FP16}"
    )

    print(
        "gradient checkpointing: "
        f"{GRADIENT_CHECKPOINTING}"
    )

    print(
        f"max_steps: {MAX_STEPS}"
    )

    print()
    print("=" * 70)
    print("Criando modelo")
    print("=" * 70)

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(LABEL2ID),
            label2id=LABEL2ID,
            id2label=ID2LABEL,
        )
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    benchmark_output_dir = (
        OUTPUT_DIR
        / f"batch_{TRAIN_BATCH_SIZE}"
    )

    benchmark_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(
            benchmark_output_dir
        ),

        learning_rate=LEARNING_RATE,
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

        max_steps=MAX_STEPS,

        # Benchmark operacional:
        # nenhuma avaliação.
        eval_strategy="no",

        # Não salvar checkpoints.
        save_strategy="no",

        fp16=FP16,

        logging_strategy="steps",
        logging_steps=10,

        report_to="none",

        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,

        # Não será utilizado porque
        # eval_strategy="no".
        eval_dataset=eval_dataset,
    )

    # ========================================================
    # PREPARAÇÃO DA GPU
    # ========================================================

    gc.collect()

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()

    memoria_inicial = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    print()
    print("=" * 70)
    print("INICIANDO BENCHMARK")
    print("=" * 70)

    print(
        "VRAM alocada antes do treino: "
        f"{memoria_inicial:.2f} GB"
    )

    inicio = time.perf_counter()

    try:
        trainer.train()

        torch.cuda.synchronize()

        tempo_total = (
            time.perf_counter()
            - inicio
        )

        pico_vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        pico_reservado = (
            torch.cuda.max_memory_reserved()
            / 1024**3
        )

        tempo_por_step = (
            tempo_total
            / MAX_STEPS
        )

        print()
        print("=" * 70)
        print("BENCHMARK CONCLUÍDO")
        print("=" * 70)

        print(
            f"max_length: {MAX_LENGTH}"
        )

        print(
            f"batch físico: "
            f"{TRAIN_BATCH_SIZE}"
        )

        print(
            "gradient accumulation: "
            f"{GRADIENT_ACCUMULATION_STEPS}"
        )

        print(
            f"batch efetivo: "
            f"{effective_batch_size}"
        )

        print()

        print(
            f"Tempo total "
            f"({MAX_STEPS} steps): "
            f"{tempo_total:.2f} s"
        )

        print(
            f"Tempo médio por step: "
            f"{tempo_por_step:.4f} s"
        )

        print(
            f"Pico VRAM alocada: "
            f"{pico_vram:.2f} GB"
        )

        print(
            f"Pico VRAM reservada: "
            f"{pico_reservado:.2f} GB"
        )

        print()

        print(
            "O benchmark terminou sem "
            "CUDA Out Of Memory."
        )

    except torch.OutOfMemoryError:

        print()
        print("=" * 70)
        print("CUDA OUT OF MEMORY")
        print("=" * 70)

        print(
            "A configuração "
            f"max_length={MAX_LENGTH}, "
            f"batch={TRAIN_BATCH_SIZE} "
            "não coube na GPU."
        )

        print()
        print(
            "Próxima configuração a testar:"
        )

        print(
            "TRAIN_BATCH_SIZE = 2"
        )

        print(
            "EVAL_BATCH_SIZE = 2"
        )

        print(
            "GRADIENT_ACCUMULATION_STEPS = 8"
        )

        raise

    finally:

        del trainer
        del model

        gc.collect()

        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    executar_benchmark()