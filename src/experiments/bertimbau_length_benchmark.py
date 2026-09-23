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

OUTPUT_DIR = ROOT_DIR / "output" / "bertimbau_length_benchmark"


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
# este arquivo é APENAS para dimensionamento computacional.
# O resultado NÃO entra na comparação de modelos.
# ============================================================

FOLD = 0

MAX_LENGTH = 384

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
# CARREGAMENTO DOS DADOS
# ============================================================

def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("=" * 70)
    print("Carregando dados")
    print("=" * 70)

    train_df = pd.read_excel(TRAIN_PATH)
    folds_df = pd.read_csv(FOLDS_PATH)

    if len(train_df) != len(folds_df):
        raise ValueError(
            "train.xlsx e folds.csv possuem números diferentes de linhas: "
            f"{len(train_df)} != {len(folds_df)}"
        )

    if "resp_text" not in train_df.columns:
        raise ValueError(
            "Coluna 'resp_text' não encontrada em train.xlsx."
        )

    if "clarity" not in train_df.columns:
        raise ValueError(
            "Coluna 'clarity' não encontrada em train.xlsx."
        )

    if "fold" not in folds_df.columns:
        raise ValueError(
            "Coluna 'fold' não encontrada em folds.csv."
        )

    print(f"Instâncias: {len(train_df)}")
    print(f"Folds encontrados: {sorted(folds_df['fold'].unique())}")

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

    print(f"Modelo: {MODEL_NAME}")
    print(f"max_length: {MAX_LENGTH}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inicio = time.perf_counter()

    encodings = tokenizer(
        textos,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    tempo = time.perf_counter() - inicio

    print(f"Tokenização concluída em {tempo:.2f} s")

    return encodings


# ============================================================
# BENCHMARK
# ============================================================

def executar_benchmark() -> None:
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não está disponível. "
            "Este benchmark foi criado para testar o uso da GPU."
        )

    train_df, folds_df = carregar_dados()

    textos = (
        train_df["resp_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = (
        train_df["clarity"]
        .map(LABEL2ID)
        .to_numpy(dtype=np.int64)
    )

    if np.isnan(labels.astype(float)).any():
        raise ValueError(
            "Foi encontrado algum rótulo que não pertence a "
            f"{list(LABEL2ID.keys())}."
        )

    encodings = tokenizar_corpus(textos)

    fold_values = folds_df["fold"].to_numpy()

    train_indices = np.where(fold_values != FOLD)[0]
    eval_indices = np.where(fold_values == FOLD)[0]

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

    print()
    print("=" * 70)
    print("Configuração")
    print("=" * 70)

    effective_batch_size = (
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    print(f"Fold: {FOLD}")
    print(f"Treino: {len(train_dataset)}")
    print(f"Validação: {len(eval_dataset)}")
    print()
    print(f"max_length: {MAX_LENGTH}")
    print(f"learning_rate: {LEARNING_RATE}")
    print(f"train batch físico: {TRAIN_BATCH_SIZE}")
    print(f"eval batch físico: {EVAL_BATCH_SIZE}")
    print(
        "gradient_accumulation_steps: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )
    print(f"batch efetivo: {effective_batch_size}")
    print(f"weight_decay: {WEIGHT_DECAY}")
    print(f"FP16: {FP16}")
    print(
        "gradient checkpointing: "
        f"{GRADIENT_CHECKPOINTING}"
    )
    print(f"max_steps: {MAX_STEPS}")

    print()
    print("=" * 70)
    print("Criando modelo")
    print("=" * 70)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    benchmark_output_dir = (
        OUTPUT_DIR
        / f"len_{MAX_LENGTH}"
        / f"batch_{TRAIN_BATCH_SIZE}"
    )

    benchmark_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_args = TrainingArguments(
        output_dir=str(benchmark_output_dir),

        # -----------------------------
        # treinamento
        # -----------------------------
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        max_steps=MAX_STEPS,

        # -----------------------------
        # benchmark: NÃO avaliar
        # -----------------------------
        eval_strategy="no",

        # -----------------------------
        # não salvar checkpoints
        # -----------------------------
        save_strategy="no",

        # -----------------------------
        # precisão
        # -----------------------------
        fp16=FP16,

        # -----------------------------
        # logging
        # -----------------------------
        logging_strategy="steps",
        logging_steps=10,

        report_to="none",

        # -----------------------------
        # reprodutibilidade
        # -----------------------------
        seed=SEED,
        data_seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,

        # Mantemos referência ao eval_dataset,
        # mas eval_strategy="no" garante que
        # ele não será avaliado durante o benchmark.
        eval_dataset=eval_dataset,
    )

    # ========================================================
    # MEDIÇÃO DA GPU
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
        f"VRAM alocada antes do treino: "
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

        print(f"max_length: {MAX_LENGTH}")
        print(
            f"batch físico: "
            f"{TRAIN_BATCH_SIZE}"
        )
        print(
            f"gradient accumulation: "
            f"{GRADIENT_ACCUMULATION_STEPS}"
        )
        print(
            f"batch efetivo: "
            f"{effective_batch_size}"
        )
        print()

        print(
            f"Tempo total ({MAX_STEPS} steps): "
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
            f"A configuração max_length={MAX_LENGTH}, "
            f"batch={TRAIN_BATCH_SIZE} "
            "não coube na GPU."
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