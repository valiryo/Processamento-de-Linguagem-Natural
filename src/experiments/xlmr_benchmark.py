from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT / "data" / "splits" / "folds.csv"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TEXT_COL = "resp_text"
LABEL_COL = "clarity"

SEED = 42

MAX_LENGTH = 384

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

PHYSICAL_BATCH = 4
GRAD_ACC = 4

MAX_STEPS = 100

class FixedGATrainer(Trainer):
    """
    Evita que o Trainer tente tratar o modelo como responsável
    pelo escalonamento da loss durante gradient accumulation.

    XLMRobertaForSequenceClassification retorna uma loss já
    reduzida (mean), portanto o Trainer deve fazer o tratamento
    padrão de gradient accumulation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

# ============================================================
# REPRODUCIBILITY
# ============================================================

def reset_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("XLM-R BASE — BENCHMARK OPERACIONAL")
    print("=" * 80)

    print(f"\nModelo: {MODEL_NAME}")
    print(f"max_length: {MAX_LENGTH}")
    print(f"physical batch: {PHYSICAL_BATCH}")
    print(f"gradient accumulation: {GRAD_ACC}")
    print(
        f"effective batch: "
        f"{PHYSICAL_BATCH * GRAD_ACC}"
    )
    print(f"max_steps: {MAX_STEPS}")

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não disponível. "
            "Benchmark deve ser executado na GPU."
        )

    device_name = torch.cuda.get_device_name(0)

    print(f"\nGPU: {device_name}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nCarregando dados...")

    df = pd.read_excel(DATA_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    # Normalizar texto para evitar tipos mistos vindos do Excel
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    # Normalizar labels também
    df[LABEL_COL] = df[LABEL_COL].astype(str)

    # Os folds precisam estar perfeitamente alinhados
    if len(df) != len(folds):
        raise ValueError(
            f"train.xlsx possui {len(df)} linhas, "
            f"mas folds.csv possui {len(folds)}."
        )

    # --------------------------------------------------------
    # FOLD 0
    # --------------------------------------------------------

    train_mask = folds["fold"] != 0

    train_df = df.loc[train_mask].copy()

    print(f"Treino fold 0: {len(train_df):,}")

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    # Fit no dataset inteiro é aceitável aqui porque estamos apenas
    # mapeando nomes fixos de classes para IDs.
    label_encoder.fit(df[LABEL_COL])

    print(
        "Classes:",
        list(label_encoder.classes_)
    )

    train_df["labels"] = label_encoder.transform(
        train_df[LABEL_COL]
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nCarregando tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = Dataset.from_pandas(
        train_df[
            [TEXT_COL, "labels"]
        ].reset_index(drop=True),
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizando...")

    dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[TEXT_COL],
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # reset seed IMMEDIATELY before model initialization
    # --------------------------------------------------------

    reset_seed(SEED)

    print("\nInicializando modelo...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_encoder.classes_),
    )

    # --------------------------------------------------------
    # TRAINING ARGS
    # --------------------------------------------------------

    output_dir = ROOT / "results" / "_tmp_xlmr_benchmark"

    args = TrainingArguments(
        output_dir=str(output_dir),

        # benchmark
        max_steps=MAX_STEPS,

        # optimization
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=PHYSICAL_BATCH,
        gradient_accumulation_steps=GRAD_ACC,

        # precision
        fp16=True,
        bf16=False,

        # memory
        gradient_checkpointing=False,

        # logging
        logging_strategy="steps",
        logging_steps=10,

        # não precisamos avaliar/salvar
        eval_strategy="no",
        save_strategy="no",

        # reproducibility
        seed=SEED,
        data_seed=SEED,

        # misc
        report_to="none",
        dataloader_num_workers=0,

        max_grad_norm=1.0,

        lr_scheduler_type="constant",
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    trainer = FixedGATrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # MEMORY RESET
    # --------------------------------------------------------

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------------
    # BENCHMARK
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("INICIANDO BENCHMARK")
    print("=" * 80)

    start = time.perf_counter()

    result = trainer.train()

    elapsed = time.perf_counter() - start

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    sec_per_step = elapsed / MAX_STEPS

    print("\n" + "=" * 80)
    print("RESULTADO DO BENCHMARK")
    print("=" * 80)

    print(f"\nTempo total: {elapsed:.2f} s")
    print(
        f"Tempo / step: "
        f"{sec_per_step:.3f} s"
    )

    print(
        f"\nPeak VRAM allocated: "
        f"{peak_allocated:.3f} GB"
    )

    print(
        f"Peak VRAM reserved: "
        f"{peak_reserved:.3f} GB"
    )

    print(
        f"\nTrain loss: "
        f"{result.training_loss:.6f}"
    )

    print("\nBenchmark concluído.")


if __name__ == "__main__":
    main()