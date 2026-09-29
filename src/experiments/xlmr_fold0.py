from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT / "data" / "splits" / "folds.csv"

RESULTS_DIR = ROOT / "results"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TEXT_COL = "resp_text"
LABEL_COL = "clarity"

FOLD = 0
SEED = 42

MAX_LENGTH = 384

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

EPOCHS = 2

PHYSICAL_BATCH = 4
GRAD_ACC = 4


# Ordem explícita para manter compatibilidade
# com os demais OOFs do projeto.
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
# FIX GRADIENT ACCUMULATION
# ============================================================

class FixedGATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Necessário no Transformers 5.17.0 para evitar
        # escalonamento incorreto da loss com GA.
        self.model_accepts_loss_kwargs = False


# ============================================================
# CALLBACK PARA VRAM
# ============================================================

class MemoryCallback(TrainerCallback):

    def on_epoch_begin(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        if torch.cuda.is_available():

            allocated = (
                torch.cuda.max_memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda.max_memory_reserved()
                / 1024**3
            )

            print(
                f"\n[VRAM] epoch={state.epoch:.2f} "
                f"allocated={allocated:.3f} GB "
                f"reserved={reserved:.3f} GB"
            )


# ============================================================
# REPRODUCIBILITY
# ============================================================

def reset_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(
        logits,
        axis=-1,
    )

    return {
        "accuracy": accuracy_score(
            labels,
            preds,
        ),

        "macro_f1": f1_score(
            labels,
            preds,
            average="macro",
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("XLM-R BASE — FOLD 0")
    print("=" * 80)

    print(f"Model: {MODEL_NAME}")
    print(f"max_length: {MAX_LENGTH}")
    print(f"LR: {LEARNING_RATE}")
    print(f"epochs: {EPOCHS}")

    print(
        f"batch: {PHYSICAL_BATCH} "
        f"x GA {GRAD_ACC} "
        f"= {PHYSICAL_BATCH * GRAD_ACC}"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA não disponível."
        )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    # ========================================================
    # DATA
    # ========================================================

    print("\nCarregando dados...")

    df = pd.read_excel(DATA_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    if len(df) != len(folds):
        raise ValueError(
            "train.xlsx e folds.csv "
            "possuem tamanhos diferentes."
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
    )

    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
    )

    unexpected = (
        set(df[LABEL_COL].unique())
        - set(LABEL2ID)
    )

    if unexpected:
        raise ValueError(
            f"Labels inesperados: {unexpected}"
        )

    # --------------------------------------------------------
    # ROW ID
    # --------------------------------------------------------

    if "row_id" in folds.columns:

        df["row_id"] = folds["row_id"].values

    elif "row_id" in df.columns:

        pass

    else:

        # fallback determinístico:
        # posição original no train.xlsx
        df["row_id"] = np.arange(len(df))

    # --------------------------------------------------------
    # FOLD
    # --------------------------------------------------------

    df["fold"] = folds["fold"].values

    train_df = (
        df[df["fold"] != FOLD]
        .copy()
        .reset_index(drop=True)
    )

    val_df = (
        df[df["fold"] == FOLD]
        .copy()
        .reset_index(drop=True)
    )

    train_df["labels"] = (
        train_df[LABEL_COL]
        .map(LABEL2ID)
        .astype(int)
    )

    val_df["labels"] = (
        val_df[LABEL_COL]
        .map(LABEL2ID)
        .astype(int)
    )

    print(
        f"Train: {len(train_df):,}"
    )

    print(
        f"Validation: {len(val_df):,}"
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print("\nCarregando tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = Dataset.from_pandas(
        train_df[
            [TEXT_COL, "labels"]
        ],
        preserve_index=False,
    )

    val_dataset = Dataset.from_pandas(
        val_df[
            [TEXT_COL, "labels"]
        ],
        preserve_index=False,
    )

    def tokenize(batch):

        return tokenizer(
            batch[TEXT_COL],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizando treino...")

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[TEXT_COL],
    )

    print("Tokenizando validação...")

    val_dataset = val_dataset.map(
        tokenize,
        batched=True,
        remove_columns=[TEXT_COL],
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    # ========================================================
    # MODEL
    # ========================================================

    reset_seed(SEED)

    print("\nInicializando modelo...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,

            num_labels=3,

            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    # ========================================================
    # TRAINING
    # ========================================================

    output_dir = (
        RESULTS_DIR
        / "_tmp_xlmr_fold0"
    )

    args = TrainingArguments(

        output_dir=str(output_dir),

        num_train_epochs=EPOCHS,

        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,

        per_device_train_batch_size=(
            PHYSICAL_BATCH
        ),

        per_device_eval_batch_size=(
            PHYSICAL_BATCH
        ),

        gradient_accumulation_steps=(
            GRAD_ACC
        ),

        fp16=True,
        bf16=False,

        gradient_checkpointing=False,

        max_grad_norm=1.0,

        # -----------------------------
        # EVALUATION
        # -----------------------------

        eval_strategy="epoch",

        # Não vamos escolher checkpoint
        # automaticamente neste piloto.
        save_strategy="no",

        # -----------------------------
        # LOGGING
        # -----------------------------

        logging_strategy="steps",
        logging_steps=100,

        # -----------------------------
        # REPRODUCIBILITY
        # -----------------------------

        seed=SEED,
        data_seed=SEED,

        # -----------------------------
        # OTHER
        # -----------------------------

        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = FixedGATrainer(

        model=model,

        args=args,

        train_dataset=train_dataset,
        eval_dataset=val_dataset,

        processing_class=tokenizer,

        data_collator=collator,

        compute_metrics=compute_metrics,

        callbacks=[
            MemoryCallback()
        ],
    )

    # ========================================================
    # TRAIN
    # ========================================================

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\n" + "=" * 80)
    print("TREINAMENTO")
    print("=" * 80)

    start = time.perf_counter()

    train_result = trainer.train()

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\nTempo total: "
        f"{elapsed / 60:.2f} min"
    )

    print(
        f"Train loss: "
        f"{train_result.training_loss:.6f}"
    )

    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    print("\nGerando OOF fold 0...")

    prediction = trainer.predict(
        val_dataset
    )

    logits = prediction.predictions

    probs = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    y_true = (
        val_df["labels"]
        .to_numpy()
    )

    y_pred = np.argmax(
        probs,
        axis=1,
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    print("\n" + "=" * 80)
    print("RESULTADO FINAL — FOLD 0")
    print("=" * 80)

    print(
        f"Accuracy: {accuracy:.6f}"
    )

    print(
        f"Macro-F1: {macro_f1:.6f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[
                "c1",
                "c234",
                "c5",
            ],
            digits=6,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2],
        )
    )

    # ========================================================
    # OOF
    # ========================================================

    oof = pd.DataFrame({

        "row_id":
            val_df["row_id"].values,

        "fold":
            FOLD,

        "y_true_id":
            y_true,

        "y_pred_id":
            y_pred,

        "y_true":
            [ID2LABEL[x] for x in y_true],

        "y_pred":
            [ID2LABEL[x] for x in y_pred],

        "prob_c1":
            probs[:, 0],

        "prob_c234":
            probs[:, 1],

        "prob_c5":
            probs[:, 2],

        "correct":
            y_true == y_pred,

        "resp_text":
            val_df[TEXT_COL].values,
    })

    oof_path = (
        RESULTS_DIR
        / "xlmr_384_fold0_oof.csv"
    )

    oof.to_csv(
        oof_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame([{

        "model": MODEL_NAME,

        "fold": FOLD,

        "max_length": MAX_LENGTH,

        "learning_rate":
            LEARNING_RATE,

        "epochs":
            EPOCHS,

        "physical_batch":
            PHYSICAL_BATCH,

        "gradient_accumulation":
            GRAD_ACC,

        "effective_batch":
            PHYSICAL_BATCH * GRAD_ACC,

        "accuracy":
            accuracy,

        "macro_f1":
            macro_f1,

        "runtime_minutes":
            elapsed / 60,
    }])

    summary_path = (
        RESULTS_DIR
        / "xlmr_384_fold0_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nArquivos salvos:")

    print(oof_path)
    print(summary_path)


if __name__ == "__main__":
    main()