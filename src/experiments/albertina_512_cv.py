from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "PORTULAN/albertina-100m-portuguese-ptbr-encoder"
LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
LABEL_ORDER = ["c1", "c234", "c5"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.xlsx"
DEFAULT_FOLDS_PATH = PROJECT_ROOT / "data" / "splits" / "folds.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints" / "albertina_512"

DEFAULT_CV_PATH = RESULTS_DIR / "albertina_512_cv.csv"
DEFAULT_HISTORY_PATH = RESULTS_DIR / "albertina_512_history.csv"
DEFAULT_OOF_PATH = RESULTS_DIR / "albertina_512_oof.csv"
DEFAULT_SUMMARY_PATH = RESULTS_DIR / "albertina_512_summary.csv"

DEFAULT_MAX_LENGTH = 512
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCHS = 2.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_EVAL_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 16
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_SEED = 42


class FixedGASTrainer(Trainer):
    """
    Mantem a normalizacao de gradient accumulation controlada pelo Trainer.

    Usamos model_accepts_loss_kwargs=False para evitar que o Trainer suponha
    que o modelo downstream normaliza a loss por num_items_in_batch.
    Isso preserva o mesmo protocolo já validado nos experimentos anteriores.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False


class NonFiniteGradNormCallback(TrainerCallback):
    """
    Interrompe imediatamente o treino se o Trainer registrar grad_norm NaN/Inf.

    Isso evita gastar horas de GPU em uma execução numericamente inválida.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "grad_norm" not in logs:
            return control

        value = logs.get("grad_norm")
        if value is None:
            return control

        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False

        if not finite:
            raise RuntimeError(
                f"grad_norm não finito detectado no step {state.global_step}: {value}"
            )

        return control


class EncodedDataset(Dataset):
    def __init__(self, encodings: dict[str, list[Any]], labels: list[int]) -> None:
        if len(labels) == 0:
            raise ValueError("Dataset vazio.")

        n = len(labels)
        for key, values in encodings.items():
            if len(values) != n:
                raise ValueError(
                    f"Campo '{key}' possui {len(values)} itens, esperado {n}."
                )

        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {key: values[idx] for key, values in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Primeira CV cientifica da Albertina 100M PT-BR. "
            "Por padrao roda apenas o fold 0."
        )
    )

    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--folds-file", type=Path, default=DEFAULT_FOLDS_PATH)

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0],
        help="Folds de validacao a executar. Padrao: 0",
    )

    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tokenization-batch-size", type=int, default=256)

    parser.add_argument("--cv-output", type=Path, default=DEFAULT_CV_PATH)
    parser.add_argument("--history-output", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--oof-output", type=Path, default=DEFAULT_OOF_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--checkpoints-dir", type=Path, default=CHECKPOINTS_DIR)

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Nao retoma checkpoint existente de um fold interrompido.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Refaz fold mesmo que ele ja conste no CSV de CV.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.folds:
        raise ValueError("Informe ao menos um fold.")

    if len(set(args.folds)) != len(args.folds):
        raise ValueError("--folds contem valores repetidos.")

    for fold in args.folds:
        if fold < 0:
            raise ValueError("Fold nao pode ser negativo.")

    positive = {
        "--max-length": args.max_length,
        "--batch-size": args.batch_size,
        "--eval-batch-size": args.eval_batch_size,
        "--gradient-accumulation": args.gradient_accumulation,
        "--tokenization-batch-size": args.tokenization_batch_size,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} deve ser maior que zero.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate deve ser maior que zero.")

    if args.epochs <= 0:
        raise ValueError("--epochs deve ser maior que zero.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay nao pode ser negativo.")


def detect_fold_columns(folds: pd.DataFrame) -> tuple[str, str]:
    fold_candidates = ("fold", "fold_id")
    row_candidates = ("row_index", "row_id", "index")

    fold_col = next((c for c in fold_candidates if c in folds.columns), None)
    row_col = next((c for c in row_candidates if c in folds.columns), None)

    if fold_col is None:
        raise ValueError(
            "Nao encontrei coluna de fold. Esperava uma entre: "
            + ", ".join(fold_candidates)
        )

    if row_col is None:
        raise ValueError(
            "Nao encontrei ID de linha nos folds. Esperava uma entre: "
            + ", ".join(row_candidates)
        )

    return row_col, fold_col


def load_dataset_with_folds(
    train_path: Path,
    folds_path: Path,
) -> pd.DataFrame:
    if not train_path.exists():
        raise FileNotFoundError(f"train.xlsx nao encontrado: {train_path}")

    if not folds_path.exists():
        raise FileNotFoundError(f"folds.csv nao encontrado: {folds_path}")

    train = pd.read_excel(train_path).copy()
    folds = pd.read_csv(folds_path).copy()

    required = {"resp_text", "clarity"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(
            "Colunas ausentes em train.xlsx: " + ", ".join(sorted(missing))
        )

    row_col, fold_col = detect_fold_columns(folds)

    if len(folds) != len(train):
        raise ValueError(
            f"folds.csv possui {len(folds)} linhas, train.xlsx possui {len(train)}."
        )

    if folds[row_col].duplicated().any():
        raise ValueError(f"A coluna '{row_col}' de folds.csv possui IDs duplicados.")

    numeric_ids = pd.to_numeric(folds[row_col], errors="raise").astype(int)
    expected = np.arange(len(train), dtype=int)

    if not np.array_equal(np.sort(numeric_ids.to_numpy()), expected):
        raise ValueError(
            f"'{row_col}' nao corresponde exatamente aos IDs 0..{len(train)-1}. "
            "Interrompendo para nao correr risco de usar folds incorretos."
        )

    train["row_id"] = np.arange(len(train), dtype=int)

    fold_map = pd.DataFrame(
        {
            "row_id": numeric_ids.to_numpy(),
            "fold": pd.to_numeric(folds[fold_col], errors="raise").astype(int),
        }
    )

    merged = train.merge(
        fold_map,
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    if merged["fold"].isna().any():
        raise ValueError("Ha linhas de train.xlsx sem fold.")

    merged["resp_text"] = merged["resp_text"].fillna("").astype(str)
    merged["clarity"] = merged["clarity"].astype(str).str.strip()
    merged["fold"] = merged["fold"].astype(int)

    unexpected_labels = sorted(set(merged["clarity"]) - set(LABEL2ID))
    if unexpected_labels:
        raise ValueError(f"Classes inesperadas: {unexpected_labels}")

    return merged.sort_values("row_id").reset_index(drop=True)


def tokenize_texts(
    texts: list[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    label: str,
) -> dict[str, list[Any]]:
    outputs: dict[str, list[Any]] = {}
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        encoded = tokenizer(
            texts[start:end],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        for key, values in encoded.items():
            outputs.setdefault(key, []).extend(values)

        print(
            f"\rTokenizando {label}: {end:>5}/{total} "
            f"({100.0 * end / total:6.2f}%)",
            end="",
            flush=True,
        )

    print()
    return outputs


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_c1": f1_score(
            labels, preds, labels=[LABEL2ID["c1"]], average="macro", zero_division=0
        ),
        "f1_c234": f1_score(
            labels,
            preds,
            labels=[LABEL2ID["c234"]],
            average="macro",
            zero_division=0,
        ),
        "f1_c5": f1_score(
            labels, preds, labels=[LABEL2ID["c5"]], average="macro", zero_division=0
        ),
    }


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def latest_checkpoint(fold_dir: Path) -> Path | None:
    if not fold_dir.exists():
        return None

    checkpoints: list[tuple[int, Path]] = []
    pattern = re.compile(r"^checkpoint-(\d+)$")

    for path in fold_dir.iterdir():
        if not path.is_dir():
            continue

        match = pattern.match(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def completed_folds(cv_path: Path) -> set[int]:
    if not cv_path.exists():
        return set()

    df = pd.read_csv(cv_path)
    if "fold" not in df.columns:
        return set()

    if "status" in df.columns:
        df = df[df["status"] == "success"]

    return set(pd.to_numeric(df["fold"], errors="coerce").dropna().astype(int))


def append_or_replace_by_key(
    path: Path,
    new_rows: pd.DataFrame,
    key_columns: list[str],
    sort_columns: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old = pd.read_csv(path)

        if all(col in old.columns for col in key_columns):
            new_keys = set(map(tuple, new_rows[key_columns].to_numpy().tolist()))
            old_keys = old[key_columns].apply(tuple, axis=1)
            old = old[~old_keys.isin(new_keys)]

        combined = pd.concat([old, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()

    if sort_columns and all(col in combined.columns for col in sort_columns):
        combined = combined.sort_values(sort_columns).reset_index(drop=True)

    combined.to_csv(path, index=False, encoding="utf-8-sig")


def save_summary(
    cv_path: Path,
    oof_path: Path,
    summary_path: Path,
) -> None:
    if not cv_path.exists() or not oof_path.exists():
        return

    cv = pd.read_csv(cv_path)
    oof = pd.read_csv(oof_path)

    cv_ok = cv[cv["status"] == "success"].copy() if "status" in cv.columns else cv

    if cv_ok.empty or oof.empty:
        return

    y_true = oof["y_true_id"].astype(int).to_numpy()
    y_pred = oof["y_pred_id"].astype(int).to_numpy()

    completed = sorted(pd.to_numeric(cv_ok["fold"]).astype(int).unique().tolist())

    row = {
        "model_name": MODEL_NAME,
        "completed_folds": ",".join(map(str, completed)),
        "n_completed_folds": len(completed),
        "n_oof": len(oof),
        "oof_accuracy": accuracy_score(y_true, y_pred),
        "oof_macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "oof_f1_c1": f1_score(
            y_true,
            y_pred,
            labels=[LABEL2ID["c1"]],
            average="macro",
            zero_division=0,
        ),
        "oof_f1_c234": f1_score(
            y_true,
            y_pred,
            labels=[LABEL2ID["c234"]],
            average="macro",
            zero_division=0,
        ),
        "oof_f1_c5": f1_score(
            y_true,
            y_pred,
            labels=[LABEL2ID["c5"]],
            average="macro",
            zero_division=0,
        ),
        "cv_accuracy_mean": cv_ok["accuracy"].mean(),
        "cv_accuracy_std": cv_ok["accuracy"].std(ddof=0),
        "cv_macro_f1_mean": cv_ok["macro_f1"].mean(),
        "cv_macro_f1_std": cv_ok["macro_f1"].std(ddof=0),
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[LABEL2ID[label] for label in LABEL_ORDER],
    )

    for i, true_label in enumerate(LABEL_ORDER):
        for j, pred_label in enumerate(LABEL_ORDER):
            row[f"cm_{true_label}_to_{pred_label}"] = int(cm[i, j])

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )


def gb(value: int) -> float:
    return value / (1024**3)


def run_fold(
    df: pd.DataFrame,
    tokenizer,
    data_collator,
    fold: int,
    args: argparse.Namespace,
) -> None:
    fold_dir = args.checkpoints_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_df = df[df["fold"] != fold].copy()
    val_df = df[df["fold"] == fold].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Fold {fold}: treino ou validacao vazios "
            f"(train={len(train_df)}, val={len(val_df)})."
        )

    print("\n" + "=" * 80)
    print(f"NORBERTO BASE 1024 — FOLD {fold}")
    print("=" * 80)
    print(f"Treino                  : {len(train_df)}")
    print(f"Validacao               : {len(val_df)}")
    print(f"max_length              : {args.max_length}")
    print(f"learning_rate           : {args.learning_rate}")
    print(f"epochs                  : {args.epochs}")
    print(f"batch fisico            : {args.batch_size}")
    print(f"gradient accumulation   : {args.gradient_accumulation}")
    print(
        f"batch efetivo           : "
        f"{args.batch_size * args.gradient_accumulation}"
    )
    print("Precisao                : FP32 forçado no carregamento")
    print("gradient checkpointing  : False")
    print("GAS fix                 : model_accepts_loss_kwargs=False")
    print("=" * 80)

    train_enc = tokenize_texts(
        train_df["resp_text"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.tokenization_batch_size,
        label=f"treino fold {fold}",
    )

    val_enc = tokenize_texts(
        val_df["resp_text"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.tokenization_batch_size,
        label=f"validacao fold {fold}",
    )

    train_labels = train_df["clarity"].map(LABEL2ID).astype(int).tolist()
    val_labels = val_df["clarity"].map(LABEL2ID).astype(int).tolist()

    train_dataset = EncodedDataset(train_enc, train_labels)
    val_dataset = EncodedDataset(val_enc, val_labels)

    # IMPORTANTE: a cabeça de classificação é inicializada aleatoriamente.
    # O seed precisa ser redefinido imediatamente antes de carregar o modelo,
    # caso contrário folds executados sequencialmente recebem inicializações
    # diferentes apesar de TrainingArguments(seed=42).
    set_seed(args.seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        problem_type="single_label_classification",
        # Transformers 5.x respeita o dtype salvo no checkpoint.
        # A Albertina declara BF16, mas a implementação DeBERTa desta
        # versão apresenta incompatibilidades de dtype em mixed precision.
        # Forçamos integralmente FP32, configuração já validada no benchmark.
        dtype=torch.float32,
    )

    floating_dtypes = sorted(
        {str(p.dtype) for p in model.parameters() if p.is_floating_point()}
    )
    print(f"model dtype             : {model.dtype}")
    print(f"floating parameter dtypes: {floating_dtypes}")

    if floating_dtypes != ["torch.float32"]:
        raise RuntimeError(
            "Albertina não foi carregada integralmente em FP32. "
            f"Dtypes encontrados: {floating_dtypes}"
        )

    training_args = TrainingArguments(
        output_dir=str(fold_dir),

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        # Transformers 5.x unificou warmup_ratio em warmup_steps.
        # 0 preserva exatamente nossa configuração: sem warmup.
        warmup_steps=0,

        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=50,
        logging_first_step=True,

        report_to=[],
        seed=args.seed,
        data_seed=args.seed,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        remove_unused_columns=True,
        disable_tqdm=False,

        load_best_model_at_end=False,
    )

    trainer = FixedGASTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[NonFiniteGradNormCallback()],
    )

    if trainer.model_accepts_loss_kwargs is not False:
        raise RuntimeError("GAS fix nao foi aplicado corretamente.")

    resume_checkpoint = None
    if not args.no_resume:
        resume_checkpoint = latest_checkpoint(fold_dir)

    if resume_checkpoint is not None:
        print(f"Retomando checkpoint: {resume_checkpoint}")
    else:
        print("Treino iniciando do checkpoint base da Albertina.")

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    started = time.perf_counter()

    status = "success"
    error_type = None
    error_message = None
    train_output = None

    try:
        train_output = trainer.train(
            resume_from_checkpoint=(
                str(resume_checkpoint) if resume_checkpoint is not None else None
            )
        )

        torch.cuda.synchronize()
        runtime_seconds = time.perf_counter() - started

        # Avaliacao final explicita usando os pesos ao fim da epoca 2.
        eval_metrics = trainer.evaluate(eval_dataset=val_dataset)

        prediction_output = trainer.predict(val_dataset)
        logits = prediction_output.predictions

        if isinstance(logits, tuple):
            logits = logits[0]

        probs = softmax_numpy(np.asarray(logits))
        preds = probs.argmax(axis=1).astype(int)
        y_true = np.asarray(val_labels, dtype=int)

        accuracy = accuracy_score(y_true, preds)
        macro_f1 = f1_score(
            y_true,
            preds,
            average="macro",
            zero_division=0,
        )

        f1_values = f1_score(
            y_true,
            preds,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])

        oof = pd.DataFrame(
            {
                "row_id": val_df["row_id"].astype(int).to_numpy(),
                "fold": fold,
                "y_true_id": y_true,
                "y_pred_id": preds,
                "y_true": [ID2LABEL[int(v)] for v in y_true],
                "y_pred": [ID2LABEL[int(v)] for v in preds],
                "prob_c1": probs[:, LABEL2ID["c1"]],
                "prob_c234": probs[:, LABEL2ID["c234"]],
                "prob_c5": probs[:, LABEL2ID["c5"]],
                "correct": (y_true == preds),
                "resp_text": val_df["resp_text"].to_numpy(),
            }
        )

        append_or_replace_by_key(
            args.oof_output,
            oof,
            key_columns=["row_id"],
            sort_columns=["row_id"],
        )

        history = pd.DataFrame(trainer.state.log_history)
        if not history.empty:
            history.insert(0, "fold", fold)
            history.insert(1, "model_name", MODEL_NAME)

            # Um fold reexecutado substitui todo o historico antigo dele.
            if args.history_output.exists():
                old_hist = pd.read_csv(args.history_output)
                if "fold" in old_hist.columns:
                    old_hist = old_hist[
                        pd.to_numeric(old_hist["fold"], errors="coerce") != fold
                    ]
                history = pd.concat(
                    [old_hist, history],
                    ignore_index=True,
                    sort=False,
                )

            args.history_output.parent.mkdir(parents=True, exist_ok=True)
            history.to_csv(
                args.history_output,
                index=False,
                encoding="utf-8-sig",
            )

        peak_allocated = gb(torch.cuda.max_memory_allocated())
        peak_reserved = gb(torch.cuda.max_memory_reserved())

        cv_row = pd.DataFrame(
            [
                {
                    "fold": fold,
                    "status": status,
                    "error_type": error_type,
                    "error_message": error_message,
                    "model_name": MODEL_NAME,
                    "max_length": args.max_length,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation,
                    "effective_batch_size": (
                        args.batch_size * args.gradient_accumulation
                    ),
                    "weight_decay": args.weight_decay,
                    "fp16": False,
                    "bf16": False,
                    "forced_fp32": True,
                    "gradient_checkpointing": False,
                    "gas_normalization_fix": True,
                    "seed_reset_before_model_init": True,
                    "abort_on_nonfinite_grad_norm": True,
                    "seed": args.seed,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "f1_c1": float(f1_values[0]),
                    "f1_c234": float(f1_values[1]),
                    "f1_c5": float(f1_values[2]),
                    "eval_loss": eval_metrics.get("eval_loss"),
                    "train_loss": (
                        train_output.metrics.get("train_loss")
                        if train_output is not None
                        else None
                    ),
                    "runtime_seconds": runtime_seconds,
                    "peak_vram_allocated_gb": peak_allocated,
                    "peak_vram_reserved_gb": peak_reserved,
                    "cm_c1_to_c1": int(cm[0, 0]),
                    "cm_c1_to_c234": int(cm[0, 1]),
                    "cm_c1_to_c5": int(cm[0, 2]),
                    "cm_c234_to_c1": int(cm[1, 0]),
                    "cm_c234_to_c234": int(cm[1, 1]),
                    "cm_c234_to_c5": int(cm[1, 2]),
                    "cm_c5_to_c1": int(cm[2, 0]),
                    "cm_c5_to_c234": int(cm[2, 1]),
                    "cm_c5_to_c5": int(cm[2, 2]),
                    "transformers_version": transformers.__version__,
                    "torch_version": torch.__version__,
                    "python_version": platform.python_version(),
                    "cuda_device": torch.cuda.get_device_name(0),
                }
            ]
        )

        append_or_replace_by_key(
            args.cv_output,
            cv_row,
            key_columns=["fold"],
            sort_columns=["fold"],
        )

        save_summary(
            args.cv_output,
            args.oof_output,
            args.summary_output,
        )

        print("\n" + "-" * 80)
        print(f"FOLD {fold} CONCLUIDO")
        print("-" * 80)
        print(f"Accuracy              : {accuracy:.6f}")
        print(f"Macro-F1              : {macro_f1:.6f}")
        print(f"F1 c1                 : {f1_values[0]:.6f}")
        print(f"F1 c234               : {f1_values[1]:.6f}")
        print(f"F1 c5                 : {f1_values[2]:.6f}")
        print(f"Eval loss             : {eval_metrics.get('eval_loss')}")
        print(f"Tempo                  : {runtime_seconds / 60:.2f} min")
        print(f"VRAM allocated pico    : {peak_allocated:.3f} GB")
        print(f"VRAM reserved pico     : {peak_reserved:.3f} GB")
        print("Matriz de confusao:")
        print(cm)
        print("-" * 80)

    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        runtime_seconds = time.perf_counter() - started
        status = (
            "oom"
            if isinstance(exc, torch.cuda.OutOfMemoryError)
            else "error"
        )
        error_type = type(exc).__name__
        error_message = str(exc)[:2000]

        cv_row = pd.DataFrame(
            [
                {
                    "fold": fold,
                    "status": status,
                    "error_type": error_type,
                    "error_message": error_message,
                    "model_name": MODEL_NAME,
                    "max_length": args.max_length,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation,
                    "effective_batch_size": (
                        args.batch_size * args.gradient_accumulation
                    ),
                    "weight_decay": args.weight_decay,
                    "fp16": False,
                    "bf16": False,
                    "forced_fp32": True,
                    "gradient_checkpointing": False,
                    "gas_normalization_fix": True,
                    "seed_reset_before_model_init": True,
                    "abort_on_nonfinite_grad_norm": True,
                    "seed": args.seed,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "runtime_seconds": runtime_seconds,
                    "transformers_version": transformers.__version__,
                    "torch_version": torch.__version__,
                    "python_version": platform.python_version(),
                    "cuda_device": torch.cuda.get_device_name(0),
                }
            ]
        )

        append_or_replace_by_key(
            args.cv_output,
            cv_row,
            key_columns=["fold"],
            sort_columns=["fold"],
        )

        print(
            f"\nFold {fold} terminou com {status}: "
            f"{error_type}: {error_message}",
            file=sys.stderr,
        )
        raise

    finally:
        del trainer
        del model
        del train_dataset
        del val_dataset
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    validate_args(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nao esta disponivel. Este treino foi planejado para GPU."
        )

    print("=" * 80)
    print("NORBERTO BASE 1024 — PRIMEIRA CV")
    print("=" * 80)
    print(f"Modelo                  : {MODEL_NAME}")
    print("Precisao do modelo      : FP32 forçado")
    print(f"Transformers            : {transformers.__version__}")
    print(f"PyTorch                 : {torch.__version__}")
    print(f"CUDA device             : {torch.cuda.get_device_name(0)}")
    print(f"Folds solicitados       : {args.folds}")
    print(f"max_length              : {args.max_length}")
    print(f"LR                      : {args.learning_rate}")
    print(f"epochs                  : {args.epochs}")
    print(
        f"batch efetivo           : "
        f"{args.batch_size * args.gradient_accumulation}"
    )
    print("Precisao                : FP16")
    print("GAS fix                 : ATIVO")
    print("=" * 80)

    set_seed(args.seed)

    df = load_dataset_with_folds(
        train_path=args.train,
        folds_path=args.folds_file,
    )

    available_folds = sorted(df["fold"].unique().tolist())
    print(f"Folds encontrados       : {available_folds}")

    for fold in args.folds:
        if fold not in available_folds:
            raise ValueError(
                f"Fold {fold} nao existe. Disponiveis: {available_folds}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    already_done = completed_folds(args.cv_output)

    for fold in args.folds:
        if fold in already_done and not args.rerun_completed:
            print(
                f"\nFold {fold} ja consta como success em {args.cv_output}. "
                "Pulando. Use --rerun-completed para refazer."
            )
            continue

        run_fold(
            df=df,
            tokenizer=tokenizer,
            data_collator=data_collator,
            fold=fold,
            args=args,
        )

    print("\nArquivos atuais:")
    print(f"  CV      : {args.cv_output}")
    print(f"  History : {args.history_output}")
    print(f"  OOF     : {args.oof_output}")
    print(f"  Summary : {args.summary_output}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nExecucao interrompida. O checkpoint mais recente do fold "
            "podera ser retomado na proxima execucao.",
            file=sys.stderr,
        )
        raise SystemExit(130)
