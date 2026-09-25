"""
Experimento: BERTimbau 256 + global label smoothing.

Ablação limpa em relação ao BERTimbau 256 padrão:
- mesmos folds congelados;
- mesmos hard labels;
- mesma tokenização/truncamento pelo início;
- mesmos hiperparâmetros;
- única variável metodológica: label_smoothing_factor = 0.1.

Compatibilidade alvo:
- Python 3.11.9
- PyTorch 2.14.0+cu130
- Transformers 5.17.0
"""

from __future__ import annotations

import gc
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


# =============================================================================
# Configuração congelada
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = PROJECT_ROOT / "data" / "splits" / "folds.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

CV_PATH = RESULTS_DIR / "bertimbau_label_smoothing_256_cv.csv"
HISTORY_PATH = RESULTS_DIR / "bertimbau_label_smoothing_256_history.csv"
OOF_PATH = RESULTS_DIR / "bertimbau_label_smoothing_256_oof.csv"
SUMMARY_PATH = RESULTS_DIR / "bertimbau_label_smoothing_256_summary.csv"
OOF_FOLDS_DIR = RESULTS_DIR / "bertimbau_label_smoothing_256_oof_folds"
TRAINER_OUTPUT_DIR = RESULTS_DIR / "_trainer" / "bertimbau_label_smoothing_256"

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

TEXT_COLUMN = "resp_text"
LABEL_COLUMN = "clarity"
FOLD_COLUMN = "fold"
ROW_ID_COLUMN = "row_id"

LABELS = ["c1", "c234", "c5"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

N_FOLDS = 5
EXPECTED_FOLD_SIZES = {
    0: 4019,
    1: 4019,
    2: 4019,
    3: 4016,
    4: 4019,
}

MAX_LENGTH = 256
LEARNING_RATE = 3e-5
NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
WEIGHT_DECAY = 0.01
FP16 = True
GRADIENT_CHECKPOINTING = False
LABEL_SMOOTHING_FACTOR = 0.1
SEED = 42

REFERENCE_RESULTS = {
    "BERT 256 padrão": {
        "accuracy": 0.462922,
        "macro_f1": 0.458239,
    },
    "Soft labels 256": {
        "accuracy": 0.464116,
        "macro_f1": 0.460074,
    },
    "BERT 512 padrão": {
        "accuracy": 0.464962,
        "macro_f1": 0.460948,
    },
}


# =============================================================================
# Utilidades
# =============================================================================

def print_header(title: str) -> None:
    line = "=" * 79
    print(f"\n{line}\n{title}\n{line}")


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Escreve CSV de forma atômica: primeiro temporário, depois substitui."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    tmp_path.replace(path)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def get_peak_vram_gb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024**3)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def warn_if_version_differs() -> None:
    expected = "5.17.0"
    if transformers.__version__ != expected:
        warnings.warn(
            f"Este experimento foi preparado para Transformers {expected}, "
            f"mas a versão encontrada é {transformers.__version__}.",
            RuntimeWarning,
        )


def upsert_cv_record(record: dict[str, Any]) -> None:
    if CV_PATH.exists():
        cv = pd.read_csv(CV_PATH)
        if FOLD_COLUMN in cv.columns:
            cv = cv[cv[FOLD_COLUMN] != record[FOLD_COLUMN]]
        cv = pd.concat([cv, pd.DataFrame([record])], ignore_index=True)
    else:
        cv = pd.DataFrame([record])

    cv = cv.sort_values(FOLD_COLUMN).reset_index(drop=True)
    safe_to_csv(cv, CV_PATH)


def replace_history_for_fold(fold: int, rows: list[dict[str, Any]]) -> None:
    new_history = pd.DataFrame(rows)

    if HISTORY_PATH.exists():
        old_history = pd.read_csv(HISTORY_PATH)
        if FOLD_COLUMN in old_history.columns:
            old_history = old_history[old_history[FOLD_COLUMN] != fold]
        history = pd.concat([old_history, new_history], ignore_index=True, sort=False)
    else:
        history = new_history

    if not history.empty:
        sort_columns = [
            col for col in [FOLD_COLUMN, "epoch", "step"] if col in history.columns
        ]
        if sort_columns:
            history = history.sort_values(sort_columns, na_position="last")

    safe_to_csv(history.reset_index(drop=True), HISTORY_PATH)


def load_completed_folds() -> set[int]:
    """
    Um fold só é considerado completo se:
    1) consta como status=ok no CSV de CV; e
    2) seu OOF individual existe.

    Isso permite retomar de forma segura sem terminar com CV completo e OOF incompleto.
    """
    if not CV_PATH.exists():
        return set()

    cv = pd.read_csv(CV_PATH)
    required = {FOLD_COLUMN, "status"}
    if not required.issubset(cv.columns):
        return set()

    completed: set[int] = set()
    for _, row in cv.iterrows():
        fold = int(row[FOLD_COLUMN])
        oof_fold_path = OOF_FOLDS_DIR / f"fold_{fold}.csv"
        if str(row["status"]).lower() == "ok" and oof_fold_path.exists():
            completed.add(fold)

    return completed


# =============================================================================
# Dados e folds
# =============================================================================

def load_data_with_folds() -> pd.DataFrame:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"train.xlsx não encontrado: {TRAIN_PATH}")

    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"folds.csv não encontrado: {FOLDS_PATH}")

    df = pd.read_excel(TRAIN_PATH).reset_index(drop=True)

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"train.xlsx não possui as colunas obrigatórias: {sorted(missing)}"
        )

    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    unknown_labels = sorted(set(df[LABEL_COLUMN]) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"Labels desconhecidos em {LABEL_COLUMN}: {unknown_labels}")

    df[ROW_ID_COLUMN] = np.arange(len(df), dtype=np.int64)
    df["label_id"] = df[LABEL_COLUMN].map(LABEL2ID).astype(np.int64)

    folds = pd.read_csv(FOLDS_PATH)

    if FOLD_COLUMN not in folds.columns:
        raise ValueError(
            f"folds.csv precisa conter a coluna '{FOLD_COLUMN}'. "
            "Os folds congelados nunca serão recriados automaticamente."
        )

    if ROW_ID_COLUMN in folds.columns:
        fold_map = folds[[ROW_ID_COLUMN, FOLD_COLUMN]].copy()

        if fold_map[ROW_ID_COLUMN].duplicated().any():
            raise ValueError("folds.csv contém row_id duplicado.")

        merged = df.merge(
            fold_map,
            on=ROW_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

        if merged[FOLD_COLUMN].isna().any():
            missing_ids = merged.loc[
                merged[FOLD_COLUMN].isna(), ROW_ID_COLUMN
            ].tolist()[:10]
            raise ValueError(
                "Existem linhas de train.xlsx sem fold atribuído. "
                f"Primeiros row_id: {missing_ids}"
            )

        df = merged

    elif len(folds) == len(df):
        # Fallback compatível com um folds.csv salvo na mesma ordem do train.xlsx.
        # Não fazemos qualquer tentativa de reconstruir os folds por texto/label.
        df[FOLD_COLUMN] = folds[FOLD_COLUMN].to_numpy()

    else:
        raise ValueError(
            "Não foi possível alinhar train.xlsx com folds.csv de forma segura. "
            "Esperado: folds.csv com coluna row_id, ou exatamente o mesmo número "
            "de linhas do train.xlsx na mesma ordem."
        )

    df[FOLD_COLUMN] = df[FOLD_COLUMN].astype(int)

    found_folds = sorted(df[FOLD_COLUMN].unique().tolist())
    expected_folds = list(range(N_FOLDS))
    if found_folds != expected_folds:
        raise ValueError(
            f"Folds encontrados {found_folds}; esperado {expected_folds}. "
            "Os folds não serão recriados."
        )

    # Validação dos tamanhos congelados.
    observed_sizes = df[FOLD_COLUMN].value_counts().sort_index().to_dict()
    for fold, expected_size in EXPECTED_FOLD_SIZES.items():
        observed = int(observed_sizes.get(fold, -1))
        if observed != expected_size:
            raise ValueError(
                f"Fold {fold}: tamanho de validação observado={observed}, "
                f"esperado={expected_size}. Verifique folds.csv."
            )

    # Validação central do protocolo: nenhum texto em treino e validação.
    for fold in range(N_FOLDS):
        train_texts = set(df.loc[df[FOLD_COLUMN] != fold, TEXT_COLUMN])
        val_texts = set(df.loc[df[FOLD_COLUMN] == fold, TEXT_COLUMN])
        overlap = train_texts.intersection(val_texts)
        if overlap:
            raise ValueError(
                f"Fold {fold}: detectado vazamento de {len(overlap)} resp_text "
                "entre treino e validação."
            )

    # Diagnósticos globais somente para análise pós-hoc.
    # Eles NÃO entram no treino nem na loss.
    text_stats = (
        df.groupby(TEXT_COLUMN, sort=False)
        .agg(
            duplicate_group_size=(ROW_ID_COLUMN, "size"),
            duplicate_unique_labels=(LABEL_COLUMN, "nunique"),
        )
        .reset_index()
    )

    text_stats["is_duplicate"] = text_stats["duplicate_group_size"] > 1
    text_stats["duplicate_conflict"] = (
        text_stats["duplicate_unique_labels"] > 1
    )

    df = df.merge(text_stats, on=TEXT_COLUMN, how="left", validate="many_to_one")

    return df


# =============================================================================
# Dataset já tokenizado
# =============================================================================

class IndexedTokenizedDataset(Dataset):
    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.encodings = encodings
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        original_idx = int(self.indices[item])

        sample = {
            key: tensor[original_idx]
            for key, tensor in self.encodings.items()
        }
        sample["labels"] = torch.tensor(
            int(self.labels[original_idx]),
            dtype=torch.long,
        )
        return sample


# =============================================================================
# Métricas
# =============================================================================

def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    y_true = np.asarray(eval_pred.label_ids)
    y_pred = np.asarray(logits).argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


# =============================================================================
# Execução de um fold
# =============================================================================

def run_fold(
    fold: int,
    df: pd.DataFrame,
    encodings: dict[str, torch.Tensor],
) -> None:
    print_header(f"FOLD {fold}")

    train_indices = df.index[df[FOLD_COLUMN] != fold].to_numpy()
    val_indices = df.index[df[FOLD_COLUMN] == fold].to_numpy()

    train_dataset = IndexedTokenizedDataset(
        encodings=encodings,
        labels=df["label_id"].to_numpy(),
        indices=train_indices,
    )
    val_dataset = IndexedTokenizedDataset(
        encodings=encodings,
        labels=df["label_id"].to_numpy(),
        indices=val_indices,
    )

    fold_output_dir = TRAINER_OUTPUT_DIR / f"fold_{fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification",
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),

        # Mesma configuração congelada do BERTimbau 256.
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,

        fp16=FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,

        # ÚNICA variável metodológica nova deste experimento.
        label_smoothing_factor=LABEL_SMOOTHING_FACTOR,

        # Avaliação ao fim de cada época, sem early stopping.
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,

        seed=SEED,
        data_seed=SEED,

        # Windows / reprodutibilidade operacional.
        dataloader_num_workers=0,

        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    start = time.perf_counter()

    try:
        trainer.train()

        # Predict é necessário para construir OOF; suas métricas servem como
        # avaliação final do fold, evitando uma terceira passada redundante.
        prediction_output = trainer.predict(val_dataset)

        elapsed_seconds = time.perf_counter() - start
        peak_vram_gb = get_peak_vram_gb()

        logits = prediction_output.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        probabilities = softmax_numpy(logits)
        y_true = df.loc[val_indices, "label_id"].to_numpy(dtype=np.int64)
        y_pred = probabilities.argmax(axis=1).astype(np.int64)

        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

        # No predict(), o Trainer prefixa a loss como test_loss.
        eval_loss = float(
            prediction_output.metrics.get("test_loss", np.nan)
        )

        # ---------------------------------------------------------------------
        # OOF do fold: salvar ANTES de marcar o fold como concluído no CV.
        # ---------------------------------------------------------------------
        val_rows = df.loc[val_indices].copy().reset_index(drop=True)

        oof_fold = pd.DataFrame(
            {
                ROW_ID_COLUMN: val_rows[ROW_ID_COLUMN].astype(int),
                FOLD_COLUMN: fold,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_true_label": [ID2LABEL[int(x)] for x in y_true],
                "y_pred_label": [ID2LABEL[int(x)] for x in y_pred],
                "prob_c1": probabilities[:, LABEL2ID["c1"]],
                "prob_c234": probabilities[:, LABEL2ID["c234"]],
                "prob_c5": probabilities[:, LABEL2ID["c5"]],
                TEXT_COLUMN: val_rows[TEXT_COLUMN],
                "duplicate_group_size": val_rows["duplicate_group_size"].astype(int),
                "duplicate_unique_labels": val_rows["duplicate_unique_labels"].astype(int),
                "is_duplicate": val_rows["is_duplicate"].astype(bool),
                "duplicate_conflict": val_rows["duplicate_conflict"].astype(bool),
            }
        )

        OOF_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
        oof_fold_path = OOF_FOLDS_DIR / f"fold_{fold}.csv"
        safe_to_csv(oof_fold, oof_fold_path)

        # ---------------------------------------------------------------------
        # Histórico por época / logs.
        # ---------------------------------------------------------------------
        history_rows: list[dict[str, Any]] = []
        for log in trainer.state.log_history:
            row = {
                FOLD_COLUMN: fold,
                "label_smoothing_factor": LABEL_SMOOTHING_FACTOR,
            }
            row.update(log)
            history_rows.append(row)

        replace_history_for_fold(fold, history_rows)

        # ---------------------------------------------------------------------
        # Resultado consolidado do fold.
        # ---------------------------------------------------------------------
        record = {
            FOLD_COLUMN: fold,
            "status": "ok",
            "error": "",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "eval_loss": eval_loss,
            "time_seconds": elapsed_seconds,
            "time_minutes": elapsed_seconds / 60.0,
            "vram_peak_gb": peak_vram_gb,
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_train_batch_size": (
                TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            ),
            "weight_decay": WEIGHT_DECAY,
            "fp16": FP16,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "label_smoothing_factor": LABEL_SMOOTHING_FACTOR,
            "seed": SEED,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }

        upsert_cv_record(record)

        print(
            f"Fold {fold} concluído | "
            f"Accuracy={accuracy:.6f} | "
            f"Macro-F1={macro_f1:.6f} | "
            f"eval_loss={eval_loss:.6f} | "
            f"tempo={elapsed_seconds / 60.0:.2f} min | "
            f"VRAM pico={peak_vram_gb:.2f} GB"
        )

    except Exception as exc:
        elapsed_seconds = time.perf_counter() - start
        peak_vram_gb = get_peak_vram_gb()
        trace = traceback.format_exc()

        record = {
            FOLD_COLUMN: fold,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "eval_loss": np.nan,
            "time_seconds": elapsed_seconds,
            "time_minutes": elapsed_seconds / 60.0,
            "vram_peak_gb": peak_vram_gb,
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "epochs": NUM_EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "effective_train_batch_size": (
                TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
            ),
            "weight_decay": WEIGHT_DECAY,
            "fp16": FP16,
            "gradient_checkpointing": GRADIENT_CHECKPOINTING,
            "label_smoothing_factor": LABEL_SMOOTHING_FACTOR,
            "seed": SEED,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }

        upsert_cv_record(record)

        error_path = RESULTS_DIR / f"bertimbau_label_smoothing_256_fold_{fold}_error.txt"
        error_path.write_text(trace, encoding="utf-8")

        print(f"\nERRO no fold {fold}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Traceback salvo em: {error_path}", file=sys.stderr)

        # Prosseguir para os próximos folds mantém o comportamento de resume.
    finally:
        del trainer
        del model
        del train_dataset
        del val_dataset
        cleanup_cuda()


# =============================================================================
# Consolidação
# =============================================================================

def consolidate_results(df: pd.DataFrame) -> None:
    print_header("RESUMO DE CROSS-VALIDATION")

    if not CV_PATH.exists():
        print("Nenhum resultado de CV encontrado.")
        return

    cv = pd.read_csv(CV_PATH)
    ok = (
        cv[cv["status"].astype(str).str.lower() == "ok"]
        .sort_values(FOLD_COLUMN)
        .reset_index(drop=True)
    )

    if ok.empty:
        print("Nenhum fold concluído com sucesso.")
        return

    print(
        ok[
            [
                FOLD_COLUMN,
                "accuracy",
                "macro_f1",
                "eval_loss",
                "time_minutes",
                "vram_peak_gb",
            ]
        ].to_string(index=False)
    )

    acc_mean = float(ok["accuracy"].mean())
    acc_std = float(ok["accuracy"].std(ddof=1)) if len(ok) > 1 else np.nan

    f1_mean = float(ok["macro_f1"].mean())
    f1_std = float(ok["macro_f1"].std(ddof=1)) if len(ok) > 1 else np.nan

    loss_mean = float(ok["eval_loss"].mean())
    loss_std = float(ok["eval_loss"].std(ddof=1)) if len(ok) > 1 else np.nan

    summary_rows = [
        {
            "metric": "accuracy",
            "mean": acc_mean,
            "std": acc_std,
            "n_folds": len(ok),
        },
        {
            "metric": "macro_f1",
            "mean": f1_mean,
            "std": f1_std,
            "n_folds": len(ok),
        },
        {
            "metric": "eval_loss",
            "mean": loss_mean,
            "std": loss_std,
            "n_folds": len(ok),
        },
    ]

    safe_to_csv(pd.DataFrame(summary_rows), SUMMARY_PATH)

    print(
        f"\nAccuracy : {acc_mean:.6f} ± {acc_std:.6f}\n"
        f"Macro-F1 : {f1_mean:.6f} ± {f1_std:.6f}\n"
        f"Eval loss: {loss_mean:.6f} ± {loss_std:.6f}"
    )

    # -------------------------------------------------------------------------
    # Só gera OOF completo quando todos os cinco folds realmente existem.
    # -------------------------------------------------------------------------
    expected_folds = set(range(N_FOLDS))
    successful_folds = set(ok[FOLD_COLUMN].astype(int).tolist())

    oof_fold_paths = {
        fold: OOF_FOLDS_DIR / f"fold_{fold}.csv"
        for fold in range(N_FOLDS)
    }

    all_oof_exist = all(path.exists() for path in oof_fold_paths.values())

    if successful_folds == expected_folds and all_oof_exist:
        oof_parts = [
            pd.read_csv(oof_fold_paths[fold])
            for fold in range(N_FOLDS)
        ]
        oof = pd.concat(oof_parts, ignore_index=True)
        oof = oof.sort_values(ROW_ID_COLUMN).reset_index(drop=True)

        if len(oof) != len(df):
            raise ValueError(
                f"OOF agregado possui {len(oof)} linhas; esperado {len(df)}."
            )

        if oof[ROW_ID_COLUMN].duplicated().any():
            raise ValueError("OOF agregado contém row_id duplicado.")

        expected_row_ids = set(df[ROW_ID_COLUMN].astype(int))
        observed_row_ids = set(oof[ROW_ID_COLUMN].astype(int))
        if observed_row_ids != expected_row_ids:
            raise ValueError("OOF agregado não cobre exatamente todos os row_id.")

        safe_to_csv(oof, OOF_PATH)

        y_true = oof["y_true"].to_numpy(dtype=np.int64)
        y_pred = oof["y_pred"].to_numpy(dtype=np.int64)

        oof_accuracy = float(accuracy_score(y_true, y_pred))
        oof_macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )

        print_header("OOF COMPLETO")
        print(f"Accuracy OOF : {oof_accuracy:.6f}")
        print(f"Macro-F1 OOF : {oof_macro_f1:.6f}")

        print("\nF1 por classe:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                target_names=LABELS,
                digits=6,
                zero_division=0,
            )
        )

        print("Matriz de confusão [c1, c234, c5]:")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

        # Diagnóstico simples dos grupos de duplicatas.
        unique_mask = ~oof["is_duplicate"].astype(bool)
        consistent_mask = (
            oof["is_duplicate"].astype(bool)
            & ~oof["duplicate_conflict"].astype(bool)
        )
        conflict_mask = oof["duplicate_conflict"].astype(bool)

        print("\nDiagnóstico OOF por tipo de texto:")
        for name, mask in [
            ("Únicos", unique_mask),
            ("Duplicatas consistentes", consistent_mask),
            ("Duplicatas conflitantes", conflict_mask),
        ]:
            subset = oof.loc[mask]
            if subset.empty:
                continue

            acc = accuracy_score(subset["y_true"], subset["y_pred"])
            mf1 = f1_score(
                subset["y_true"],
                subset["y_pred"],
                average="macro",
                zero_division=0,
            )

            print(
                f"- {name}: n={len(subset)} | "
                f"Accuracy={acc:.6f} | Macro-F1={mf1:.6f}"
            )

        # Acrescenta métricas OOF ao summary sem apagar média/desvio dos folds.
        oof_summary = pd.DataFrame(
            [
                {
                    "metric": "oof_accuracy",
                    "mean": oof_accuracy,
                    "std": np.nan,
                    "n_folds": N_FOLDS,
                },
                {
                    "metric": "oof_macro_f1",
                    "mean": oof_macro_f1,
                    "std": np.nan,
                    "n_folds": N_FOLDS,
                },
            ]
        )

        summary = pd.read_csv(SUMMARY_PATH)
        summary = summary[
            ~summary["metric"].isin(["oof_accuracy", "oof_macro_f1"])
        ]
        summary = pd.concat([summary, oof_summary], ignore_index=True)
        safe_to_csv(summary, SUMMARY_PATH)

    else:
        missing = sorted(expected_folds - successful_folds)
        print(
            "\nOOF completo ainda não foi agregado porque nem todos os folds "
            f"estão concluídos. Folds faltantes/sem sucesso: {missing}"
        )

    # -------------------------------------------------------------------------
    # Comparação metodológica com referências já congeladas.
    # -------------------------------------------------------------------------
    print_header("COMPARAÇÃO COM REFERÊNCIAS")

    print(
        f"Label smoothing 0.1 | "
        f"Accuracy={acc_mean:.6f} | Macro-F1={f1_mean:.6f}"
    )

    for name, ref in REFERENCE_RESULTS.items():
        delta_acc_pp = (acc_mean - ref["accuracy"]) * 100.0
        delta_f1_pp = (f1_mean - ref["macro_f1"]) * 100.0

        print(
            f"{name:20s} | "
            f"Acc={ref['accuracy']:.6f} "
            f"(Δ={delta_acc_pp:+.3f} p.p.) | "
            f"Macro-F1={ref['macro_f1']:.6f} "
            f"(Δ={delta_f1_pp:+.3f} p.p.)"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OOF_FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    warn_if_version_differs()

    print_header("BERTIMBAU 256 + GLOBAL LABEL SMOOTHING")
    print(f"Projeto: {PROJECT_ROOT}")
    print(f"Transformers: {transformers.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(
        "\nConfiguração:\n"
        f"  model={MODEL_NAME}\n"
        f"  max_length={MAX_LENGTH}\n"
        f"  learning_rate={LEARNING_RATE}\n"
        f"  epochs={NUM_EPOCHS}\n"
        f"  batch físico={TRAIN_BATCH_SIZE}\n"
        f"  gradient_accumulation={GRADIENT_ACCUMULATION_STEPS}\n"
        f"  batch efetivo={TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}\n"
        f"  weight_decay={WEIGHT_DECAY}\n"
        f"  fp16={FP16}\n"
        f"  gradient_checkpointing={GRADIENT_CHECKPOINTING}\n"
        f"  label_smoothing_factor={LABEL_SMOOTHING_FACTOR}\n"
        f"  seed={SEED}"
    )

    df = load_data_with_folds()

    print(
        f"\nDataset carregado: {len(df)} instâncias | "
        f"{df[TEXT_COLUMN].nunique()} textos únicos"
    )
    print("Tamanhos dos folds:")
    print(df[FOLD_COLUMN].value_counts().sort_index().to_string())

    # Tokenização idêntica ao BERT 256 convencional:
    # preserva o início e trunca o restante quando excede max_length.
    print("\nCarregando BertTokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizando todo o dataset uma única vez...")
    encodings = tokenizer(
        df[TEXT_COLUMN].tolist(),
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    completed_folds = load_completed_folds()
    if completed_folds:
        print(f"\nResume ativo. Folds já concluídos: {sorted(completed_folds)}")

    for fold in range(N_FOLDS):
        if fold in completed_folds:
            print(f"\nFold {fold}: já concluído, pulando.")
            continue

        run_fold(
            fold=fold,
            df=df,
            encodings=encodings,
        )

    consolidate_results(df)


if __name__ == "__main__":
    main()
