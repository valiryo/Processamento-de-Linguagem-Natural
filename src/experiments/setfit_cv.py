from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import set_seed


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT_DIR / "data" / "splits" / "folds.csv"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "setfit_cv.csv"
EXPERIMENTS_PATH = RESULTS_DIR / "setfit_experiments.csv"
OUTPUT_DIR = ROOT_DIR / "output" / "setfit"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
FOLDS = [0, 1, 2, 3, 4]
SEED = 42


def calcular_metricas(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def carregar_dados() -> pd.DataFrame:
    train = pd.read_excel(TRAIN_PATH).reset_index(drop=True)
    folds = pd.read_csv(FOLDS_PATH).reset_index(drop=True)

    if len(train) != 20092:
        raise ValueError(f"Esperadas 20092 linhas em train.xlsx; encontrado {len(train)}.")
    if len(folds) != len(train):
        raise ValueError("train.xlsx e folds.csv possuem números diferentes de linhas.")
    if set(folds.columns) < {"row_index", "fold"}:
        raise ValueError("folds.csv precisa conter as colunas row_index e fold.")
    if not np.array_equal(folds["row_index"].to_numpy(), np.arange(len(train))):
        raise ValueError("folds.csv não corresponde ao índice original de train.xlsx.")
    if sorted(folds["fold"].astype(int).unique().tolist()) != FOLDS:
        raise ValueError("folds.csv deve conter exatamente os folds 0, 1, 2, 3 e 4.")
    if {"resp_text", "clarity"} - set(train.columns):
        raise ValueError("train.xlsx precisa conter resp_text e clarity.")
    if train["resp_text"].isna().any() or train["clarity"].isna().any():
        raise ValueError("resp_text e clarity não podem conter valores ausentes.")
    if not set(train["clarity"].unique()).issubset(LABEL2ID):
        raise ValueError(f"Rótulos inesperados: {sorted(set(train['clarity']) - set(LABEL2ID))}.")

    train["fold"] = folds["fold"].astype(int).to_numpy()
    return train


def validar_sem_leakage(train: pd.DataFrame, fold: int) -> None:
    training = train[train["fold"] != fold]
    validation = train[train["fold"] == fold]
    overlap = set(training["resp_text"]) & set(validation["resp_text"])
    if overlap:
        raise ValueError(f"Leakage no fold {fold}: {len(overlap)} textos aparecem nos dois conjuntos.")


def criar_dataset(data: pd.DataFrame) -> Dataset:
    return Dataset.from_dict(
        {
            "text": data["resp_text"].astype(str).tolist(),
            "label": data["clarity"].map(LABEL2ID).astype("int64").tolist(),
        }
    )


def executar_fold(
    train: pd.DataFrame,
    fold: int,
    max_length: int,
    learning_rate: float,
    num_epochs: float,
    batch_size: int,
    sampling_strategy: str,
    num_iterations: int,
    use_amp: bool,
    config_id: str,
) -> dict[str, object]:
    validar_sem_leakage(train, fold)
    train_data = train[train["fold"] != fold]
    eval_data = train[train["fold"] == fold]
    started = time.perf_counter()
    row: dict[str, object] = {
        "config_id": config_id,
        "model_name": MODEL_NAME,
        "fold": fold,
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "max_length": max_length,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "sampling_strategy": sampling_strategy,
        "num_iterations": num_iterations,
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "time_minutes": np.nan,
        "status": "error",
        "error": "",
    }

    try:
        model = SetFitModel.from_pretrained(
            MODEL_NAME,
            labels=list(LABEL2ID),
        )
        if isinstance(model.model_head, LogisticRegression):
            model.model_head.set_params(max_iter=1000)
        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / config_id / f"fold_{fold}"),
            batch_size=batch_size,
            num_epochs=num_epochs,
            sampling_strategy=sampling_strategy,
            num_iterations=num_iterations,
            body_learning_rate=learning_rate,
            use_amp=use_amp,
            max_length=max_length,
            seed=SEED,
            report_to="none",
            save_strategy="no",
            show_progress_bar=True,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=criar_dataset(train_data),
            eval_dataset=criar_dataset(eval_data),
            metric=calcular_metricas,
        )
        trainer.train()
        raw_predictions = np.asarray(
            model.predict(
                eval_data["resp_text"].astype(str).tolist()
            )
        )
        predictions = np.asarray(
            [LABEL2ID[str(label)] for label in raw_predictions],
            dtype=int,
        )
        metrics = calcular_metricas(
            predictions,
            eval_data["clarity"].map(LABEL2ID).to_numpy(),
        )
        row.update(metrics)
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Fold {fold} falhou: {row['error']}")
    finally:
        row["time_minutes"] = (time.perf_counter() - started) / 60
        gc.collect()

    return row


def salvar_resultados(rows: list[dict[str, object]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(rows)
    if RESULTS_PATH.exists():
        previous = pd.read_csv(RESULTS_PATH)
        current = pd.concat([previous, current], ignore_index=True)
    current = current.drop_duplicates(
        subset=["config_id", "fold"],
        keep="last",
    )
    current.to_csv(RESULTS_PATH, index=False)

    successful = current[current["status"] == "ok"]
    summary = (
        successful.groupby("config_id", as_index=False)
        .agg(
            model_name=("model_name", "first"),
            folds=("fold", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            time_minutes_total=("time_minutes", "sum"),
        )
    )
    summary.to_csv(EXPERIMENTS_PATH, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV SetFit usando folds congelados.")
    parser.add_argument("--all-folds", action="store_true", help="Executa os cinco folds; por padrão executa apenas o smoke test do fold 0.")
    parser.add_argument("--fold", type=int, choices=FOLDS, default=0, help="Fold único para smoke test.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampling-strategy", choices=["oversampling", "undersampling", "unique"], default="oversampling")
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--config-id", default="minilm_l256_lr2e-5_ep1_bs16_iter1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    train = carregar_dados()
    folds = FOLDS if args.all_folds else [args.fold]
    print(f"Instâncias: {len(train)} | Folds encontrados: {FOLDS} | Executando: {folds}")
    rows = [
        executar_fold(
            train=train,
            fold=fold,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            sampling_strategy=args.sampling_strategy,
            num_iterations=args.num_iterations,
            use_amp=not args.no_amp,
            config_id=args.config_id,
        )
        for fold in folds
    ]
    salvar_resultados(rows)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()