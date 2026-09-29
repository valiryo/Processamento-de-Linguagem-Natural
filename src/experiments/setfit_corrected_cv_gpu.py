from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from setfit import SetFitModel, Trainer, TrainingArguments
from transformers import set_seed


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT_DIR / "data" / "splits" / "folds.csv"

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "setfit_corrected_cv.csv"
EXPERIMENTS_PATH = RESULTS_DIR / "setfit_corrected_experiments.csv"
OOF_PATH = RESULTS_DIR / "setfit_corrected_oof.csv"
OUTPUT_DIR = ROOT_DIR / "output" / "setfit_corrected"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
FOLDS = [0, 1, 2, 3, 4]
SEED = 42


def calcular_metricas(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    per_class = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_c1": float(per_class[0]),
        "f1_c234": float(per_class[1]),
        "f1_c5": float(per_class[2]),
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

    unexpected = set(train["clarity"].unique()) - set(LABEL2ID)
    if unexpected:
        raise ValueError(f"Rótulos inesperados: {sorted(unexpected)}.")

    train["row_id"] = np.arange(len(train))
    train["fold"] = folds["fold"].astype(int).to_numpy()
    return train


def validar_sem_leakage(train: pd.DataFrame, fold: int) -> None:
    training = train[train["fold"] != fold]
    validation = train[train["fold"] == fold]
    overlap = set(training["resp_text"]) & set(validation["resp_text"])
    if overlap:
        raise ValueError(
            f"Leakage no fold {fold}: {len(overlap)} textos aparecem nos dois conjuntos."
        )


def criar_dataset(data: pd.DataFrame) -> Dataset:
    return Dataset.from_dict(
        {
            "text": data["resp_text"].astype(str).tolist(),
            "label": data["clarity"].map(LABEL2ID).astype("int64").tolist(),
        }
    )


def converter_classe_para_id(value: Any) -> int:
    if isinstance(value, str):
        if value not in LABEL2ID:
            raise ValueError(f"Classe inesperada retornada pelo modelo: {value!r}")
        return LABEL2ID[value]

    value_int = int(value)
    if value_int not in ID2LABEL:
        raise ValueError(f"ID de classe inesperado retornado pelo modelo: {value_int}")
    return value_int


def probabilidades_em_ordem_canonica(
    model: SetFitModel,
    raw_probabilities: np.ndarray,
) -> np.ndarray:
    """Retorna colunas sempre na ordem [c1, c234, c5]."""
    probs = np.asarray(raw_probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError(f"Shape inesperado de predict_proba: {probs.shape}")

    classes = getattr(model.model_head, "classes_", None)
    if classes is None:
        source_ids = np.array([0, 1, 2], dtype=int)
    else:
        source_ids = np.asarray(
            [converter_classe_para_id(v) for v in classes],
            dtype=int,
        )

    if sorted(source_ids.tolist()) != [0, 1, 2]:
        raise ValueError(f"Classes inesperadas no head: {source_ids.tolist()}")

    canonical = np.zeros_like(probs, dtype=float)
    for source_col, class_id in enumerate(source_ids):
        canonical[:, class_id] = probs[:, source_col]

    if not np.allclose(canonical.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("As probabilidades não somam aproximadamente 1 por linha.")

    return canonical


def gerar_oof(
    model: SetFitModel,
    eval_data: pd.DataFrame,
    fold: int,
    config_id: str,
    predict_batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    texts = eval_data["resp_text"].astype(str).tolist()

    raw_predictions = model.predict(
        texts,
        batch_size=predict_batch_size,
        as_numpy=True,
        use_labels=False,
        show_progress_bar=True,
    )
    predictions = np.asarray(
        [converter_classe_para_id(v) for v in raw_predictions],
        dtype=int,
    )

    raw_probabilities = model.predict_proba(
        texts,
        batch_size=predict_batch_size,
        as_numpy=True,
        show_progress_bar=True,
    )
    probabilities = probabilidades_em_ordem_canonica(model, raw_probabilities)

    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise ValueError("predict() e argmax(predict_proba()) produziram classes diferentes.")

    y_true = eval_data["clarity"].map(LABEL2ID).astype(int).to_numpy()

    oof = pd.DataFrame(
        {
            "config_id": config_id,
            "row_id": eval_data["row_id"].astype(int).to_numpy(),
            "fold": fold,
            "y_true_id": y_true,
            "y_pred_id": predictions,
            "y_true": [ID2LABEL[int(v)] for v in y_true],
            "y_pred": [ID2LABEL[int(v)] for v in predictions],
            "prob_c1": probabilities[:, 0],
            "prob_c234": probabilities[:, 1],
            "prob_c5": probabilities[:, 2],
            "correct": predictions == y_true,
            "resp_text": eval_data["resp_text"].astype(str).to_numpy(),
        }
    )
    return predictions, oof


def executar_fold(
    train: pd.DataFrame,
    fold: int,
    max_length: int,
    learning_rate: float,
    num_epochs: int,
    batch_size: int,
    predict_batch_size: int,
    sampling_strategy: str,
    num_iterations: int | None,
    use_amp: bool,
    config_id: str,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    validar_sem_leakage(train, fold)
    train_data = train[train["fold"] != fold].copy()
    eval_data = train[train["fold"] == fold].copy()
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
        "predict_batch_size": predict_batch_size,
        "sampling_strategy": sampling_strategy,
        "num_iterations": num_iterations,
        "use_amp": use_amp,
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "f1_c1": np.nan,
        "f1_c234": np.nan,
        "f1_c5": np.nan,
        "time_minutes": np.nan,
        "status": "error",
        "error": "",
    }

    model = None
    trainer = None
    oof = None

    try:
        set_seed(SEED)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA não está disponível. Este experimento foi configurado "
                "para rodar obrigatoriamente na GPU."
            )

        device = "cuda"

        model = SetFitModel.from_pretrained(
            MODEL_NAME,
            labels=["c1", "c234", "c5"],
            device=device,
        )

        # Garante explicitamente que o SentenceTransformer body está na GPU.
        model.to(device)

        print(f"CUDA disponível         : {torch.cuda.is_available()}")
        print(f"GPU                     : {torch.cuda.get_device_name(0)}")
        print(f"Dispositivo do modelo   : {model.device}")

        if isinstance(model.model_head, LogisticRegression):
            model.model_head.set_params(max_iter=1000, random_state=SEED)

        args = TrainingArguments(
            output_dir=str(OUTPUT_DIR / config_id / f"fold_{fold}"),
            batch_size=batch_size,
            num_epochs=num_epochs,
            sampling_strategy=sampling_strategy,
            # Limitamos explicitamente a geração de pares. Com ~16 mil
            # exemplos por fold, oversampling completo (num_iterations=None)
            # é impraticável porque a geração de pares ocorre na CPU antes
            # do fine-tuning do encoder na GPU.
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

        predictions, oof = gerar_oof(
            model=model,
            eval_data=eval_data,
            fold=fold,
            config_id=config_id,
            predict_batch_size=predict_batch_size,
        )

        y_true = eval_data["clarity"].map(LABEL2ID).astype(int).to_numpy()
        row.update(calcular_metricas(predictions, y_true))
        row["status"] = "ok"

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Fold {fold} falhou: {row['error']}")

    finally:
        row["time_minutes"] = (time.perf_counter() - started) / 60
        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return row, oof


def salvar_cv(rows: list[dict[str, object]]) -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    current = pd.DataFrame(rows)

    if RESULTS_PATH.exists():
        current = pd.concat([pd.read_csv(RESULTS_PATH), current], ignore_index=True)

    current = (
        current.drop_duplicates(subset=["config_id", "fold"], keep="last")
        .sort_values(["config_id", "fold"])
    )
    current.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    return current


def salvar_oof(oof: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    current = oof.copy()

    if OOF_PATH.exists():
        current = pd.concat([pd.read_csv(OOF_PATH), current], ignore_index=True)

    current = (
        current.drop_duplicates(subset=["config_id", "row_id"], keep="last")
        .sort_values(["config_id", "row_id"])
    )
    current.to_csv(OOF_PATH, index=False, encoding="utf-8-sig")


def salvar_resumo(cv: pd.DataFrame) -> None:
    successful = cv[cv["status"] == "ok"].copy()

    if successful.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            successful.groupby("config_id", as_index=False)
            .agg(
                model_name=("model_name", "first"),
                folds=("fold", "count"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                macro_f1_mean=("macro_f1", "mean"),
                macro_f1_std=("macro_f1", "std"),
                f1_c1_mean=("f1_c1", "mean"),
                f1_c234_mean=("f1_c234", "mean"),
                f1_c5_mean=("f1_c5", "mean"),
                time_minutes_total=("time_minutes", "sum"),
            )
        )

    summary.to_csv(EXPERIMENTS_PATH, index=False, encoding="utf-8-sig")


def salvar_resultado_fold(row: dict[str, object], oof: pd.DataFrame | None) -> None:
    cv = salvar_cv([row])
    if oof is not None:
        salvar_oof(oof)
    salvar_resumo(cv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CV SetFit corrigida usando folds congelados e salvando OOF completo."
    )
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--fold", type=int, choices=FOLDS, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[1e-5],
        help="Padrão: somente 1e-5, melhor LR do experimento anterior.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help=(
            "Número inteiro de épocas do SetFit. "
            "Não passe float aqui, pois a API atual espera int ou tupla."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--predict-batch-size", type=int, default=32)
    parser.add_argument(
        "--sampling-strategy",
        choices=["oversampling", "undersampling", "unique"],
        default="oversampling",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help=(
            "Número explícito de iterações para geração de pares. "
            "Padrão: 5. Evita a explosão combinatória de "
            "oversampling + num_iterations=None neste dataset grande."
        ),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--config-prefix",
        default="mpnet_l256_ep1_bs16_iter5_gpu",
    )
    return parser.parse_args()


def formatar_learning_rate(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def main() -> None:
    args = parse_args()

    if args.num_iterations is not None and args.num_iterations <= 0:
        raise ValueError("--num-iterations deve ser > 0 quando informado.")

    if args.num_iterations is not None:
        print(
            "INFO: num_iterations foi definido explicitamente. "
            "Isto limita a geração de pares e evita a explosão combinatória "
            "do oversampling completo."
        )

    set_seed(SEED)

    print(f"PyTorch                 : {torch.__version__}")
    print(f"CUDA disponível         : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA PyTorch            : {torch.version.cuda}")
        print(f"GPU                     : {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError(
            "torch.cuda.is_available() retornou False. "
            "Não iniciar SetFit em CPU."
        )

    train = carregar_dados()
    folds = FOLDS if args.all_folds else [args.fold]

    print("=" * 78)
    print("SETFIT CORRIGIDO — multilingual MPNet")
    print("=" * 78)
    print(f"Modelo             : {MODEL_NAME}")
    print(f"Instâncias         : {len(train)}")
    print(f"Folds              : {folds}")
    print(f"Learning rates     : {args.learning_rates}")
    print(f"max_length         : {args.max_length}")
    print(f"epochs             : {args.num_epochs}")
    print(f"batch_size         : {args.batch_size}")
    print(f"sampling_strategy  : {args.sampling_strategy}")
    print(f"num_iterations     : {args.num_iterations}")
    print(f"AMP                : {not args.no_amp}")
    print("=" * 78)

    executed_rows = []

    for learning_rate in args.learning_rates:
        iter_tag = "auto" if args.num_iterations is None else str(args.num_iterations)
        config_id = (
            f"{args.config_prefix}_lr{formatar_learning_rate(learning_rate)}_iter{iter_tag}"
        )

        print(f"\nConfiguração: {config_id}")

        for fold in folds:
            print(f"\n--- Fold {fold} ---")

            row, oof = executar_fold(
                train=train,
                fold=fold,
                max_length=args.max_length,
                learning_rate=learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                predict_batch_size=args.predict_batch_size,
                sampling_strategy=args.sampling_strategy,
                num_iterations=args.num_iterations,
                use_amp=not args.no_amp,
                config_id=config_id,
            )

            salvar_resultado_fold(row, oof)
            executed_rows.append(row)
            print(pd.DataFrame([row]).to_string(index=False))

    print("\n" + "=" * 78)
    print("RESULTADOS DESTA EXECUÇÃO")
    print("=" * 78)
    print(pd.DataFrame(executed_rows).to_string(index=False))
    print("\nArquivos gerados/atualizados:")
    print(f"  CV     : {RESULTS_PATH}")
    print(f"  Resumo : {EXPERIMENTS_PATH}")
    print(f"  OOF    : {OOF_PATH}")


if __name__ == "__main__":
    main()
