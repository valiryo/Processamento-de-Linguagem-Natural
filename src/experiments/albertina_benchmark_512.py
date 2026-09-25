from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "PORTULAN/albertina-100m-portuguese-ptbr-encoder"
LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.xlsx"
DEFAULT_FOLDS_PATH = PROJECT_ROOT / "data" / "splits" / "folds.csv"
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "albertina_benchmark_512.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "albertina_benchmark_512"


@dataclass
class BenchmarkResult:
    timestamp: str
    status: str
    error_type: str | None
    error_message: str | None

    model_name: str
    transformers_version: str
    torch_version: str
    python_version: str

    cuda_available: bool
    cuda_device: str | None
    cuda_capability: str | None

    fold: int
    train_rows: int
    max_length: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    gradient_checkpointing: bool
    fp16: bool
    max_steps: int
    seed: int

    train_runtime_seconds: float | None
    steps_per_second: float | None
    samples_per_second: float | None
    final_train_loss: float | None

    peak_vram_allocated_gb: float | None
    peak_vram_reserved_gb: float | None


class FixedGASTrainer(Trainer):
    """
    Workaround defensivo para ModernBERT + Trainer + gradient accumulation.

    ModernBertForSequenceClassification aceita **kwargs, mas sua loss de
    classificacao e calculada diretamente com CrossEntropyLoss(mean) e nao usa
    num_items_in_batch. O Trainer pode interpretar **kwargs como sinal de que o
    modelo faz a normalizacao do batch acumulado por conta propria.

    Forcar model_accepts_loss_kwargs=False faz o Trainer aplicar a divisao por
    gradient_accumulation_steps, que e o comportamento correto para esta loss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, list[Any]], labels: list[int]) -> None:
        if not labels:
            raise ValueError("Dataset de treino vazio.")

        n = len(labels)
        for key, values in encodings.items():
            if len(values) != n:
                raise ValueError(
                    f"Campo '{key}' possui {len(values)} itens, mas existem {n} labels."
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
            "Benchmark operacional da Albertina 100M PT-BR com max_length=512. "
            "Nao faz selecao cientifica nem avaliacao de validacao."
        )
    )

    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument(
        "--history-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "albertina_benchmark_512_history.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=100)

    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)

    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Ativa gradient checkpointing. Deixe desligado na primeira tentativa.",
    )
    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--bf16",
        action="store_true",
        help="Usa BF16 mixed precision (recomendado para o probe de estabilidade em Ampere+).",
    )
    precision_group.add_argument(
        "--no-fp16",
        action="store_true",
        help="Desativa mixed precision (FP32).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=256,
        help="Batch usado somente no pre-processamento/tokenizacao em CPU.",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = {
        "--max-length": args.max_length,
        "--max-steps": args.max_steps,
        "--batch-size": args.batch_size,
        "--gradient-accumulation": args.gradient_accumulation,
        "--tokenization-batch-size": args.tokenization_batch_size,
    }

    for name, value in positive_int_fields.items():
        if value <= 0:
            raise ValueError(f"{name} deve ser maior que zero.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate deve ser maior que zero.")

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
            "Nao encontrei identificador de linha nos folds. Esperava uma entre: "
            + ", ".join(row_candidates)
        )

    return row_col, fold_col


def load_train_fold(
    train_path: Path,
    folds_path: Path,
    held_out_fold: int,
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
            "Colunas obrigatorias ausentes no train.xlsx: "
            + ", ".join(sorted(missing))
        )

    row_col, fold_col = detect_fold_columns(folds)

    if folds[row_col].duplicated().any():
        raise ValueError(f"A coluna '{row_col}' dos folds possui IDs duplicados.")

    if len(folds) != len(train):
        raise ValueError(
            f"folds.csv possui {len(folds)} linhas, mas train.xlsx possui {len(train)}."
        )

    # O projeto historicamente usa a posição original de train.xlsx como identificador.
    # Esta coluna serve apenas para fazer o vínculo com os folds congelados.
    train["_row_index_internal"] = range(len(train))

    fold_map = folds[[row_col, fold_col]].rename(
        columns={row_col: "_fold_row_id", fold_col: "_fold"}
    )

    # Caso comum: row_index/row_id zero-based exatamente igual à posição do train.xlsx.
    expected_zero_based = set(range(len(train)))
    actual_ids = set(pd.to_numeric(fold_map["_fold_row_id"], errors="coerce").dropna().astype(int))

    if actual_ids == expected_zero_based:
        merged = train.merge(
            fold_map,
            left_on="_row_index_internal",
            right_on="_fold_row_id",
            how="left",
            validate="one_to_one",
        )
    else:
        # Fallback conservador: se os folds usam a mesma ordem e têm um ID diferente,
        # NÃO tentamos "adivinhar" o mapeamento.
        raise ValueError(
            f"A coluna '{row_col}' de folds.csv nao corresponde aos indices 0..{len(train)-1}. "
            "Por seguranca, o benchmark foi interrompido para nao usar folds errados."
        )

    if merged["_fold"].isna().any():
        raise ValueError("Algumas linhas de train.xlsx ficaram sem fold apos o merge.")

    merged["_fold"] = merged["_fold"].astype(int)

    available_folds = sorted(merged["_fold"].unique().tolist())
    if held_out_fold not in available_folds:
        raise ValueError(
            f"Fold {held_out_fold} inexistente. Folds encontrados: {available_folds}"
        )

    # Benchmark operacional no treino do fold escolhido.
    # Nenhum exemplo da validacao do fold entra no treinamento.
    train_fold = merged[merged["_fold"] != held_out_fold].copy()

    train_fold["resp_text"] = train_fold["resp_text"].fillna("").astype(str)
    train_fold["clarity"] = train_fold["clarity"].astype(str).str.strip()

    unexpected = sorted(set(train_fold["clarity"]) - set(LABEL2ID))
    if unexpected:
        raise ValueError(f"Classes inesperadas encontradas: {unexpected}")

    return train_fold


def tokenize_texts(
    texts: list[str],
    tokenizer,
    max_length: int,
    batch_size: int,
) -> dict[str, list[Any]]:
    pieces: dict[str, list[Any]] = {}

    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        batch = tokenizer(
            texts[start:end],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

        for key, values in batch.items():
            pieces.setdefault(key, []).extend(values)

        print(
            f"\rTokenizando treino: {end:>5}/{total} "
            f"({100.0 * end / total:6.2f}%)",
            end="",
            flush=True,
        )

    print()
    return pieces


def gb(value: int) -> float:
    return value / (1024**3)


def cuda_info() -> tuple[str | None, str | None]:
    if not torch.cuda.is_available():
        return None, None

    device = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return device, f"{major}.{minor}"


def append_result(path: Path, result: BenchmarkResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pd.DataFrame([asdict(result)])

    if path.exists():
        previous = pd.read_csv(path)
        row = pd.concat([previous, row], ignore_index=True)

    row.to_csv(path, index=False, encoding="utf-8-sig")


def save_run_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    bf16 = bool(args.bf16)
    fp16 = (not args.no_fp16) and (not bf16)

    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 foi solicitado, mas torch.cuda.is_bf16_supported() retornou False."
        )

    effective_batch = args.batch_size * args.gradient_accumulation

    print("=" * 72)
    print("ALBERTINA 100M PT-BR — BENCHMARK OPERACIONAL (GAS FIX)")
    print("=" * 72)
    print(f"Modelo                  : {MODEL_NAME}")
    print(f"Fold retido              : {args.fold}")
    print(f"max_length               : {args.max_length}")
    print(f"max_steps                : {args.max_steps}")
    print(f"batch fisico             : {args.batch_size}")
    print(f"gradient accumulation    : {args.gradient_accumulation}")
    print(f"batch efetivo            : {effective_batch}")
    print(f"learning rate            : {args.learning_rate}")
    print(f"weight decay             : {args.weight_decay}")
    print(f"FP16                     : {fp16}")
    print(f"BF16                     : {bf16}")
    print(f"BF16 suportado pela GPU  : {torch.cuda.is_bf16_supported()}")
    print(f"gradient checkpointing   : {args.gradient_checkpointing}")
    print(f"Transformers             : {transformers.__version__}")
    print(f"PyTorch                  : {torch.__version__}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nao esta disponivel. Este benchmark foi planejado para a RTX 3050."
        )

    cuda_device, cuda_capability = cuda_info()
    print(f"CUDA device              : {cuda_device}")
    print(f"CUDA capability          : {cuda_capability}")
    print("=" * 72)

    set_seed(args.seed)

    train_fold = load_train_fold(
        train_path=args.train,
        folds_path=args.folds,
        held_out_fold=args.fold,
    )

    print(f"Linhas usadas no treino do fold {args.fold}: {len(train_fold)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    encodings = tokenize_texts(
        texts=train_fold["resp_text"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.tokenization_batch_size,
    )

    labels = train_fold["clarity"].map(LABEL2ID).astype(int).tolist()
    dataset = EncodedTextDataset(encodings, labels)

    # Padding dinamico: cada batch e' preenchido apenas ate o maior exemplo daquele batch,
    # respeitando max_length=1024 como teto de truncamento.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="longest",
        return_tensors="pt",
    )

    print("Carregando modelo para classificacao...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        problem_type="single_label_classification",
    )

    print(f"classifier_pooling        : {getattr(model.config, 'classifier_pooling', None)}")
    print(f"problem_type              : {model.config.problem_type}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),

        # Transformers 5.x removeu overwrite_output_dir de TrainingArguments.
        # Como este benchmark usa save_strategy="no", nenhum checkpoint antigo
        # sera retomado ou sobrescrito automaticamente.
        # Benchmark: apenas treino, sem avaliacao e sem checkpoints.
        eval_strategy="no",
        save_strategy="no",

        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,

        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,

        report_to=[],
        seed=args.seed,
        data_seed=args.seed,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,

        remove_unused_columns=True,
        disable_tqdm=False,
    )

    trainer = FixedGASTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"model_accepts_loss_kwargs : {trainer.model_accepts_loss_kwargs}")
    print(f"max_grad_norm             : {training_args.max_grad_norm}")

    if trainer.model_accepts_loss_kwargs is not False:
        raise RuntimeError(
            "GAS fix nao foi aplicado: model_accepts_loss_kwargs deveria ser False."
        )

    metadata_path = args.output_dir / "benchmark_config.json"
    save_run_metadata(
        metadata_path,
        {
            "model_name": MODEL_NAME,
            "fold": args.fold,
            "train_rows": len(train_fold),
            "max_length": args.max_length,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "effective_batch_size": effective_batch,
            "model_accepts_loss_kwargs": trainer.model_accepts_loss_kwargs,
            "gas_normalization_fix": True,
            "max_grad_norm": training_args.max_grad_norm,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "fp16": fp16,
            "bf16": bf16,
            "classifier_pooling": getattr(model.config, "classifier_pooling", None),
            "problem_type": model.config.problem_type,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "cuda_device": cuda_device,
            "cuda_capability": cuda_capability,
        },
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    started = time.perf_counter()

    status = "success"
    error_type = None
    error_message = None
    train_runtime = None
    steps_per_second = None
    samples_per_second = None
    final_train_loss = None

    caught_exception: BaseException | None = None

    try:
        train_output = trainer.train()

        torch.cuda.synchronize()
        train_runtime = time.perf_counter() - started

        metrics = train_output.metrics

        # O Trainer normalmente fornece estas metricas.
        steps_per_second = metrics.get("train_steps_per_second")
        samples_per_second = metrics.get("train_samples_per_second")
        final_train_loss = metrics.get("train_loss")

    except BaseException as exc:
        # Registramos OOM/erro antes de propagar a excecao.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_runtime = time.perf_counter() - started
        status = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error"
        error_type = type(exc).__name__
        error_message = str(exc)[:2000]
        caught_exception = exc

    peak_allocated = gb(torch.cuda.max_memory_allocated())
    peak_reserved = gb(torch.cuda.max_memory_reserved())

    # Salva a trajetoria de treinamento (loss, grad_norm, learning_rate etc.)
    # para diagnosticar estabilidade numerica antes da CV completa.
    if trainer.state.log_history:
        history_df = pd.DataFrame(trainer.state.log_history)
        args.history_output.parent.mkdir(parents=True, exist_ok=True)
        history_df.to_csv(args.history_output, index=False, encoding="utf-8-sig")

    result = BenchmarkResult(
        timestamp=pd.Timestamp.now().isoformat(),
        status=status,
        error_type=error_type,
        error_message=error_message,

        model_name=MODEL_NAME,
        transformers_version=transformers.__version__,
        torch_version=torch.__version__,
        python_version=platform.python_version(),

        cuda_available=torch.cuda.is_available(),
        cuda_device=cuda_device,
        cuda_capability=cuda_capability,

        fold=args.fold,
        train_rows=len(train_fold),
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        effective_batch_size=effective_batch,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=fp16,
        max_steps=args.max_steps,
        seed=args.seed,

        train_runtime_seconds=train_runtime,
        steps_per_second=(
            float(steps_per_second) if steps_per_second is not None else None
        ),
        samples_per_second=(
            float(samples_per_second) if samples_per_second is not None else None
        ),
        final_train_loss=(
            float(final_train_loss) if final_train_loss is not None else None
        ),

        peak_vram_allocated_gb=peak_allocated,
        peak_vram_reserved_gb=peak_reserved,
    )

    append_result(args.results, result)

    print("\n" + "=" * 72)
    print("RESULTADO DO BENCHMARK")
    print("=" * 72)
    print(f"status                   : {status}")
    print(f"tempo treino             : {train_runtime:.2f} s")
    print(f"VRAM max allocated       : {peak_allocated:.3f} GB")
    print(f"VRAM max reserved        : {peak_reserved:.3f} GB")

    if final_train_loss is not None:
        print(f"train loss               : {final_train_loss:.6f}")

    if steps_per_second is not None:
        print(f"steps/s                  : {float(steps_per_second):.4f}")

    print(f"resultado salvo em       : {args.results}")
    print(f"historico salvo em       : {args.history_output}")
    print("=" * 72)

    if caught_exception is not None:
        print(
            f"\nBenchmark terminou com {status}: "
            f"{error_type}: {error_message}",
            file=sys.stderr,
        )
        raise caught_exception

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nExecucao interrompida pelo usuario.", file=sys.stderr)
        raise SystemExit(130)
