from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import transformers
from transformers import AutoConfig, AutoTokenizer


MODEL_NAME = "PORTULAN/albertina-100m-portuguese-ptbr-encoder"
MAX_LENGTHS = (128, 256, 384, 512)
EXPECTED_CLASSES = ("c1", "c234", "c5")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "train.xlsx"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "results" / "albertina_tokenization_summary.csv"
DEFAULT_LENGTHS_OUTPUT = PROJECT_ROOT / "results" / "albertina_token_lengths.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analisa a tokenizacao da Albertina 100M PT-BR no train.xlsx "
            "sem treinar nenhum modelo."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Arquivo de treino. Padrao: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help=f"CSV resumo. Padrao: {DEFAULT_SUMMARY_OUTPUT}",
    )
    parser.add_argument(
        "--lengths-output",
        type=Path,
        default=DEFAULT_LENGTHS_OUTPUT,
        help=f"CSV com comprimento por linha. Padrao: {DEFAULT_LENGTHS_OUTPUT}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Quantidade de textos tokenizados por lote. Padrao: 256.",
    )

    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    df = pd.read_excel(path).copy()

    required_columns = {"resp_text", "clarity"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            "Colunas obrigatorias ausentes em train.xlsx: "
            + ", ".join(sorted(missing))
        )

    df["resp_text"] = df["resp_text"].fillna("").astype(str)
    df["clarity"] = df["clarity"].astype(str).str.strip()

    unexpected_classes = sorted(
        set(df["clarity"].unique()) - set(EXPECTED_CLASSES)
    )
    if unexpected_classes:
        raise ValueError(
            "Classes inesperadas em 'clarity': "
            f"{unexpected_classes}. Esperado: {list(EXPECTED_CLASSES)}"
        )

    # Mantem compatibilidade com os OOFs do projeto:
    # row_id = posicao zero-based no train.xlsx original.
    df["row_id"] = range(len(df))

    return df


def load_model_assets():
    print(f"Carregando config/tokenizer: {MODEL_NAME}")
    print(f"Transformers: {transformers.__version__}")

    config = AutoConfig.from_pretrained(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {getattr(config, 'hidden_size', None)}")
    print(f"Layers: {getattr(config, 'num_hidden_layers', None)}")
    print(
        "max_position_embeddings: "
        f"{getattr(config, 'max_position_embeddings', None)}"
    )
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Fast tokenizer: {tokenizer.is_fast}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"tokenizer.model_max_length: {tokenizer.model_max_length}")

    return config, tokenizer


def compute_token_lengths(
    texts: Iterable[str],
    tokenizer,
    batch_size: int,
) -> list[int]:
    texts = list(texts)
    lengths: list[int] = []
    total = len(texts)

    # add_special_tokens=True:
    # o comprimento representa exatamente a sequencia final que seria
    # entregue ao encoder antes de padding/truncamento.
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        batch_lengths = [
            len(input_ids)
            for input_ids in encoded["input_ids"]
        ]
        lengths.extend(batch_lengths)

        print(
            f"\rTokenizando: {end:>5}/{total} "
            f"({100.0 * end / total:6.2f}%)",
            end="",
            flush=True,
        )

    print()

    if len(lengths) != total:
        raise RuntimeError(
            f"Quantidade de comprimentos ({len(lengths)}) diferente "
            f"da quantidade de textos ({total})."
        )

    return lengths


def length_statistics(lengths: pd.Series) -> dict[str, float | int]:
    if lengths.empty:
        raise ValueError(
            "Nao e possivel calcular estatisticas de um subconjunto vazio."
        )

    return {
        "n_tokens_mean": float(lengths.mean()),
        "n_tokens_median": float(lengths.median()),
        "n_tokens_p75": float(lengths.quantile(0.75)),
        "n_tokens_p90": float(lengths.quantile(0.90)),
        "n_tokens_p95": float(lengths.quantile(0.95)),
        "n_tokens_p99": float(lengths.quantile(0.99)),
        "n_tokens_max": int(lengths.max()),
    }


def build_summary(
    df: pd.DataFrame,
    config,
    tokenizer,
) -> pd.DataFrame:
    groups: list[tuple[str, str, pd.DataFrame]] = [
        ("overall", "ALL", df),
        *[
            ("class", class_name, df[df["clarity"] == class_name])
            for class_name in EXPECTED_CLASSES
        ],
    ]

    rows: list[dict] = []

    for scope, class_name, subset in groups:
        lengths = subset["n_tokens"]
        stats = length_statistics(lengths)
        total = len(subset)

        for max_length in MAX_LENGTHS:
            fits_mask = lengths <= max_length
            truncated_mask = ~fits_mask

            fits_count = int(fits_mask.sum())
            truncated_count = int(truncated_mask.sum())

            discarded = (
                lengths.loc[truncated_mask] - max_length
            ).clip(lower=0)

            rows.append(
                {
                    "model_name": MODEL_NAME,
                    "transformers_version": transformers.__version__,
                    "model_type": config.model_type,
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_hidden_layers": getattr(
                        config,
                        "num_hidden_layers",
                        None,
                    ),
                    "max_position_embeddings": getattr(
                        config,
                        "max_position_embeddings",
                        None,
                    ),
                    "tokenizer_class": tokenizer.__class__.__name__,
                    "tokenizer_is_fast": bool(tokenizer.is_fast),
                    "tokenizer_vocab_size": int(tokenizer.vocab_size),
                    "tokenizer_model_max_length": (
                        tokenizer.model_max_length
                    ),
                    "scope": scope,
                    "clarity": class_name,
                    "max_length": max_length,
                    "total_count": total,
                    "fits_count": fits_count,
                    "fits_pct": 100.0 * fits_count / total,
                    "truncated_count": truncated_count,
                    "truncated_pct": 100.0 * truncated_count / total,
                    "discarded_tokens_total": int(discarded.sum()),
                    "discarded_tokens_mean_truncated": (
                        float(discarded.mean())
                        if truncated_count > 0
                        else 0.0
                    ),
                    **stats,
                }
            )

    return pd.DataFrame(rows)


def print_console_summary(summary: pd.DataFrame) -> None:
    columns = [
        "scope",
        "clarity",
        "max_length",
        "total_count",
        "fits_count",
        "fits_pct",
        "truncated_count",
        "truncated_pct",
        "discarded_tokens_total",
        "discarded_tokens_mean_truncated",
    ]

    printable = summary[columns].copy()

    for column in (
        "fits_pct",
        "truncated_pct",
        "discarded_tokens_mean_truncated",
    ):
        printable[column] = printable[column].map(
            lambda x: round(float(x), 4)
        )

    print("\nResumo de cobertura/truncamento:")
    print(printable.to_string(index=False))

    overall = summary[
        (summary["scope"] == "overall")
        & (summary["max_length"] == MAX_LENGTHS[0])
    ].iloc[0]

    print(
        "\nEstatisticas globais do comprimento tokenizado "
        "(incluindo tokens especiais):"
    )
    print(f"  media   : {overall['n_tokens_mean']:.2f}")
    print(f"  mediana : {overall['n_tokens_median']:.2f}")
    print(f"  p75     : {overall['n_tokens_p75']:.2f}")
    print(f"  p90     : {overall['n_tokens_p90']:.2f}")
    print(f"  p95     : {overall['n_tokens_p95']:.2f}")
    print(f"  p99     : {overall['n_tokens_p99']:.2f}")
    print(f"  maximo  : {int(overall['n_tokens_max'])}")


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser maior que zero.")

    print("=" * 78)
    print("ALBERTINA 100M PT-BR — ANALISE DE TOKENIZACAO")
    print("=" * 78)
    print(f"Projeto: {PROJECT_ROOT}")
    print(f"Entrada: {args.input}")

    df = load_dataset(args.input)

    print(f"Instancias carregadas: {len(df)}")
    print("Distribuicao de classes:")
    print(
        df["clarity"]
        .value_counts()
        .reindex(EXPECTED_CLASSES)
        .to_string()
    )

    config, tokenizer = load_model_assets()

    architecture_limit = getattr(
        config,
        "max_position_embeddings",
        None,
    )

    if architecture_limit is not None:
        invalid_lengths = [
            value
            for value in MAX_LENGTHS
            if value > architecture_limit
        ]
        if invalid_lengths:
            raise ValueError(
                "MAX_LENGTHS contem valores acima de "
                f"max_position_embeddings={architecture_limit}: "
                f"{invalid_lengths}"
            )

    print(
        "\nCalculando comprimento SEM truncamento para todas as "
        "instancias..."
    )

    df["n_tokens"] = compute_token_lengths(
        texts=df["resp_text"].tolist(),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )

    lengths_output = df[
        ["row_id", "clarity", "n_tokens"]
    ].copy()

    summary = build_summary(
        df=df,
        config=config,
        tokenizer=tokenizer,
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.lengths_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.summary_output,
        index=False,
        encoding="utf-8-sig",
    )
    lengths_output.to_csv(
        args.lengths_output,
        index=False,
        encoding="utf-8-sig",
    )

    print_console_summary(summary)

    print("\nArquivos salvos:")
    print(f"  {args.summary_output}")
    print(f"  {args.lengths_output}")

    print(
        "\nConcluido. Este script nao treina modelo, "
        "nao carrega folds e nao acessa test.xlsx."
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nExecucao interrompida pelo usuario.",
            file=sys.stderr,
        )
        raise SystemExit(130)
