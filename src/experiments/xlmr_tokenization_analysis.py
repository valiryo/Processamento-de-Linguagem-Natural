from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "train.xlsx"
RESULTS_DIR = ROOT / "results"

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TEXT_COL = "resp_text"
LABEL_COL = "clarity"

MAX_LENGTHS = [256, 384, 512]


# ============================================================
# MAIN
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dataset: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)

    texts = df[TEXT_COL].fillna("").astype(str).tolist()

    print(f"N exemplos: {len(df):,}")
    print(f"Modelo: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nTokenizando sem truncamento...")

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    lengths = np.array(
        [len(ids) for ids in encoded["input_ids"]],
        dtype=np.int32,
    )

    df_lengths = df[[LABEL_COL]].copy()
    df_lengths["n_tokens"] = lengths

    print("\nEstatísticas gerais:")
    print(f"mean   = {lengths.mean():.2f}")
    print(f"median = {np.median(lengths):.2f}")
    print(f"p75    = {np.percentile(lengths, 75):.2f}")
    print(f"p90    = {np.percentile(lengths, 90):.2f}")
    print(f"p95    = {np.percentile(lengths, 95):.2f}")
    print(f"p99    = {np.percentile(lengths, 99):.2f}")
    print(f"max    = {lengths.max()}")

    total_tokens = int(lengths.sum())

    summary_rows = []
    class_rows = []

    for max_length in MAX_LENGTHS:
        truncated_mask = lengths > max_length

        n_truncated = int(truncated_mask.sum())
        n_fit = len(lengths) - n_truncated

        discarded = np.maximum(lengths - max_length, 0)
        tokens_discarded = int(discarded.sum())

        summary_rows.append({
            "model": MODEL_NAME,
            "max_length": max_length,
            "n_samples": len(lengths),
            "n_fit": n_fit,
            "pct_fit": 100 * n_fit / len(lengths),
            "n_truncated": n_truncated,
            "pct_truncated": 100 * n_truncated / len(lengths),
            "total_tokens": total_tokens,
            "tokens_discarded": tokens_discarded,
            "pct_tokens_discarded": (
                100 * tokens_discarded / total_tokens
            ),
            "mean_tokens": lengths.mean(),
            "median_tokens": np.median(lengths),
            "p75_tokens": np.percentile(lengths, 75),
            "p90_tokens": np.percentile(lengths, 90),
            "p95_tokens": np.percentile(lengths, 95),
            "p99_tokens": np.percentile(lengths, 99),
            "max_tokens": lengths.max(),
        })

        for label, group in df_lengths.groupby(LABEL_COL):
            group_lengths = group["n_tokens"].to_numpy()

            group_truncated = group_lengths > max_length

            group_discarded = np.maximum(
                group_lengths - max_length,
                0,
            )

            class_rows.append({
                "model": MODEL_NAME,
                "max_length": max_length,
                "class": label,
                "n_samples": len(group),
                "n_truncated": int(group_truncated.sum()),
                "pct_truncated": (
                    100 * group_truncated.mean()
                ),
                "tokens_discarded": int(
                    group_discarded.sum()
                ),
            })

    summary_df = pd.DataFrame(summary_rows)
    class_df = pd.DataFrame(class_rows)

    summary_path = (
        RESULTS_DIR / "xlmr_tokenization_summary.csv"
    )

    class_path = (
        RESULTS_DIR / "xlmr_tokenization_by_class.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    class_df.to_csv(
        class_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)

    cols = [
        "max_length",
        "n_fit",
        "pct_fit",
        "n_truncated",
        "pct_truncated",
        "tokens_discarded",
        "pct_tokens_discarded",
    ]

    print(
        summary_df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print("\n" + "=" * 80)
    print("TRUNCAMENTO POR CLASSE")
    print("=" * 80)

    print(
        class_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(f"\nSalvo em:\n{summary_path}")
    print(class_path)


if __name__ == "__main__":
    main()