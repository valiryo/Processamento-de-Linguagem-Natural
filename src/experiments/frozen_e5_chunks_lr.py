"""
Embeddings congelados por chunks (multilingual-e5-base) + LogisticRegression.

Hipótese: um encoder congelado lendo o texto INTEIRO (chunks, sem truncamento)
+ classificador linear comete erros diferentes dos Transformers fine-tunados e
do TF-IDF, sendo barato e complementar ao Ensemble V3.

Uso:
    python src/experiments/frozen_e5_chunks_lr.py
    python src/experiments/frozen_e5_chunks_lr.py --model intfloat/multilingual-e5-large --batch-size 8

Saídas (results/):
    frozen_e5_chunks_lr_cv.csv
    frozen_e5_chunks_lr_oof.csv
    frozen_e5_chunks_lr_summary.csv
    frozen_e5_chunks_lr_vs_v3.csv
Cache: results/cache/frozen_e5_chunks_<modelo>.npy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer, set_seed

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT / "data" / "splits" / "folds.csv"
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
V3_OOF = RESULTS / "ensemble_majority_5_v3_norberto_oof.csv"

LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SEED = 42
EXPECTED_N = 20_092


# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    df = pd.read_excel(TRAIN_PATH).reset_index(drop=True)
    folds = pd.read_csv(FOLDS_PATH)

    if len(df) != len(folds) or len(df) != EXPECTED_N:
        raise ValueError("train.xlsx e folds.csv incompatíveis.")
    if not np.array_equal(folds["row_index"].to_numpy(), np.arange(len(df))):
        raise ValueError("folds.csv não corresponde ao índice de train.xlsx.")

    df["resp_text"] = df["resp_text"].fillna("").astype(str)
    df["clarity"] = df["clarity"].astype(str).str.strip()
    df["y"] = df["clarity"].map(LABEL2ID)
    if df["y"].isna().any():
        raise ValueError("Labels desconhecidos.")
    df["row_id"] = np.arange(len(df))
    df["fold"] = folds["fold"].astype(int).to_numpy()

    for f in sorted(df["fold"].unique()):
        a = set(df.loc[df["fold"] != f, "resp_text"])
        b = set(df.loc[df["fold"] == f, "resp_text"])
        if a & b:
            raise ValueError(f"Leakage de resp_text no fold {f}.")
    return df


# --------------------------------------------------------------------------- #
# Embeddings por chunks
# --------------------------------------------------------------------------- #
@torch.no_grad()
def embed_texts(
    texts: list[str],
    model_name: str,
    chunk_tokens: int,
    max_chunks: int,
    batch_size: int,
    prefix: str,
) -> np.ndarray:
    """Retorna [N, 2*H]: [emb do 1º chunk | média dos embeddings dos chunks]."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    if device == "cuda":
        model = model.half()

    if tok.cls_token_id is None or tok.sep_token_id is None:
        raise ValueError("O tokenizer precisa definir cls_token_id e sep_token_id.")

    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"] if prefix else []

    # 1) Tokeniza tudo sem truncar e quebra em chunks.
    raw = tok(texts, add_special_tokens=False, truncation=False,
              return_attention_mask=False)["input_ids"]

    chunks: list[tuple[int, int, list[int]]] = []  # (text_idx, chunk_pos, ids)
    for i, ids in enumerate(raw):
        pieces = [ids[s:s + chunk_tokens] for s in range(0, max(len(ids), 1), chunk_tokens)]
        for pos, piece in enumerate(pieces[:max_chunks]):
            full = [tok.cls_token_id, *prefix_ids, *piece, tok.sep_token_id]
            chunks.append((i, pos, full))

    print(f"Textos: {len(texts)} | chunks: {len(chunks)} | device: {device}")

    # 2) Ordena por tamanho para reduzir padding.
    order = sorted(range(len(chunks)), key=lambda k: len(chunks[k][2]))
    hidden = model.config.hidden_size
    chunk_emb = np.zeros((len(chunks), hidden), dtype=np.float32)

    t0 = time.perf_counter()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = tok.pad(
            {"input_ids": [chunks[k][2] for k in idx]},
            padding=True, return_tensors="pt",
        ).to(device)

        out = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(out.dtype)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        chunk_emb[idx] = pooled.cpu().numpy()

        if (start // batch_size) % 50 == 0:
            print(f"\r  {start + len(idx):>6}/{len(order)} chunks "
                  f"({time.perf_counter() - t0:.0f}s)", end="", flush=True)
    print()

    # 3) Agrega por texto.
    first = np.zeros((len(texts), hidden), dtype=np.float32)
    mean = np.zeros((len(texts), hidden), dtype=np.float32)
    count = np.zeros(len(texts), dtype=np.int32)
    for k, (i, pos, _) in enumerate(chunks):
        if pos == 0:
            first[i] = chunk_emb[k]
        mean[i] += chunk_emb[k]
        count[i] += 1
    mean /= np.maximum(count, 1)[:, None]

    del model
    torch.cuda.empty_cache()
    return np.hstack([first, mean])


def get_embeddings(df: pd.DataFrame, args) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace("/", "_")
    path = CACHE / f"frozen_e5_chunks_{tag}_c{args.chunk_tokens}_m{args.max_chunks}.npy"
    if path.exists():
        X = np.load(path)
        if X.shape[0] == len(df):
            print(f"Embeddings em cache: {path}")
            return X
    X = embed_texts(
        df["resp_text"].tolist(), args.model, args.chunk_tokens,
        args.max_chunks, args.batch_size, args.prefix,
    )
    np.save(path, X)
    return X


# --------------------------------------------------------------------------- #
# CV
# --------------------------------------------------------------------------- #
def run_cv(df: pd.DataFrame, X: np.ndarray, C: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["y"].to_numpy(int)
    folds = df["fold"].to_numpy(int)
    probs = np.zeros((len(df), 3))
    rows = []

    for f in sorted(np.unique(folds)):
        tr, va = folds != f, folds == f
        t0 = time.perf_counter()

        scaler = StandardScaler().fit(X[tr])          # ajustado só no treino do fold
        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=3000, random_state=SEED,
        ).fit(scaler.transform(X[tr]), y[tr])

        p = clf.predict_proba(scaler.transform(X[va]))
        probs[va] = p
        pred = p.argmax(1)

        rows.append({
            "fold": int(f), "C": C,
            "accuracy": accuracy_score(y[va], pred),
            "macro_f1": f1_score(y[va], pred, average="macro"),
            "runtime_seconds": time.perf_counter() - t0,
        })
        print(f"Fold {f}: acc={rows[-1]['accuracy']:.6f} "
              f"f1={rows[-1]['macro_f1']:.6f}")

    pred = probs.argmax(1)
    oof = pd.DataFrame({
        "row_id": df["row_id"], "fold": folds,
        "y_true_id": y, "y_pred_id": pred,
        "y_true": [ID2LABEL[v] for v in y],
        "y_pred": [ID2LABEL[int(v)] for v in pred],
        "prob_c1": probs[:, 0], "prob_c234": probs[:, 1], "prob_c5": probs[:, 2],
        "correct": pred == y,
        "resp_text": df["resp_text"],
    })
    return pd.DataFrame(rows), oof


# --------------------------------------------------------------------------- #
# Comparação com V3 (diagnóstico) + regra única pré-declarada (exploratória)
# --------------------------------------------------------------------------- #
def compare_with_v3(oof: pd.DataFrame) -> pd.DataFrame | None:
    if not V3_OOF.exists():
        print(f"\n[aviso] {V3_OOF.name} não encontrado; comparação pulada.")
        return None

    v3 = pd.read_csv(V3_OOF)
    m = oof.merge(v3, on="row_id", suffixes=("_new", "_v3"), validate="one_to_one")
    assert (m["y_true_id_new"] == m["y_true_id_v3"]).all(), "y_true inconsistente"

    y = m["y_true_id_new"].to_numpy(int)
    new = m["y_pred_id"].to_numpy(int)
    base = m["y_pred_id_v3"].to_numpy(int)
    fold = m["fold_new"].to_numpy(int)

    print("\n" + "=" * 78)
    print("DIAGNÓSTICO vs ENSEMBLE V3 (oracle é só diagnóstico)")
    print("=" * 78)
    print(f"Novo  acc={accuracy_score(y, new):.6f}  V3 acc={accuracy_score(y, base):.6f}")
    print(f"Disagreement        : {np.mean(new != base):.4%}")
    print(f"Novo certo / V3 errado: {int(((new == y) & (base != y)).sum())}")
    print(f"V3 certo / Novo errado: {int(((new != y) & (base == y)).sum())}")
    print(f"Oracle (diagnóstico): {np.mean((new == y) | (base == y)):.4%}")
    for c, name in ID2LABEL.items():
        s = y == c
        gain = int(((new == y) & (base != y) & s).sum())
        loss = int(((new != y) & (base == y) & s).sum())
        print(f"  true {name:5s}: novo corrige {gain:4d} | V3 corrige {loss:4d} | saldo {gain - loss:+d}")

    # REGRA ÚNICA (declarada antes de ver o resultado):
    # empates 2-2-1 SEM c234 (que o V3 não altera) -> desempata pelas
    # probabilidades do novo modelo entre as duas classes líderes.
    probs = m[["prob_c1_new", "prob_c234_new", "prob_c5_new"]].to_numpy(float)
    final = base.copy()
    target = (
        (m["vote_pattern"] == "2-2-1")
        & (m["tie_leader_a"] >= 0)
        & (~m["v3_targeted_tie"].astype(bool))
    ).to_numpy()
    for i in np.flatnonzero(target):
        leaders = [int(m.at[i, "tie_leader_a"]), int(m.at[i, "tie_leader_b"])]
        final[i] = leaders[int(np.argmax(probs[i, leaders]))]

    changed = final != base
    fixes = int(((final == y) & (base != y) & changed).sum())
    breaks = int(((final != y) & (base == y) & changed).sum())

    print("\nREGRA EXPLORATÓRIA (desempate nos 2-2-1 sem c234):")
    print(f"Empates alvo     : {int(target.sum())}")
    print(f"Alteradas        : {int(changed.sum())}  (corrige {fixes} | quebra {breaks} | saldo {fixes - breaks:+d})")
    print(f"V3 acc           : {accuracy_score(y, base):.6f}")
    print(f"V3+regra acc     : {accuracy_score(y, final):.6f}")
    print(f"V3+regra macroF1 : {f1_score(y, final, average='macro'):.6f}")
    per_fold = [
        (int(f), accuracy_score(y[fold == f], base[fold == f]),
         accuracy_score(y[fold == f], final[fold == f]))
        for f in sorted(np.unique(fold))
    ]
    print("Por fold (V3 -> V3+regra):")
    for f, a, b in per_fold:
        print(f"  fold {f}: {a:.6f} -> {b:.6f} ({b - a:+.6f})")
    print("Promover só se o ganho for consistente nos 5 folds; saldo de poucos "
          "exemplos é ruído.")

    return pd.DataFrame([{
        "new_accuracy": accuracy_score(y, new),
        "v3_accuracy": accuracy_score(y, base),
        "disagreement": float(np.mean(new != base)),
        "new_right_v3_wrong": int(((new == y) & (base != y)).sum()),
        "v3_right_new_wrong": int(((new != y) & (base == y)).sum()),
        "oracle_diagnostic": float(np.mean((new == y) | (base == y))),
        "rule_targeted_ties": int(target.sum()),
        "rule_fixes": fixes, "rule_breaks": breaks,
        "v3_plus_rule_accuracy": accuracy_score(y, final),
        "v3_plus_rule_macro_f1": f1_score(y, final, average="macro"),
    }])


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-base")
    p.add_argument("--prefix", default="passage: ", help="E5 exige 'passage: '.")
    p.add_argument("--chunk-tokens", type=int, default=254)
    p.add_argument("--max-chunks", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--C", type=float, default=0.05,
                   help="Fixo e pré-definido (regularização forte p/ 1536 features).")
    args = p.parse_args()

    set_seed(SEED)
    RESULTS.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = get_embeddings(df, args)
    print(f"Features: {X.shape}")

    cv, oof = run_cv(df, X, args.C)

    y, pred = oof["y_true_id"].to_numpy(), oof["y_pred_id"].to_numpy()
    f1s = f1_score(y, pred, labels=[0, 1, 2], average=None)
    summary = pd.DataFrame([{
        "model": args.model, "C": args.C,
        "chunk_tokens": args.chunk_tokens, "max_chunks": args.max_chunks,
        "oof_accuracy": accuracy_score(y, pred),
        "oof_macro_f1": f1_score(y, pred, average="macro"),
        "f1_c1": f1s[0], "f1_c234": f1s[1], "f1_c5": f1s[2],
        "cv_accuracy_mean": cv["accuracy"].mean(),
        "cv_accuracy_std": cv["accuracy"].std(ddof=1),
    }])

    cv.to_csv(RESULTS / "frozen_e5_chunks_lr_cv.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(RESULTS / "frozen_e5_chunks_lr_oof.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "frozen_e5_chunks_lr_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(f"OOF Accuracy = {summary.at[0, 'oof_accuracy']:.6f} | "
          f"Macro-F1 = {summary.at[0, 'oof_macro_f1']:.6f}")
    print(f"F1 c1={f1s[0]:.4f} | c234={f1s[1]:.4f} | c5={f1s[2]:.4f}")
    print(confusion_matrix(y, pred, labels=[0, 1, 2]))

    cmp_df = compare_with_v3(oof)
    if cmp_df is not None:
        cmp_df.to_csv(RESULTS / "frozen_e5_chunks_lr_vs_v3.csv", index=False)


if __name__ == "__main__":
    main()