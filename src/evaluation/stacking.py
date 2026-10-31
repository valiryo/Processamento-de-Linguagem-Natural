"""
Utilitários compartilhados para combinar componentes ao nível de probabilidade.

Usado por:
    src/experiments/stacking_oof_cv.py
    src/experiments/stacking_apply_test.py
    src/experiments/holdout_test1_v3.py

Tudo é alinhado por row_id, nunca pela posição das linhas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score


LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

PROB_COLS = ["prob_c1", "prob_c234", "prob_c5"]
SCORE_COLS = ["score_c1", "score_c234", "score_c5"]

EPS = 1e-6


# ============================================================
# LEITURA
# ============================================================

def read_csv_bom(path: Path) -> pd.DataFrame:
    """Alguns CSVs do projeto foram gravados com BOM (utf-8-sig)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def read_component(paths: list[Path], kind: str) -> pd.DataFrame:
    """
    Lê um ou mais CSVs do mesmo componente (por exemplo, várias seeds) e
    devolve a média das colunas de probabilidade/score, indexada por row_id.

    kind = "prob"  -> colunas prob_c1, prob_c234, prob_c5
    kind = "score" -> colunas score_c1, score_c234, score_c5
    """
    cols = PROB_COLS if kind == "prob" else SCORE_COLS
    frames = []

    for path in paths:
        df = read_csv_bom(path)
        missing = set(["row_id", *cols]) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: colunas ausentes {sorted(missing)}")
        if df["row_id"].duplicated().any():
            raise ValueError(f"{path.name}: row_id duplicado.")
        frames.append(df.set_index("row_id")[cols].astype(float).sort_index())

    base_index = frames[0].index
    for path, frame in zip(paths[1:], frames[1:]):
        if not frame.index.equals(base_index):
            raise ValueError(f"{path.name}: row_ids diferentes dos demais seeds.")

    mean = sum(frames) / len(frames)

    if not np.isfinite(mean.to_numpy()).all():
        raise ValueError(f"Valores não finitos em {[p.name for p in paths]}")

    return mean


# ============================================================
# FEATURES
# ============================================================

def clr_block(values: np.ndarray, kind: str) -> np.ndarray:
    """
    Representação centrada em log (CLR) de um bloco de 3 colunas.

    - prob : log(p) - média(log p)
    - score: score - média(score)   (decision_function do LinearSVC)

    A centragem remove o grau de liberdade redundante das probabilidades e
    deixa a regressão logística do meta-modelo bem condicionada.
    """
    if kind == "prob":
        logp = np.log(np.clip(values, EPS, 1.0))
        return logp - logp.mean(axis=1, keepdims=True)
    return values - values.mean(axis=1, keepdims=True)


def build_features(
    components: dict[str, pd.DataFrame],
    kinds: dict[str, str],
    order: list[str],
    row_ids: np.ndarray,
) -> np.ndarray:
    blocks = []
    for name in order:
        block = components[name].reindex(row_ids)
        if block.isna().any().any():
            raise ValueError(f"{name}: há row_id sem previsão.")
        blocks.append(clr_block(block.to_numpy(dtype=float), kinds[name]))
    return np.hstack(blocks)


def soft_vote(
    components: dict[str, pd.DataFrame],
    kinds: dict[str, str],
    row_ids: np.ndarray,
) -> np.ndarray:
    """Média simples das probabilidades (sem parâmetros). Ignora scores."""
    mats = [
        components[name].reindex(row_ids).to_numpy(dtype=float)
        for name, kind in kinds.items()
        if kind == "prob"
    ]
    return np.mean(mats, axis=0)


# ============================================================
# MÉTRICAS
# ============================================================

def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    f1s = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_c1": float(f1s[0]),
        "f1_c234": float(f1s[1]),
        "f1_c5": float(f1s[2]),
    }


def paired_vs_reference(
    y_true: np.ndarray,
    y_new: np.ndarray,
    y_ref: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Comparação pareada contra uma referência (V3).

    - new_right_ref_wrong / ref_right_new_wrong: contagens discordantes
    - mcnemar_p: teste binomial exato bicaudal sobre os discordantes
    - ci_low/ci_high: IC 95% bootstrap da diferença de accuracy, reamostrando
      GRUPOS de texto idêntico (as duplicatas não são independentes)
    """
    right_new = (y_new == y_true).astype(float)
    right_ref = (y_ref == y_true).astype(float)

    b = int(((right_new == 1) & (right_ref == 0)).sum())
    c = int(((right_new == 0) & (right_ref == 1)).sum())
    p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else 1.0

    out = {
        "delta_accuracy": float(right_new.mean() - right_ref.mean()),
        "new_right_ref_wrong": b,
        "ref_right_new_wrong": c,
        "mcnemar_p": p,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
    }

    if groups is None:
        return out

    _, g = np.unique(groups, return_inverse=True)
    n_groups = int(g.max()) + 1
    diff_by_group = np.bincount(g, weights=right_new - right_ref, minlength=n_groups)
    size_by_group = np.bincount(g, minlength=n_groups).astype(float)

    rng = np.random.default_rng(seed)
    diffs = []
    chunk = 200
    for start in range(0, n_boot, chunk):
        m = min(chunk, n_boot - start)
        idx = rng.integers(0, n_groups, size=(m, n_groups))
        diffs.append(diff_by_group[idx].sum(axis=1) / size_by_group[idx].sum(axis=1))
    diffs = np.concatenate(diffs)

    out["ci_low"] = float(np.percentile(diffs, 2.5))
    out["ci_high"] = float(np.percentile(diffs, 97.5))
    return out
