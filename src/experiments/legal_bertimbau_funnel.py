"""
Seção 7.3 — novo Transformer fine-tunado, seguindo o funil da seção 40.

Candidato padrão: rufimelo/Legal-BERTimbau-base
  BERTimbau com pré-treino continuado em textos jurídicos em português.
  Hipótese: respostas de e-SIC citam leis, artigos e decretos; um encoder com
  domínio jurídico pode errar de forma diferente do BERTimbau/NorBERTo,
  principalmente em c234. Cabe em 4 GB na mesma configuração do BERT512.

Configuração IDÊNTICA ao BERTimbau 512 (LR 3e-5, 2 épocas, batch 4x4, FP16,
seed 42). Só muda o checkpoint, então a diferença vem do pré-treino.

Funil (rode em ordem, um estágio por vez):
    python src/experiments/legal_bertimbau_funnel.py --stage tokenize
    python src/experiments/legal_bertimbau_funnel.py --stage benchmark
    python src/experiments/legal_bertimbau_funnel.py --stage cv --folds 0
        -> imprime o veredito GO / NO-GO contra o V3
    python src/experiments/legal_bertimbau_funnel.py --stage cv --folds 1 2 3 4
        -> SÓ se o veredito do fold 0 for GO

Outro checkpoint (mesmo funil):
    --model google-bert/bert-base-multilingual-cased --tag mbert_512

Saídas (results/): <tag>_cv.csv | <tag>_oof.csv | <tag>_summary.csv
                   <tag>_vs_v3_fold<k>.csv | <tag>_benchmark.csv
"""

from __future__ import annotations

import argparse
import gc
import math
import platform
import sys
import time
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "train.xlsx"
FOLDS_PATH = ROOT / "data" / "splits" / "folds.csv"
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
V3_OOF = RESULTS / "ensemble_majority_5_v3_norberto_oof.csv"

DEFAULT_MODEL = "rufimelo/Legal-BERTimbau-base"
LABEL2ID = {"c1": 0, "c234": 1, "c5": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
SEED = 42
EXPECTED_N = 20_092

# Critério GO/NO-GO declarado ANTES de ver qualquer resultado.
# Referência: NorBERTo fold 0 = 47,55% com saldo positivo em c234.
GO_MIN_ACCURACY = 0.450      # competitivo (BERT512 fica em 0,45-0,47 por fold)
GO_MIN_C234_BALANCE = 1      # saldo (novo certo/V3 errado - inverso) em c234 > 0


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


class EncodedDataset(Dataset):
    def __init__(self, enc: dict[str, list], labels: np.ndarray, idx: np.ndarray):
        self.enc, self.labels, self.idx = enc, labels, idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict:
        j = int(self.idx[i])
        item = {k: v[j] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[j])
        return item


def encode(texts: list[str], tokenizer, max_length: int) -> dict[str, list]:
    out = tokenizer(texts, truncation=True, max_length=max_length,
                    padding=False, return_attention_mask=True)
    return {k: out[k] for k in out.keys()}


# --------------------------------------------------------------------------- #
# Trainer / callbacks
# --------------------------------------------------------------------------- #
class FixedGASTrainer(Trainer):
    """Normalização de gradient accumulation controlada pelo Trainer."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model_accepts_loss_kwargs = False


class NonFiniteGradNormCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and logs.get("grad_norm") is not None:
            if not math.isfinite(float(logs["grad_norm"])):
                raise RuntimeError(
                    f"grad_norm não finito no step {state.global_step}: {logs['grad_norm']}")
        return control


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, pred),
            "macro_f1": f1_score(p.label_ids, pred, average="macro", zero_division=0)}


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def make_args(args, out_dir: Path, **over) -> TrainingArguments:
    kw = dict(
        output_dir=str(out_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="linear",
        warmup_steps=0,
        max_grad_norm=1.0,
        fp16=True,
        gradient_checkpointing=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=100,
        report_to=[],
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=0,
    )
    kw.update(over)
    return TrainingArguments(**kw)


def new_model(model_name: str):
    set_seed(SEED)  # seed imediatamente antes de inicializar a cabeça aleatória
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL,
        problem_type="single_label_classification",
    )


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def upsert(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        drop = set(map(tuple, new[keys].to_numpy().tolist()))
        old = old[~old[keys].apply(tuple, axis=1).isin(drop)]
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.sort_values(keys).to_csv(path, index=False, encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# ESTÁGIO 1 — análise de tokenização
# --------------------------------------------------------------------------- #
def stage_tokenize(df: pd.DataFrame, args) -> None:
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"Tokenizer: {tok.__class__.__name__} | vocab={tok.vocab_size} | "
          f"model_max_length={tok.model_max_length}")

    ids = tok(df["resp_text"].tolist(), add_special_tokens=True, truncation=False,
              padding=False, return_attention_mask=False)["input_ids"]
    lengths = np.array([len(x) for x in ids])
    unk = tok.unk_token_id
    unk_rate = np.mean([np.mean(np.array(x) == unk) for x in ids]) if unk is not None else float("nan")

    print(f"média={lengths.mean():.0f} | mediana={np.median(lengths):.0f} | "
          f"p95={np.percentile(lengths, 95):.0f} | p99={np.percentile(lengths, 99):.0f}")
    print(f"Taxa média de [UNK]: {unk_rate:.4%}")
    print("\nTruncamento (BERTimbau de referência: 256=35,86% | 384=18,94% | 512=10,82%)")
    for L in (256, 384, 512):
        print(f"  max_length={L}: {np.mean(lengths > L):.2%}")
    print("\nTruncamento por classe em 512:")
    for c, n in ID2LABEL.items():
        m = df["y"].to_numpy() == c
        print(f"  {n:5s}: {np.mean(lengths[m] > 512):.2%}")

    print("\nSe o truncamento for parecido com o BERTimbau e [UNK] for baixo, "
          "siga para --stage benchmark.")


# --------------------------------------------------------------------------- #
# ESTÁGIO 2 — benchmark operacional (não usa validação)
# --------------------------------------------------------------------------- #
def stage_benchmark(df: pd.DataFrame, args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível.")
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = encode(df["resp_text"].tolist(), tok, args.max_length)
    y = df["y"].to_numpy(int)
    tr_idx = np.flatnonzero(df["fold"].to_numpy() != 0)

    model = new_model(args.model)
    cleanup()
    torch.cuda.reset_peak_memory_stats()

    trainer = FixedGASTrainer(
        model=model,
        args=make_args(args, CHECKPOINTS / args.tag / "bench",
                       max_steps=100, eval_strategy="no", logging_steps=10),
        train_dataset=EncodedDataset(enc, y, tr_idx),
        data_collator=DataCollatorWithPadding(tok, padding="longest", return_tensors="pt"),
        callbacks=[NonFiniteGradNormCallback()],
    )

    t0 = time.perf_counter()
    out = trainer.train()
    elapsed = time.perf_counter() - t0
    torch.cuda.synchronize()

    sec_per_step = elapsed / 100
    eff_batch = args.batch_size * args.grad_accum
    steps_fold = math.ceil(len(tr_idx) / eff_batch) * args.epochs
    est_min = sec_per_step * steps_fold / 60
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print("\n" + "=" * 78)
    print(f"Tempo/step: {sec_per_step:.2f}s | pico VRAM: {peak:.2f} GB | "
          f"loss final: {out.training_loss:.4f}")
    print(f"Estimativa por fold: {est_min:.0f} min | 5 folds: {est_min * 5 / 60:.1f} h")
    print("Go operacional: sem OOM, loss finita e custo aceitável -> --stage cv --folds 0")

    pd.DataFrame([{
        "model": args.model, "max_length": args.max_length,
        "sec_per_step": sec_per_step, "peak_vram_gb": peak,
        "est_minutes_per_fold": est_min, "train_loss": out.training_loss,
        "transformers": transformers.__version__, "torch": torch.__version__,
    }]).to_csv(RESULTS / f"{args.tag}_benchmark.csv", index=False)


# --------------------------------------------------------------------------- #
# ESTÁGIO 3/5 — CV por fold + diagnóstico contra V3
# --------------------------------------------------------------------------- #
def compare_with_v3(oof: pd.DataFrame, tag: str, label: str) -> None:
    if not V3_OOF.exists():
        print(f"[aviso] {V3_OOF.name} não encontrado; comparação pulada.")
        return

    v3 = pd.read_csv(V3_OOF)
    m = oof.merge(v3, on="row_id", suffixes=("_new", "_v3"), validate="one_to_one")
    if not (m["y_true_id_new"] == m["y_true_id_v3"]).all():
        raise ValueError("y_true inconsistente entre novo modelo e V3.")

    y = m["y_true_id_new"].to_numpy(int)
    new = m["y_pred_id"].to_numpy(int)
    base = m["y_pred_id_v3"].to_numpy(int)
    acc_new, acc_v3 = accuracy_score(y, new), accuracy_score(y, base)

    print("\n" + "=" * 78)
    print(f"DIAGNÓSTICO vs ENSEMBLE V3 — {label} (n={len(m)}; oracle só diagnóstico)")
    print("=" * 78)
    print(f"Novo acc={acc_new:.6f} | V3 acc={acc_v3:.6f}")
    print(f"Disagreement          : {np.mean(new != base):.2%}")
    print(f"Novo certo / V3 errado: {int(((new == y) & (base != y)).sum())}")
    print(f"V3 certo / Novo errado: {int(((new != y) & (base == y)).sum())}")
    print(f"Oracle (diagnóstico)  : {np.mean((new == y) | (base == y)):.2%}")

    balances = {}
    for c, name in ID2LABEL.items():
        s = y == c
        g = int(((new == y) & (base != y) & s).sum())
        l = int(((new != y) & (base == y) & s).sum())
        balances[name] = g - l
        print(f"  true {name:5s}: novo corrige {g:4d} | V3 corrige {l:4d} | saldo {g - l:+d}")

    go = (acc_new >= GO_MIN_ACCURACY) and (balances["c234"] >= GO_MIN_C234_BALANCE)
    print("\nCritério pré-declarado: "
          f"accuracy >= {GO_MIN_ACCURACY:.3f} E saldo c234 >= {GO_MIN_C234_BALANCE}")
    print(f"  accuracy {acc_new:.4f} -> {'ok' if acc_new >= GO_MIN_ACCURACY else 'FALHA'}")
    print(f"  saldo c234 {balances['c234']:+d} -> "
          f"{'ok' if balances['c234'] >= GO_MIN_C234_BALANCE else 'FALHA'}")
    print(f"VEREDITO: {'GO' if go else 'NO-GO'}")
    if not go:
        print("Não execute os folds 1-4. Registre como resultado negativo.")

    pd.DataFrame([{
        "label": label, "n": len(m), "new_accuracy": acc_new, "v3_accuracy": acc_v3,
        "disagreement": float(np.mean(new != base)),
        "new_right_v3_wrong": int(((new == y) & (base != y)).sum()),
        "v3_right_new_wrong": int(((new != y) & (base == y)).sum()),
        "balance_c1": balances["c1"], "balance_c234": balances["c234"],
        "balance_c5": balances["c5"], "verdict": "GO" if go else "NO-GO",
    }]).to_csv(RESULTS / f"{tag}_vs_v3_{label.replace(' ', '_')}.csv", index=False)


def run_fold(df: pd.DataFrame, enc: dict, tok, fold: int, args) -> None:
    y_all = df["y"].to_numpy(int)
    folds = df["fold"].to_numpy(int)
    tr_idx, va_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)

    print("\n" + "=" * 78)
    print(f"{args.tag.upper()} — FOLD {fold} | treino={len(tr_idx)} | validação={len(va_idx)}")
    print("=" * 78)

    train_ds = EncodedDataset(enc, y_all, tr_idx)
    val_ds = EncodedDataset(enc, y_all, va_idx)
    model = new_model(args.model)
    cleanup()
    torch.cuda.reset_peak_memory_stats()

    trainer = FixedGASTrainer(
        model=model,
        args=make_args(args, CHECKPOINTS / args.tag / f"fold_{fold}"),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok, padding="longest", return_tensors="pt"),
        compute_metrics=compute_metrics,
        callbacks=[NonFiniteGradNormCallback()],
    )

    t0 = time.perf_counter()
    try:
        train_out = trainer.train()
        torch.cuda.synchronize()
        runtime = time.perf_counter() - t0

        pred_out = trainer.predict(val_ds)
        logits = pred_out.predictions[0] if isinstance(pred_out.predictions, tuple) \
            else pred_out.predictions
        probs = softmax(np.asarray(logits))
        pred = probs.argmax(1)
        y = y_all[va_idx]

        acc = accuracy_score(y, pred)
        mf1 = f1_score(y, pred, average="macro", zero_division=0)
        f1s = f1_score(y, pred, labels=[0, 1, 2], average=None, zero_division=0)

        oof = pd.DataFrame({
            "row_id": df["row_id"].to_numpy()[va_idx], "fold": fold,
            "y_true_id": y, "y_pred_id": pred,
            "y_true": [ID2LABEL[int(v)] for v in y],
            "y_pred": [ID2LABEL[int(v)] for v in pred],
            "prob_c1": probs[:, 0], "prob_c234": probs[:, 1], "prob_c5": probs[:, 2],
            "correct": pred == y,
            "resp_text": df["resp_text"].to_numpy()[va_idx],
        })
        upsert(RESULTS / f"{args.tag}_oof.csv", oof, ["row_id"])

        history = pd.DataFrame(trainer.state.log_history)
        history.insert(0, "fold", fold)
        history.to_csv(RESULTS / f"{args.tag}_history_fold{fold}.csv",
                       index=False, encoding="utf-8-sig")

        upsert(RESULTS / f"{args.tag}_cv.csv", pd.DataFrame([{
            "fold": fold, "status": "success", "model_name": args.model,
            "max_length": args.max_length, "learning_rate": args.learning_rate,
            "epochs": args.epochs, "physical_batch": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "fp16": True, "seed": SEED,
            "accuracy": acc, "macro_f1": mf1,
            "f1_c1": f1s[0], "f1_c234": f1s[1], "f1_c5": f1s[2],
            "train_loss": train_out.training_loss,
            "runtime_minutes": runtime / 60,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "python": platform.python_version(),
        }]), ["fold"])

        print(f"\nFold {fold}: acc={acc:.6f} | macro-F1={mf1:.6f} | "
              f"F1 c1={f1s[0]:.4f} c234={f1s[1]:.4f} c5={f1s[2]:.4f} | "
              f"{runtime / 60:.1f} min")
        print(confusion_matrix(y, pred, labels=[0, 1, 2]))

        # Diagnóstico só com as linhas OOF deste fold.
        compare_with_v3(oof, args.tag, f"fold{fold}")

    finally:
        del trainer, model, train_ds, val_ds
        cleanup()


def consolidate(args) -> None:
    cv_path, oof_path = RESULTS / f"{args.tag}_cv.csv", RESULTS / f"{args.tag}_oof.csv"
    if not (cv_path.exists() and oof_path.exists()):
        return
    cv = pd.read_csv(cv_path)
    cv = cv[cv["status"] == "success"]
    if set(cv["fold"].astype(int)) != {0, 1, 2, 3, 4}:
        print(f"\nFolds concluídos: {sorted(cv['fold'].astype(int))} (CV completa ainda não disponível).")
        return

    oof = pd.read_csv(oof_path).sort_values("row_id")
    if len(oof) != EXPECTED_N or oof["row_id"].duplicated().any():
        raise RuntimeError("OOF consolidado inválido.")
    y, p = oof["y_true_id"].to_numpy(), oof["y_pred_id"].to_numpy()
    f1s = f1_score(y, p, labels=[0, 1, 2], average=None)

    summary = pd.DataFrame([{
        "model": args.model, "oof_accuracy": accuracy_score(y, p),
        "oof_macro_f1": f1_score(y, p, average="macro"),
        "f1_c1": f1s[0], "f1_c234": f1s[1], "f1_c5": f1s[2],
        "cv_accuracy_mean": cv["accuracy"].mean(),
        "cv_accuracy_std": cv["accuracy"].std(ddof=1),
        "cv_macro_f1_mean": cv["macro_f1"].mean(),
        "cv_macro_f1_std": cv["macro_f1"].std(ddof=1),
    }])
    summary.to_csv(RESULTS / f"{args.tag}_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("CV COMPLETA")
    print("=" * 78)
    print(f"OOF Accuracy={summary.at[0, 'oof_accuracy']:.6f} | "
          f"Macro-F1={summary.at[0, 'oof_macro_f1']:.6f}")
    print(f"CV Accuracy={summary.at[0, 'cv_accuracy_mean']:.6f} ± "
          f"{summary.at[0, 'cv_accuracy_std']:.6f}")
    compare_with_v3(oof, args.tag, "todos_os_folds")
    print("\nPróximo passo: decidir se o modelo entra no ensemble. Não criar grid "
          "de regras sobre o OOF; se entrar, uma única regra declarada antes.")


def stage_cv(df: pd.DataFrame, args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível.")
    bad = [f for f in args.folds if f not in range(5)]
    if bad or len(set(args.folds)) != len(args.folds):
        raise ValueError("--folds deve conter valores únicos entre 0 e 4.")

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = encode(df["resp_text"].tolist(), tok, args.max_length)

    done = set()
    cv_path = RESULTS / f"{args.tag}_cv.csv"
    if cv_path.exists() and not args.rerun:
        old = pd.read_csv(cv_path)
        done = set(old.loc[old["status"] == "success", "fold"].astype(int))

    for fold in args.folds:
        if fold in done:
            print(f"Fold {fold} já concluído; pulando (use --rerun para refazer).")
            continue
        run_fold(df, enc, tok, fold, args)

    consolidate(args)


# --------------------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["tokenize", "benchmark", "cv"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tag", default="legal_bertimbau_512")
    p.add_argument("--folds", type=int, nargs="+", default=[0])
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rerun", action="store_true")
    args = p.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    df = load_data()

    print(f"Modelo: {args.model} | estágio: {args.stage} | "
          f"transformers {transformers.__version__} | torch {torch.__version__}")

    {"tokenize": stage_tokenize, "benchmark": stage_benchmark, "cv": stage_cv}[args.stage](df, args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        raise SystemExit(130)