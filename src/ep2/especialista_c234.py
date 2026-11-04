#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP2/PLN — BERTimbau binário c234 versus c1/c5, fold congelado.

Treina e avalia SOMENTE a partição escolhida do train.xlsx. Nunca lê teste1/2.
Nenhum threshold é ajustado com os rótulos do fold de avaliação.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)

CLASSES = ("c1", "c234", "c5")
LABEL2ID = {name: i for i, name in enumerate(CLASSES)}
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
THRESHOLD = 0.5  # PRÉ-FIXADO; não otimizar no fold de avaliação


def args_cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train", type=Path, default=Path("data/raw/train.xlsx"))
    p.add_argument("--folds", type=Path, default=Path("data/splits/folds.csv"))
    p.add_argument("--v3", type=Path, default=Path("results/ensemble_majority_5_v3_norberto_oof.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/ep2/especialista_c234"))
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-model", action="store_true", help="Não gravar checkpoint (~400 MB)")
    p.add_argument("--check-data", action="store_true", help="Validar arquivos e encerrar sem GPU/modelo")
    p.add_argument("--allow-cpu", action="store_true", help="Permitir treinamento lento sem CUDA")
    p.add_argument("--self-test", action="store_true", help="Testar avaliador sem arquivos/modelos")
    a = p.parse_args()
    if a.max_length < 8 or a.batch_size < 1 or a.eval_batch_size < 1 or a.grad_accum < 1 or a.epochs < 1:
        p.error("max-length >= 8 e demais inteiros positivos são obrigatórios")
    if not (0 <= a.warmup_ratio < 1):
        p.error("warmup-ratio deve estar em [0, 1)")
    return a


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def serializavel(d):
    if isinstance(d, dict):
        return {str(k): serializavel(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [serializavel(v) for v in d]
    if isinstance(d, (np.integer,)):
        return int(d)
    if isinstance(d, (np.floating,)):
        return float(d)
    if isinstance(d, Path):
        return str(d)
    return d


def carregar_e_validar(a):
    for path in (a.train, a.folds, a.v3):
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")

    print("[1/4] Lendo train.xlsx e CSVs ...", flush=True)
    train = pd.read_excel(a.train, engine="openpyxl").reset_index(drop=True)
    folds = pd.read_csv(a.folds, encoding="utf-8-sig")
    oof = pd.read_csv(a.v3, encoding="utf-8-sig")

    n = len(train)
    if n != 20092:
        raise ValueError(f"train.xlsx deve ter 20.092 linhas, encontrou {n}.")
    if len(folds) != n or len(oof) != n:
        raise ValueError("train, folds e OOF devem ter 20.092 linhas cada.")
    if not {"resp_text", "clarity"}.issubset(train.columns):
        raise ValueError("train deve conter resp_text e clarity.")
    if not {"row_index", "fold"}.issubset(folds.columns):
        raise ValueError("folds deve conter row_index e fold.")
    expected = {"row_id", "fold", "y_true_id", "y_pred_id_v3"}
    if not expected.issubset(oof.columns):
        raise ValueError(f"OOF V3 sem colunas obrigatórias: {sorted(expected-set(oof.columns))}")
    if train[["resp_text", "clarity"]].isna().any().any():
        raise ValueError("Treino contém texto/rótulo nulo.")
    if not set(train.clarity.unique()).issubset(LABEL2ID):
        raise ValueError(f"Rótulos inesperados no treino: {train.clarity.unique()}")
    if not np.array_equal(folds.row_index.to_numpy(), np.arange(n)):
        raise ValueError("folds.row_index não coincide com a ordem do train.xlsx.")
    if set(folds.fold.unique()) != set(range(5)):
        raise ValueError("Os folds devem ser 0, 1, 2, 3, 4.")
    if not np.array_equal(oof.row_id.to_numpy(), np.arange(n)):
        raise ValueError("OOF deve estar ordenado por row_id=0..20.091, sem duplicatas.")
    if not np.array_equal(oof.fold.to_numpy(), folds.fold.to_numpy()):
        raise ValueError("Folds OOF e folds.csv não coincidem.")
    y3 = train.clarity.map(LABEL2ID).astype(int).to_numpy()
    if not np.array_equal(oof.y_true_id.astype(int).to_numpy(), y3):
        raise ValueError("Rótulos do CSV OOF não coincidem com train.xlsx.")
    pred_v3 = oof.y_pred_id_v3.astype(int).to_numpy()
    if not np.isin(pred_v3, [0, 1, 2]).all():
        raise ValueError("Predições V3 devem ser IDs 0/1/2.")
    # Todos os textos idênticos devem pertencer a um único fold (protocolo original).
    if (pd.DataFrame({"text": train.resp_text.astype(str), "fold": folds.fold})
            .groupby("text", sort=False).fold.nunique() > 1).any():
        raise ValueError("Vazamento: um mesmo resp_text aparece em folds distintos.")
    if not np.isclose(accuracy_score(y3, pred_v3), 0.475661955, atol=1e-7):
        raise ValueError("OOF V3 diverge do benchmark documentado (0,475661955).")
    val_idx = np.flatnonzero(folds.fold.to_numpy() == a.fold)
    tr_idx = np.flatnonzero(folds.fold.to_numpy() != a.fold)
    print(f"  treino={len(tr_idx)} | validação fold {a.fold}={len(val_idx)}")
    print(f"  V3 OOF completo: {accuracy_score(y3, pred_v3):.8f}")
    print(f"  classe positiva c234 no treino do fold: {(y3[tr_idx]==1).sum()}/{len(tr_idx)}")
    return train, folds, y3, pred_v3, tr_idx, val_idx


def avaliar(y_true3, pred_v3, prob_positive):
    """Métricas transparentes; nenhuma regra é otimizada com y_true3."""
    positive = (y_true3 == 1).astype(int)
    binary_pred = (prob_positive >= THRESHOLD).astype(int)
    gated = np.where((binary_pred == 1) & (pred_v3 != 1), 1, pred_v3)
    base_ok = (pred_v3 == y_true3)
    gate_ok = (gated == y_true3)
    changes = (gated != pred_v3)
    pr, rc, f1, _ = precision_recall_fscore_support(
        positive, binary_pred, labels=[1], zero_division=0
    )
    out = {
        "limiar_predefinido": THRESHOLD,
        "n": len(y_true3),
        "referencia_v3": {
            "accuracy": float(accuracy_score(y_true3, pred_v3)),
            "macro_f1": float(f1_score(y_true3, pred_v3, average="macro")),
            "confusion_matrix": confusion_matrix(y_true3, pred_v3, labels=[0, 1, 2]).tolist(),
        },
        "especialista_binario": {
            "accuracy": float(accuracy_score(positive, binary_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(positive, binary_pred)),
            "precision_c234": float(pr[0]),
            "recall_c234": float(rc[0]),
            "f1_c234": float(f1[0]),
            "roc_auc_c234": float(roc_auc_score(positive, prob_positive)),
            "average_precision_c234": float(average_precision_score(positive, prob_positive)),
            "confusion_matrix_neg_pos": confusion_matrix(positive, binary_pred, labels=[0, 1]).tolist(),
        },
        "v3_mais_especialista_regra_fixa": {
            "regra": "se P(c234)>=0.5, trocar previsão V3 não-c234 para c234; caso contrário, manter V3",
            "accuracy": float(accuracy_score(y_true3, gated)),
            "macro_f1": float(f1_score(y_true3, gated, average="macro")),
            "confusion_matrix": confusion_matrix(y_true3, gated, labels=[0, 1, 2]).tolist(),
            "mudancas": int(changes.sum()),
            "erros_corrigidos": int((~base_ok & gate_ok).sum()),
            "acertos_perdidos": int((base_ok & ~gate_ok).sum()),
            "ganho_liquido_acertos": int(gate_ok.sum() - base_ok.sum()),
            "delta_accuracy_pp": float(100 * (gate_ok.mean() - base_ok.mean())),
        },
    }
    return out, binary_pred, gated


def teste_unitario():
    yt = np.array([0, 1, 2, 1, 0, 2])
    base = np.array([0, 0, 2, 1, 1, 0])
    prob = np.array([0.1, 0.8, 0.1, 0.7, 0.9, 0.6])
    out, pred_bin, gate = avaliar(yt, base, prob)
    assert pred_bin.tolist() == [0, 1, 0, 1, 1, 1]
    assert gate.tolist() == [0, 1, 2, 1, 1, 1]
    assert out["v3_mais_especialista_regra_fixa"]["erros_corrigidos"] == 1
    assert out["v3_mais_especialista_regra_fixa"]["acertos_perdidos"] == 0
    print("Self-test: OK")


def inicializar_semente(seed, torch):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def executar(a):
    train, folds, y3, pred_v3, tr_idx, val_idx = carregar_e_validar(a)
    if a.check_data:
        print("Validação de integridade: OK. Nenhum treinamento executado.")
        return

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  get_linear_schedule_with_warmup)
    except ImportError as exc:
        raise RuntimeError("Instale PyTorch com CUDA e transformers, conforme README.") from exc

    if not torch.cuda.is_available() and not a.allow_cpu:
        raise RuntimeError("CUDA indisponível: treinamento não iniciado. Use GPU NVIDIA ou --allow-cpu (muito lento).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    inicializar_semente(a.seed, torch)
    print(f"[2/4] Modelo: {a.model} | dispositivo: {device} | AMP: {amp}", flush=True)

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2,
        id2label={0: "extremos_c1_c5", 1: "c234"},
        label2id={"extremos_c1_c5": 0, "c234": 1},
    ).to(device)
    text = train.resp_text.astype(str).tolist()
    print("  Tokenizando treino e validação ...", flush=True)
    train_enc = tokenizer([text[i] for i in tr_idx], truncation=True, max_length=a.max_length,
                          padding=False, return_attention_mask=True)
    val_enc = tokenizer([text[i] for i in val_idx], truncation=True, max_length=a.max_length,
                        padding=False, return_attention_mask=True)
    train_labels = (y3[tr_idx] == 1).astype(np.int64)
    val_labels = (y3[val_idx] == 1).astype(np.int64)

    class DatasetBin(Dataset):
        def __init__(self, encoded, targets):
            self.encoded = encoded
            self.targets = targets
        def __len__(self):
            return len(self.targets)
        def __getitem__(self, ix):
            return {k: self.encoded[k][ix] for k in self.encoded}, int(self.targets[ix])

    def collate(items):
        features, labels = zip(*items)
        batch = tokenizer.pad(list(features), padding=True, pad_to_multiple_of=8,
                              return_tensors="pt")
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    # Drop_last=False mantém todas as instâncias; último grupo de acumulação reescalado.
    loader_tr = DataLoader(DatasetBin(train_enc, train_labels), batch_size=a.batch_size,
                           shuffle=True, num_workers=0, collate_fn=collate)
    loader_val = DataLoader(DatasetBin(val_enc, val_labels), batch_size=a.eval_batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_updates_per_epoch = math.ceil(len(loader_tr) / a.grad_accum)
    steps = n_updates_per_epoch * a.epochs
    warmup_steps = int(steps * a.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    if a.class_weight == "balanced":
        freq = np.bincount(train_labels, minlength=2)
        weights = len(train_labels) / (2 * freq.astype(float))
        weight_t = torch.tensor(weights, device=device, dtype=torch.float32)
    else:
        weights = None
        weight_t = None
    criterion = torch.nn.CrossEntropyLoss(weight=weight_t)

    run_dir = a.outdir / f"fold{a.fold}_{a.class_weight}_seed{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(a), "positive_id": 1, "classes": CLASSES, "binary_label": "c234 vs extremos",
              "threshold_fixed": THRESHOLD, "device": str(device), "amp": amp,
              "estimated_optimizer_steps": steps, "warmup_steps": warmup_steps,
              "class_weights": weights.tolist() if weights is not None else None,
              "inputs_sha256": {p.name: sha256_file(p) for p in (a.train, a.folds, a.v3)}}
    (run_dir / "config.json").write_text(json.dumps(serializavel(config), indent=2, ensure_ascii=False), encoding="utf-8")
    history = []
    print(f"[3/4] Treino: {a.epochs} épocas, batch={a.batch_size}, acumulação={a.grad_accum}", flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        samples = 0
        bcount = len(loader_tr)
        for batch_idx, batch in enumerate(loader_tr):
            labels = batch.pop("labels").to(device)
            features = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(**features).logits
                # Computa CE em float32 (inclusive com FP16 ativo).
                loss = criterion(logits.float(), labels)
            # Evita reduzir o peso de um último grupo de batches incompleto.
            group_start = (batch_idx // a.grad_accum) * a.grad_accum
            actual_group_size = min(a.grad_accum, bcount - group_start)
            scaler.scale(loss / actual_group_size).backward()
            loss_total += float(loss.item()) * len(labels)
            samples += len(labels)
            if ((batch_idx + 1) % a.grad_accum == 0) or (batch_idx + 1 == bcount):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        print(f"  época {epoch}/{a.epochs} | loss média: {loss_total/samples:.5f}", flush=True)
        history.append({"epoch": epoch, "train_mean_loss": loss_total / samples})

    print("[4/4] Inferência no fold de validação ...", flush=True)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch in loader_val:
            batch.pop("labels")
            features = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(**features).logits
            probabilities.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy().tolist())
    proba = np.asarray(probabilities, dtype=np.float64)
    if len(proba) != len(val_idx) or not np.isfinite(proba).all():
        raise RuntimeError("Predições incompletas ou não finitas.")

    evaluation, binary_pred, gated = avaliar(y3[val_idx], pred_v3[val_idx], proba)
    evaluation["treino"] = {"seconds": float(time.perf_counter() - t0), "history": history}
    evaluation["protocolo"] = "fold holdout; sem ajuste de threshold; V3 OOF do mesmo fold"
    df = pd.DataFrame({
        "row_id": val_idx, "fold": a.fold,
        "y_true_id": y3[val_idx], "y_true": [CLASSES[x] for x in y3[val_idx]],
        "pred_v3_id": pred_v3[val_idx],
        "prob_c234_especialista": proba,
        "pred_binaria_c234": binary_pred,
        "pred_v3_gate_id": gated,
        "n_palavras": [len(re.findall(r"\b\w+\b", text[i], flags=re.UNICODE)) for i in val_idx],
        "v3_correto": pred_v3[val_idx] == y3[val_idx],
        "gate_correto": gated == y3[val_idx],
    })
    df.to_csv(run_dir / "predicoes_fold.csv", index=False, encoding="utf-8-sig")
    df["faixa_palavras"] = pd.cut(
        df["n_palavras"], bins=[-1, 25, 50, 100, 200, 400, float("inf")],
        labels=["ate_25", "26_50", "51_100", "101_200", "201_400", "acima_400"],
    )
    rows = []
    for faixa, group in df.groupby("faixa_palavras", observed=True, sort=True):
        rows.append({"faixa_palavras": str(faixa), "n": len(group),
                     "accuracy_v3": float(group["v3_correto"].mean()),
                     "accuracy_gate": float(group["gate_correto"].mean()),
                     "mudancas": int((group["pred_v3_gate_id"] != group["pred_v3_id"]).sum()),
                     "recall_c234_gate": (float((group.loc[group.y_true_id == 1, "pred_v3_gate_id"] == 1).mean())
                                          if (group.y_true_id == 1).any() else None)})
    pd.DataFrame(rows).to_csv(run_dir / "por_extensao.csv", index=False, encoding="utf-8-sig")
    (run_dir / "metricas.json").write_text(json.dumps(serializavel(evaluation), ensure_ascii=False, indent=2), encoding="utf-8")
    if not a.no_save_model:
        model.save_pretrained(run_dir / "checkpoint")
        tokenizer.save_pretrained(run_dir / "checkpoint")
    print("\nRESULTADO FOLD", a.fold)
    print("  V3 original:    ", round(evaluation["referencia_v3"]["accuracy"], 6))
    print("  V3 + gate fixo: ", round(evaluation["v3_mais_especialista_regra_fixa"]["accuracy"], 6))
    print("  Delta (p.p.):  ", round(evaluation["v3_mais_especialista_regra_fixa"]["delta_accuracy_pp"], 4))
    print("  F1 especialista c234: ", round(evaluation["especialista_binario"]["f1_c234"], 6))
    print("  Salvo em:", run_dir.resolve())


def main():
    a = args_cli()
    if a.self_test:
        teste_unitario()
        return
    executar(a)


if __name__ == "__main__":
    main()
