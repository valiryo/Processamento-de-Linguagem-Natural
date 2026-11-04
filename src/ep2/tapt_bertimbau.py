#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP2: TAPT (MLM no corpus e-SIC) seguido de fine-tuning ternario.

O corpus MLM e o treino supervisionado sao EXCLUSIVAMENTE as linhas fora do
fold de validacao, conforme data/splits/folds.csv. NUNCA le teste1 ou teste2.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BASE_MODEL = "neuralmind/bert-base-portuguese-cased"
CLASSES = ("c1", "c234", "c5")
LABEL2ID = {x: i for i, x in enumerate(CLASSES)}


def options():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train", type=Path, default=Path("data/raw/train.xlsx"))
    p.add_argument("--folds", type=Path, default=Path("data/splits/folds.csv"))
    p.add_argument("--v3", type=Path, default=Path("results/ensemble_majority_5_v3_norberto_oof.csv"))
    p.add_argument("--bert512", type=Path, default=Path("results/bertimbau_512_oof.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/ep2/tapt_bertimbau"))
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--fold", type=int, choices=list(range(5)), default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlm-max-length", type=int, default=256)
    p.add_argument("--cls-max-length", type=int, default=512)
    p.add_argument("--mlm-epochs", type=int, default=1)
    p.add_argument("--cls-epochs", type=int, default=2)
    p.add_argument("--mlm-batch", type=int, default=8)
    p.add_argument("--mlm-accum", type=int, default=2)
    p.add_argument("--cls-batch", type=int, default=4)
    p.add_argument("--cls-accum", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--mlm-lr", type=float, default=2e-5)
    p.add_argument("--cls-lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--mlm-probability", type=float, default=0.15)
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Permite sobrescrever a saida deste fold")
    a = p.parse_args()
    for field in ("mlm_max_length", "cls_max_length"):
        if getattr(a, field) < 16:
            p.error(f"{field} >= 16")
    for field in ("mlm_epochs", "cls_epochs", "mlm_batch", "mlm_accum", "cls_batch", "cls_accum", "eval_batch"):
        if getattr(a, field) < 1:
            p.error(f"{field} deve ser positivo")
    if not (0 < a.mlm_probability < 1) or not (0 <= a.warmup_ratio < 1):
        p.error("MLM probability deve estar em (0,1), warmup em [0,1)")
    if a.mlm_lr <= 0 or a.cls_lr <= 0:
        p.error("learning rate deve ser positivo")
    return a


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(a):
    for f in (a.train, a.folds, a.v3):
        if not f.is_file():
            raise FileNotFoundError(f"Arquivo ausente: {f}")
    train = pd.read_excel(a.train, engine="openpyxl").reset_index(drop=True)
    folds = pd.read_csv(a.folds, encoding="utf-8-sig")
    v3 = pd.read_csv(a.v3, encoding="utf-8-sig")
    n = len(train)
    if n != 20092 or len(folds) != n or len(v3) != n:
        raise ValueError("Esperadas 20.092 linhas no treino, folds e V3")
    if not {"resp_text", "clarity"}.issubset(train.columns):
        raise ValueError("Treino sem resp_text/clarity")
    if not {"row_index", "fold"}.issubset(folds.columns):
        raise ValueError("Folds sem row_index/fold")
    if not {"row_id", "fold", "y_true_id", "y_pred_id_v3"}.issubset(v3.columns):
        raise ValueError("V3 OOF sem colunas obrigatorias")
    y = train.clarity.map(LABEL2ID)
    if y.isna().any() or train.resp_text.isna().any():
        raise ValueError("Treino contem rotulos inesperados ou textos ausentes")
    y = y.astype(int).to_numpy()
    ids = np.arange(n)
    if not np.array_equal(folds.row_index.to_numpy(), ids) or not np.array_equal(v3.row_id.to_numpy(), ids):
        raise ValueError("Row IDs fora da ordem original")
    if not np.array_equal(folds.fold.to_numpy(), v3.fold.to_numpy()):
        raise ValueError("Folds nao coincidem com V3")
    if not np.array_equal(v3.y_true_id.to_numpy(), y):
        raise ValueError("Rotulos do V3 nao coincidem com train.xlsx")
    if sorted(folds.fold.unique()) != list(range(5)):
        raise ValueError("Folds precisam ser 0..4")
    if (pd.DataFrame({"t": train.resp_text.astype(str), "f": folds.fold})
            .groupby("t", sort=False).f.nunique() > 1).any():
        raise ValueError("Leakage: texto identico aparece em mais de um fold")
    vpred = v3.y_pred_id_v3.to_numpy(dtype=int)
    if not np.isin(vpred, [0, 1, 2]).all() or not np.isclose(accuracy_score(y, vpred), 0.47566195500696795, atol=1e-8):
        raise ValueError("V3 nao confere com benchmark oficial do EP1")
    k = folds.fold.to_numpy()
    train_idx = np.flatnonzero(k != a.fold)
    val_idx = np.flatnonzero(k == a.fold)
    print(f"Dados: {n}; fold {a.fold}; treino {len(train_idx)}; validacao {len(val_idx)}", flush=True)
    print(f"V3 accuracy fold {a.fold}: {accuracy_score(y[val_idx], vpred[val_idx]):.6f}", flush=True)
    return train, y, train_idx, val_idx, vpred


def tokens_for_mlm(texts, tokenizer, max_length):
    """Inclui inicio e final de textos longos; na validacao nao entra nenhum texto."""
    # BERTimbau usa [CLS] texto [SEP]. A montagem explicita funciona com os
    # tokenizers do Transformers 5, que nao expõem build_inputs_with_special_tokens.
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer sem [CLS] ou [SEP]; este experimento requer BERT")
    n_special = tokenizer.num_special_tokens_to_add(pair=False)
    if n_special != 2:
        raise ValueError(f"Esperados 2 tokens especiais para BERT; encontrados {n_special}")
    window = max_length - n_special
    if window < 8:
        raise ValueError("mlm-max-length pequeno")
    blocks = []
    extra = 0
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not tokens:
            continue
        blocks.append({"input_ids": [cls_id, *tokens[:window], sep_id]})
        if len(tokens) > window:
            blocks.append({"input_ids": [cls_id, *tokens[-window:], sep_id]})
            extra += 1
    if not blocks:
        raise RuntimeError("Sem documentos para MLM")
    print(f"TAPT: {len(texts)} textos; {len(blocks)} janelas; {extra} janelas de cauda", flush=True)
    return blocks


def optimize(model, loader, torch, device, epochs, lr, accumulation, weight_decay, warmup_ratio, name):
    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = math.ceil(len(loader) / accumulation) * epochs
    sched = get_linear_schedule_with_warmup(optimizer, int(steps * warmup_ratio), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    use_amp = device.type == "cuda"
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        count = 0
        for i, batch in enumerate(loader):
            size = int(batch["input_ids"].shape[0])
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp):
                output = model(**batch)
                loss = output.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss nao-finita em {name}, epoca {epoch}: {loss.item()}")
            group_start = (i // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            scaler.scale(loss / group_size).backward()
            total_loss += float(loss.item()) * size
            count += size
            if (i + 1) % accumulation == 0 or i + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
        record = {"epoch": epoch, "loss_mean": total_loss / count}
        history.append(record)
        print(f"{name}: epoca {epoch}/{epochs}, loss={record['loss_mean']:.5f}", flush=True)
    return history


def run(a, train, y, train_idx, val_idx, vpred):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, DataCollatorForLanguageModeling

    if not torch.cuda.is_available() and not a.allow_cpu:
        raise RuntimeError("CUDA nao disponivel. Use --allow-cpu somente para depuracao; treino sera muito lento")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0), flush=True)
    seed_everything(a.seed, torch)
    out = a.outdir / f"fold{a.fold}_seed{a.seed}"
    if out.exists() and (out / "metricas.json").exists() and not a.overwrite:
        raise FileExistsError(f"Saida ja concluida: {out}; use --overwrite se for intencional")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    train_texts = train.resp_text.astype(str).iloc[train_idx].tolist()
    val_texts = train.resp_text.astype(str).iloc[val_idx].tolist()

    # 1) TAPT. Nunca inclui o texto do fold validado, mesmo sem rotulos.
    blocks = tokens_for_mlm(train_texts, tokenizer, a.mlm_max_length)
    collator_mlm = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True,
                        mlm_probability=a.mlm_probability, pad_to_multiple_of=8)
    loader_mlm = DataLoader(blocks, batch_size=a.mlm_batch, shuffle=True, num_workers=0,
                            collate_fn=collator_mlm)
    mlm_model = AutoModelForMaskedLM.from_pretrained(a.model).to(device)
    hist_mlm = optimize(mlm_model, loader_mlm, torch, device, a.mlm_epochs, a.mlm_lr,
                        a.mlm_accum, a.weight_decay, a.warmup_ratio, "TAPT-MLM")
    adapted_dir = out / "encoder_tapt"
    mlm_model.save_pretrained(adapted_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapted_dir)
    del mlm_model, loader_mlm, collator_mlm, blocks
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2) Treina classificador ternario usando SOMENTE o fold de treino.
    class TextDataset(Dataset):
        def __init__(self, enc, labels):
            self.enc, self.labels = enc, labels
        def __len__(self): return len(self.labels)
        def __getitem__(self, k):
            return {name: values[k] for name, values in self.enc.items()}, int(self.labels[k])

    def collate_clf(batch):
        features, labels = zip(*batch)
        padded = tokenizer.pad(list(features), padding=True, pad_to_multiple_of=8, return_tensors="pt")
        padded["labels"] = torch.tensor(labels, dtype=torch.long)
        return padded

    x_train = tokenizer(train_texts, truncation=True, max_length=a.cls_max_length, padding=False)
    x_val = tokenizer(val_texts, truncation=True, max_length=a.cls_max_length, padding=False)
    tr_loader = DataLoader(TextDataset(x_train, y[train_idx]), batch_size=a.cls_batch,
                           shuffle=True, num_workers=0, collate_fn=collate_clf)
    val_loader = DataLoader(TextDataset(x_val, y[val_idx]), batch_size=a.eval_batch,
                            shuffle=False, num_workers=0, collate_fn=collate_clf)
    seed_everything(a.seed, torch)
    clf = AutoModelForSequenceClassification.from_pretrained(adapted_dir, num_labels=3,
                       id2label={i: c for i, c in enumerate(CLASSES)}, label2id=LABEL2ID).to(device)
    hist_clf = optimize(clf, tr_loader, torch, device, a.cls_epochs, a.cls_lr,
                        a.cls_accum, a.weight_decay, a.warmup_ratio, "TAPT-CLS")
    clf.eval()
    ps = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = clf(**batch).logits
            ps.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs = np.concatenate(ps)
    if len(probs) != len(val_idx) or not np.isfinite(probs).all():
        raise RuntimeError("Predicoes incompletas ou invalidas")
    pred = probs.argmax(axis=1)
    real = y[val_idx]
    v3 = vpred[val_idx]
    n_correct_v3 = int((v3 == real).sum())
    new_correct = (pred == real) & (v3 != real)
    lost_correct = (pred != real) & (v3 == real)
    extras = {}
    if a.bert512.is_file():
        base = pd.read_csv(a.bert512, encoding="utf-8-sig")
        if len(base) != len(y) or not {"row_id", "fold", "y_true_id", "y_pred_id"}.issubset(base.columns):
            raise ValueError("OOF BERT512 com formato incorreto")
        orig_folds = pd.read_csv(a.folds, encoding="utf-8-sig").fold.to_numpy()
        if not np.array_equal(base.row_id.to_numpy(), np.arange(len(y))) or not np.array_equal(base.fold.to_numpy(), orig_folds) or not np.array_equal(base.y_true_id.to_numpy(), y):
            raise ValueError("OOF BERT512 sem alinhamento correto")
        extras["bertimbau_512_accuracy_fold"] = float(accuracy_score(real, base.y_pred_id.to_numpy()[val_idx]))
    result = {
        "fold": a.fold, "seed": a.seed, "n_treino": int(len(train_idx)), "n_val": int(len(val_idx)),
        "config": {"model": a.model, "mlm_max_length": a.mlm_max_length,
                   "cls_max_length": a.cls_max_length, "mlm_epochs": a.mlm_epochs,
                   "cls_epochs": a.cls_epochs, "mlm_lr": a.mlm_lr, "cls_lr": a.cls_lr,
                   "mlm_batch": a.mlm_batch, "mlm_accum": a.mlm_accum,
                   "cls_batch": a.cls_batch, "cls_accum": a.cls_accum,
                   "mlm_probability": a.mlm_probability},
        "tapt_accuracy": float(accuracy_score(real, pred)),
        "tapt_macro_f1": float(f1_score(real, pred, average="macro")),
        "v3_accuracy_fold": float(accuracy_score(real, v3)),
        "v3_macro_f1_fold": float(f1_score(real, v3, average="macro")),
        "delta_accuracy_pp": float(100 * (accuracy_score(real, pred) - accuracy_score(real, v3))),
        "recuperados_vs_v3": int(new_correct.sum()),
        "perdidos_vs_v3": int(lost_correct.sum()),
        "matriz_tapt": confusion_matrix(real, pred, labels=[0,1,2]).tolist(),
        "classification_report": classification_report(real, pred, labels=[0,1,2], target_names=CLASSES,
                                                            output_dict=True, zero_division=0),
        "mlm_history": hist_mlm, "cls_history": hist_clf,
        "elapsed_seconds": time.perf_counter() - start, **extras
    }
    prediction = pd.DataFrame({"row_id": val_idx, "fold": a.fold, "y_true_id": real,
             "pred_v3_id": v3, "pred_tapt_id": pred,
             "prob_c1": probs[:, 0], "prob_c234": probs[:, 1], "prob_c5": probs[:, 2],
             "v3_correto": v3 == real, "tapt_correto": pred == real,
             "n_palavras": [len(t.split()) for t in val_texts]})
    prediction.to_csv(out / "predicoes_fold.csv", index=False, encoding="utf-8-sig")
    (out / "metricas.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TAPT fold {a.fold}: acc={result['tapt_accuracy']:.5f}, V3={result['v3_accuracy_fold']:.5f}, "
          f"recuperou={result['recuperados_vs_v3']}, perdeu={result['perdidos_vs_v3']}", flush=True)
    print(f"Resultados: {out}", flush=True)


def main():
    a = options()
    if a.self_test:
        assert len(LABEL2ID) == 3 and len(set(LABEL2ID.values())) == 3
        assert (a.outdir / f"fold{a.fold}_seed{a.seed}").name.startswith("fold")

        class _TokenizerBertSimulado:
            # Reproduz a interface essencial do BertTokenizer v5 (sem build_inputs_with_special_tokens).
            cls_token_id, sep_token_id = 101, 102

            def num_special_tokens_to_add(self, pair=False):
                assert pair is False
                return 2

            def encode(self, text, add_special_tokens=False, truncation=False):
                assert add_special_tokens is False and truncation is False
                return [int(x) for x in text.split()] if text.strip() else []

        teste = tokens_for_mlm(["1 2 3", "1 2 3 4 5 6 7 8 9", ""],
                               _TokenizerBertSimulado(), max_length=10)
        assert [b["input_ids"] for b in teste] == [
            [101, 1, 2, 3, 102],
            [101, 1, 2, 3, 4, 5, 6, 7, 8, 102],
            [101, 2, 3, 4, 5, 6, 7, 8, 9, 102],
        ]
        assert all(len(b["input_ids"]) <= 10 for b in teste)
        print("Self-test OK: janelas BERT [CLS]/[SEP], cabeca/cauda e tamanho. ")
        print("Nao testa download, collator nem execucao neural em GPU.")
        return
    train, y, tr, va, v3 = load_data(a)
    if a.check_data:
        print("Preflight de dados OK. Nenhum treinamento realizado.")
        return
    run(a, train, y, tr, va, v3)


if __name__ == "__main__":
    main()
