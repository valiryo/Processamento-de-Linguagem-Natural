#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP2: benchmark controlado do moBERTo (ModernBERT portugues).

Usa exclusivamente dados de treino do EP1, respeitando folds congelados.
Nenhum dado do teste1/teste2 participa de ajuste ou escolha de modelos.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

CLASSES = ("c1", "c234", "c5")
LABEL2ID = {x: i for i, x in enumerate(CLASSES)}
DEFAULT_MODEL = "Tropic-AI/moBERTo"


def options():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train", type=Path, default=Path("data/raw/train.xlsx"))
    p.add_argument("--folds", type=Path, default=Path("data/splits/folds.csv"))
    p.add_argument("--v3", type=Path, default=Path("results/ensemble_majority_5_v3_norberto_oof.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/ep2/moberto_512"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fold", type=int, choices=list(range(5)), default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=2, help="batch fisico reduzido para GPU de 4 GB")
    p.add_argument("--accum", type=int, default=8, help="batch efetivo = batch * accum = 16")
    p.add_argument("--eval-batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.max_length < 32 or a.epochs < 1 or a.batch < 1 or a.accum < 1 or a.eval_batch < 1:
        p.error("max-length >=32; epochs, batch, accum e eval-batch >=1")
    if a.lr <= 0 or a.weight_decay < 0 or not 0 <= a.warmup_ratio < 1:
        p.error("Parametros de otimizacao invalidos")
    return a


def seed_all(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(a):
    import pandas as pd
    for path in [a.train, a.folds, a.v3]:
        if not path.exists():
            raise FileNotFoundError(f"Arquivo ausente: {path}")
    df = pd.read_excel(a.train, engine="openpyxl").reset_index(drop=True)
    folds = pd.read_csv(a.folds, encoding="utf-8-sig")
    v3 = pd.read_csv(a.v3, encoding="utf-8-sig")
    n = len(df)
    if n != 20092 or len(folds) != n or len(v3) != n:
        raise ValueError("Treino, folds e V3 devem conter 20.092 linhas")
    if not {"resp_text", "clarity"}.issubset(df.columns):
        raise ValueError("Treino sem resp_text e clarity")
    if not {"row_index", "fold"}.issubset(folds.columns):
        raise ValueError("folds.csv sem row_index/fold")
    if not {"row_id", "fold", "y_true_id", "y_pred_id_v3"}.issubset(v3.columns):
        raise ValueError("V3 OOF sem colunas obrigatorias")
    y = df.clarity.map(LABEL2ID)
    if y.isna().any() or df.resp_text.isna().any():
        raise ValueError("Texto ausente ou rotulo desconhecido")
    y = y.astype(int).to_numpy()
    fold = folds.fold.to_numpy(dtype=int)
    ids = np.arange(n)
    if not np.array_equal(folds.row_index.to_numpy(), ids) or not np.array_equal(v3.row_id.to_numpy(), ids):
        raise ValueError("Row IDs nao correspondem a train.xlsx")
    if sorted(set(fold)) != list(range(5)) or not np.array_equal(v3.fold.to_numpy(), fold):
        raise ValueError("Folds nao coincidem")
    if not np.array_equal(v3.y_true_id.to_numpy(), y):
        raise ValueError("Rotulos do V3 nao correspondem ao treino")
    vpred = v3.y_pred_id_v3.to_numpy(dtype=int)
    if not np.isclose(np.mean(vpred == y), 0.47566195500696795, atol=1e-10):
        raise ValueError("V3 nao coincide com benchmark do EP1")
    if (pd.DataFrame({"text": df.resp_text, "fold": fold})
            .groupby("text", sort=False).fold.nunique().gt(1).any()):
        raise ValueError("Leakage: mesmo texto aparece em mais de um fold")
    idx_train = np.flatnonzero(fold != a.fold)
    idx_val = np.flatnonzero(fold == a.fold)
    print(f"Dados verificados: treino={len(idx_train)} validacao={len(idx_val)} fold={a.fold}", flush=True)
    print(f"V3 fold {a.fold}: {np.mean(y[idx_val] == vpred[idx_val]):.6f}", flush=True)
    return df, y, idx_train, idx_val, vpred


def run(a, df, y, idx_train, idx_val, vpred):
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise RuntimeError("CUDA nao disponivel; para depurar na CPU use --allow-cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("CUDA:", torch.cuda.get_device_name(0), flush=True)
        torch.cuda.reset_peak_memory_stats()
    seed_all(a.seed, torch)
    outdir = a.outdir / f"fold{a.fold}_seed{a.seed}"
    if (outdir / "metricas.json").exists() and not a.overwrite:
        raise FileExistsError(f"Resultados ja existem: {outdir}; use --overwrite para substituir")
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    texts_train = df.resp_text.iloc[idx_train].astype(str).tolist()
    texts_val = df.resp_text.iloc[idx_val].astype(str).tolist()
    print("Tokenizando", len(texts_train), "textos de treino e", len(texts_val), "de validacao", flush=True)
    xtr = tokenizer(texts_train, truncation=True, padding=False, max_length=a.max_length)
    xva = tokenizer(texts_val, truncation=True, padding=False, max_length=a.max_length)

    class TextDataset(Dataset):
        def __init__(self, enc, labels):
            self.enc, self.labels = enc, labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, j):
            return ({key: val[j] for key, val in self.enc.items()}, int(self.labels[j]))

    def collate(batch):
        inputs, labels = zip(*batch)
        result = tokenizer.pad(list(inputs), padding=True, pad_to_multiple_of=8, return_tensors="pt")
        result["labels"] = torch.tensor(labels, dtype=torch.long)
        return result

    gen = torch.Generator()
    gen.manual_seed(a.seed)
    train_loader = DataLoader(TextDataset(xtr, y[idx_train]), batch_size=a.batch, shuffle=True,
                              generator=gen, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(TextDataset(xva, y[idx_val]), batch_size=a.eval_batch,
                            shuffle=False, collate_fn=collate, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=3, id2label={i:c for i,c in enumerate(CLASSES)}, label2id=LABEL2ID)
    if a.gradient_checkpointing:
        if not model.supports_gradient_checkpointing:
            raise RuntimeError("O modelo nao suporta gradient checkpointing; use --no-gradient-checkpointing")
        model.gradient_checkpointing_enable()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = math.ceil(len(train_loader) / a.accum) * a.epochs
    sched = get_linear_schedule_with_warmup(opt, int(steps * a.warmup_ratio), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    for ep in range(1, a.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses, examples = 0., 0
        for j, batch in enumerate(train_loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss nao finita (epoca {ep}, passo {j}): {loss.item()}")
            group_start = (j // a.accum) * a.accum
            group_size = min(a.accum, len(train_loader)-group_start)
            scaler.scale(loss / group_size).backward()
            losses += float(loss.item()) * batch["labels"].shape[0]
            examples += batch["labels"].shape[0]
            if (j + 1) % a.accum == 0 or j + 1 == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
        history.append({"epoch":ep, "train_loss": losses/examples})
        print(f"EP {ep}/{a.epochs}: loss={history[-1]['train_loss']:.5f}", flush=True)

    model.eval()
    probs_list=[]
    with torch.inference_mode():
        for batch in val_loader:
            x={k:v.to(device) for k,v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.bfloat16, enabled=use_amp):
                logits=model(**x).logits
            probs_list.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    probs=np.concatenate(probs_list)
    assert len(probs)==len(idx_val) and np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-4)
    pred=probs.argmax(1)
    gold=y[idx_val]
    prev=vpred[idx_val]
    recovered=int(((pred==gold)&(prev!=gold)).sum())
    lost=int(((pred!=gold)&(prev==gold)).sum())
    result={
        "fold":a.fold,"seed":a.seed,"n_train":len(idx_train),"n_val":len(idx_val),
        "model":a.model,"max_length":a.max_length,"epochs":a.epochs,"lr":a.lr,
        "batch":a.batch,"accum":a.accum,"gradient_checkpointing":a.gradient_checkpointing,
        "accuracy":float(accuracy_score(gold,pred)),"macro_f1":float(f1_score(gold,pred,average="macro")),
        "accuracy_v3":float(accuracy_score(gold,prev)),
        "delta_accuracy_pp":float(100*(np.mean(pred==gold)-np.mean(prev==gold))),
        "recuperados":recovered,"perdidos":lost,
        "confusion_matrix":confusion_matrix(gold,pred,labels=[0,1,2]).tolist(),
        "classification_report":classification_report(gold,pred,labels=[0,1,2],target_names=CLASSES,zero_division=0,output_dict=True),
        "history":history,"elapsed_seconds":time.perf_counter()-start,
        "gpu_peak_gb":float(torch.cuda.max_memory_allocated()/1024**3) if use_amp else None,
    }
    pd.DataFrame({"row_id":idx_val, "fold":a.fold,"y_true_id":gold,
        "pred_v3_id":prev,"pred_moberto_id":pred,
        "prob_c1":probs[:,0],"prob_c234":probs[:,1],"prob_c5":probs[:,2]
    }).to_csv(outdir/"predicoes_fold.csv",index=False,encoding="utf-8-sig")
    (outdir/"metricas.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"moBERTo: {result['accuracy']:.6f}, V3: {result['accuracy_v3']:.6f}, "
          f"delta: {result['delta_accuracy_pp']:+.3f} p.p., recuperados={recovered}, perdidos={lost}",flush=True)
    print(f"Resultados: {outdir}",flush=True)


def main():
    a=options()
    if a.self_test:
        assert CLASSES==("c1","c234","c5") and len(set(LABEL2ID.values()))==3
        x=np.array([[.2,.7,.1],[.4,.1,.5]])
        assert np.array_equal(x.argmax(1),[1,2])
        assert math.ceil(7/3)==3
        print("Self-test local OK (CLI/configuracoes); nao valida download, CUDA ou modelagem remota")
        return
    df,y,tr,va,v3=load_data(a)
    if a.check_data:
        print("Preflight OK; nenhum modelo treinado")
        return
    run(a,df,y,tr,va,v3)


if __name__=="__main__":
    main()
