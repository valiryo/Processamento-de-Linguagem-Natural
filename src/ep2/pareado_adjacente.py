#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP2: dois classificadores BERTimbau independentes para classes adjacentes.

Treino/validação: somente train.xlsx e folds.csv (nunca teste1/2).
Pares: c1 vs c234 e c234 vs c5. Fusão por probabilidades condicionais,
sem limiares escolhidos a partir de rótulos de validação.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

CLASSES = ("c1", "c234", "c5")
L2I = {v: k for k, v in enumerate(CLASSES)}
MODEL = "neuralmind/bert-base-portuguese-cased"


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--train", type=Path, default=Path("data/raw/train.xlsx"))
    p.add_argument("--folds", type=Path, default=Path("data/splits/folds.csv"))
    p.add_argument("--v3", type=Path, default=Path("results/ensemble_majority_5_v3_norberto_oof.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/ep2/pareado_adjacente"))
    p.add_argument("--fold", type=int, choices=range(5), default=0)
    p.add_argument("--model", type=str, default=MODEL)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.max_length < 8 or any(getattr(a,k)<1 for k in ("epochs", "batch_size", "eval_batch_size", "grad_accum")):
        p.error("Parâmetros de batch/epochs devem ser positivos e max-length >= 8")
    if not (0 <= a.warmup_ratio < 1) or a.lr<=0:
        p.error("warmup-ratio deve estar em [0,1) e lr > 0")
    return a


def load_inputs(a):
    for path in (a.train, a.folds, a.v3):
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo não localizado: {path}")
    train = pd.read_excel(a.train, engine="openpyxl").reset_index(drop=True)
    folds = pd.read_csv(a.folds, encoding="utf-8-sig")
    oof = pd.read_csv(a.v3, encoding="utf-8-sig")
    n = len(train)
    if n != 20092 or len(folds)!=n or len(oof)!=n:
        raise ValueError("Esperadas 20.092 linhas em treino, folds e V3 OOF")
    if not {"resp_text", "clarity"}.issubset(train):
        raise ValueError("Treino deve ter resp_text e clarity")
    if not {"row_index", "fold"}.issubset(folds):
        raise ValueError("folds.csv deve ter row_index e fold")
    if not {"row_id", "fold", "y_true_id", "y_pred_id_v3"}.issubset(oof):
        raise ValueError("V3 OOF sem campos obrigatórios")
    y = train.clarity.map(L2I)
    if y.isna().any() or train.resp_text.isna().any():
        raise ValueError("Treino com rótulo desconhecido/texto ausente")
    y = y.astype(int).to_numpy()
    ids = np.arange(n)
    if not np.array_equal(folds.row_index.to_numpy(), ids) or not np.array_equal(oof.row_id.to_numpy(),ids):
        raise ValueError("Ordem das linhas não coincide com row_id")
    if not np.array_equal(oof.fold.to_numpy(),folds.fold.to_numpy()):
        raise ValueError("OOF e folds divergentes")
    if not np.array_equal(oof.y_true_id.to_numpy(),y):
        raise ValueError("OOF e rótulos do treino divergentes")
    if sorted(folds.fold.unique()) != [0,1,2,3,4]:
        raise ValueError("Folds devem ser 0 a 4")
    if (pd.DataFrame({"t": train.resp_text.astype(str),"f": folds.fold})
          .groupby("t",sort=False).f.nunique()>1).any():
        raise ValueError("Leakage entre folds: texto idêntico em partições distintas")
    v3 = oof.y_pred_id_v3.astype(int).to_numpy()
    if not np.isin(v3,[0,1,2]).all() or abs(accuracy_score(y,v3)-0.475661955)>1e-7:
        raise ValueError("V3 não corresponde à referência EP1")
    fi = folds.fold.to_numpy()
    tr = np.flatnonzero(fi != a.fold)
    va = np.flatnonzero(fi == a.fold)
    print(f"Dados OK: {n} exemplos; fold {a.fold}, treino {len(tr)}, val {len(va)}",flush=True)
    print(f"V3 fold {a.fold}: {accuracy_score(y[va], v3[va]):.8f}",flush=True)
    print("Treino por classe:", dict(zip(CLASSES, np.bincount(y[tr],minlength=3).tolist())),flush=True)
    return train, y, v3, tr, va


def fusion_from_conditional(p_234_over_1, p_5_over_234):
    """Recupera P(c1), P(c234), P(c5) via odds adjacentes.

    Se A=P(c234|c1/c234), B=P(c5|c234/c5):
    odds(c234/c1)=A/(1-A), odds(c5/c234)=B/(1-B).
    Logo logits proporcionais a (0, logit(A), logit(A)+logit(B)).
    """
    a = np.clip(np.asarray(p_234_over_1,dtype=np.float64),1e-6,1-1e-6)
    b = np.clip(np.asarray(p_5_over_234,dtype=np.float64),1e-6,1-1e-6)
    if a.shape!=b.shape or a.ndim!=1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Vetores de probabilidades inválidos")
    la=np.log(a)-np.log1p(-a)
    lb=np.log(b)-np.log1p(-b)
    logits=np.column_stack((np.zeros_like(la),la,la+lb))
    logits-=logits.max(axis=1,keepdims=True)
    ps=np.exp(logits)
    ps/=ps.sum(axis=1,keepdims=True)
    return ps


def self_test():
    p=fusion_from_conditional([.5,.8,.2],[.5,.2,.8])
    assert np.allclose(p.sum(1),1)
    assert np.argmax(p,axis=1).tolist()==[0,1,2],p
    assert np.all(np.isfinite(fusion_from_conditional([0.,1.],[1.,0.])))
    print("Fusão por odds condicionais: OK")


def set_seed(seed,torch):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def train_pair(pair, train_text, y, tr, va, tokenizer, a, torch, run_dir):
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    i,j=pair
    subset=tr[np.isin(y[tr],[i,j])]
    # Probabilidade positiva sempre corresponde à segunda classe do par.
    yy=(y[subset]==j).astype(np.int64)
    texts=[train_text[t] for t in subset]
    vtexts=[train_text[t] for t in va]
    print(f"Par {CLASSES[i]} vs {CLASSES[j]}: {len(subset)} treino, {len(va)} validação",flush=True)
    enc_tr=tokenizer(texts, truncation=True,max_length=a.max_length,padding=False)
    enc_val=tokenizer(vtexts,truncation=True,max_length=a.max_length,padding=False)

    class PairDataset(Dataset):
        def __init__(self,enc,labels): self.enc=enc;self.labels=labels
        def __len__(self):return len(self.labels)
        def __getitem__(self,k):return {key:self.enc[key][k] for key in self.enc}, int(self.labels[k])

    def collate(batch):
        features,labels=zip(*batch)
        pad=tokenizer.pad(list(features),padding=True,pad_to_multiple_of=8,return_tensors="pt")
        pad["labels"]=torch.tensor(labels,dtype=torch.long)
        return pad

    dl_tr=DataLoader(PairDataset(enc_tr,yy),batch_size=a.batch_size,shuffle=True,num_workers=0,collate_fn=collate)
    dl_val=DataLoader(PairDataset(enc_val,np.zeros(len(va),dtype=np.int64)),batch_size=a.eval_batch_size,
                      shuffle=False,num_workers=0,collate_fn=collate)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp=device.type=="cuda"
    # Distinct, reproducible seed for each independently fitted pair.
    set_seed(a.seed + 100*i + j, torch)
    model=AutoModelForSequenceClassification.from_pretrained(a.model,num_labels=2,
            id2label={0:CLASSES[i],1:CLASSES[j]},label2id={CLASSES[i]:0,CLASSES[j]:1}).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    nsteps=math.ceil(len(dl_tr)/a.grad_accum)*a.epochs
    sched=get_linear_schedule_with_warmup(optimizer,int(nsteps*a.warmup_ratio),nsteps)
    scaler=torch.cuda.amp.GradScaler(enabled=amp)
    criterion=torch.nn.CrossEntropyLoss()
    history=[]
    for epoch in range(1,a.epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        tot=0.0;count=0;nb=len(dl_tr)
        for bi,batch in enumerate(dl_tr):
            labels=batch.pop("labels").to(device)
            xs={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                logits=model(**xs).logits
                loss=criterion(logits.float(),labels)
            group_begin=(bi//a.grad_accum)*a.grad_accum
            group_size=min(a.grad_accum,nb-group_begin)
            scaler.scale(loss/group_size).backward()
            count+=len(labels);tot+=loss.item()*len(labels)
            if (bi+1)%a.grad_accum==0 or bi+1==nb:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();sched.step()
                optimizer.zero_grad(set_to_none=True)
        loss_epoch=tot/count
        if not math.isfinite(loss_epoch):raise RuntimeError("Loss inválida no treino: cheque FP16/gradientes")
        history.append({"epoch":epoch,"train_loss":loss_epoch})
        print(f"  par {CLASSES[i]}-{CLASSES[j]} época {epoch}: loss {loss_epoch:.5f}",flush=True)

    model.eval();prob=[]
    with torch.no_grad():
        for batch in dl_val:
            batch.pop("labels")
            xs={k:v.to(device) for k,v in batch.items()}
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=amp):
                output=model(**xs).logits
            prob.extend(torch.softmax(output.float(),dim=-1)[:,1].cpu().numpy().tolist())
    prob=np.asarray(prob,dtype=np.float64)
    if len(prob)!=len(va) or not np.isfinite(prob).all():
        raise RuntimeError("Inferência retornou valores inválidos")
    name=f"par_{CLASSES[i]}_{CLASSES[j]}"
    if not a.no_save_model:
        model.save_pretrained(run_dir/name)
        tokenizer.save_pretrained(run_dir/name)
    del model,optimizer,sched,scaler,dl_tr,dl_val,enc_tr,enc_val
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return prob,history


def main():
    a=get_args()
    if a.self_test: self_test();return
    train,y,v3,tr,va=load_inputs(a)
    if a.check_data:return
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Requer torch + transformers, ver README") from exc
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise RuntimeError("CUDA não encontrado; use GPU NVIDIA ou --allow-cpu (lento)")
    set_seed(a.seed,torch)
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    run_dir=a.outdir/f"fold{a.fold}_seed{a.seed}"
    run_dir.mkdir(parents=True,exist_ok=True)
    cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    cfg.update({"protocol":"independent adjacent-pair encoders, pairwise odds coupling, fixed argmax; no test data",
                "pairs":[["c1","c234"],["c234","c5"]],"threshold_tuning":False,"device":"cuda" if torch.cuda.is_available() else "cpu"})
    (run_dir/"config.json").write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding="utf-8")
    start=time.perf_counter()
    txt=train.resp_text.astype(str).tolist()
    p_a,h_a=train_pair((0,1),txt,y,tr,va,tokenizer,a,torch,run_dir)
    p_b,h_b=train_pair((1,2),txt,y,tr,va,tokenizer,a,torch,run_dir)
    probs=fusion_from_conditional(p_a,p_b)
    pred=probs.argmax(1)
    yy=y[va];base=v3[va]
    base_ok=base==yy;new_ok=pred==yy
    scores={"fold":a.fold,"n":len(va),"v3_accuracy":float(accuracy_score(yy,base)),
            "pareado_accuracy":float(accuracy_score(yy,pred)),
            "accuracy_delta_pp":float(100*(new_ok.mean()-base_ok.mean())),
            "v3_macro_f1":float(f1_score(yy,base,average="macro")),
            "pareado_macro_f1":float(f1_score(yy,pred,average="macro")),
            "matriz_pareado":confusion_matrix(yy,pred,labels=[0,1,2]).tolist(),
            "matriz_v3":confusion_matrix(yy,base,labels=[0,1,2]).tolist(),
            "novos_acertos":int((~base_ok&new_ok).sum()),
            "acertos_perdidos":int((base_ok&~new_ok).sum()),
            "tempo_segundos":float(time.perf_counter()-start),
            "history_c1_c234":h_a,"history_c234_c5":h_b}
    df=pd.DataFrame({"row_id":va,"fold":a.fold,"y_true_id":yy,"pred_v3_id":base,
        "p_c234_sobre_c1_c234":p_a,"p_c5_sobre_c234_c5":p_b,
        "prob_c1":probs[:,0],"prob_c234":probs[:,1],"prob_c5":probs[:,2],
        "pred_pareado_id":pred,"v3_correto":base_ok,"pareado_correto":new_ok,
        "n_palavras":[len(re.findall(r"\b\w+\b",txt[k],flags=re.UNICODE)) for k in va]})
    df.to_csv(run_dir/"predicoes_fold.csv",index=False,encoding="utf-8-sig")
    (run_dir/"metricas.json").write_text(json.dumps(scores,indent=2,ensure_ascii=False),encoding="utf-8")
    print("\n== EXPERIMENTO 02 ==",flush=True)
    print(f"V3: {scores['v3_accuracy']:.5f}",flush=True)
    print(f"Pares: {scores['pareado_accuracy']:.5f}",flush=True)
    print(f"Ganho: {scores['accuracy_delta_pp']:+.3f} p.p.",flush=True)
    print("Salvo em:",run_dir.resolve(),flush=True)

if __name__=="__main__":
    main()
