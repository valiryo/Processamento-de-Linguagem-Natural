#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP2: valida consenso V3+TAPT+moBERTo com regra fixa, sem treinar.

Regra registrada: caso TAPT e moBERTo concordem entre si e discordem do V3,
substituir o V3 pelo rótulo consensual; caso contrário, manter V3.

ATENCAO: regra definida depois de observar fold 0; primeiro fold e exploratorio.
Nao usar teste1/teste2 para construir, ajustar ou escolher esta regra.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

CLASSES = ("c1", "c234", "c5")


def metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0,1,2], average="macro", zero_division=0)),
        "acertos": int((y == pred).sum()),
        "recall_classe": {c: float(recall_score(y, pred, labels=[i], average="macro", zero_division=0)) for i,c in enumerate(CLASSES)},
    }


def paired(y, baseline, candidate):
    old, new = baseline == y, candidate == y
    recovered = int((~old & new).sum())
    lost = int((old & ~new).sum())
    return {
        "corrigidos": recovered,
        "perdidos": lost,
        "saldo": recovered-lost,
        "mudancas": int((baseline != candidate).sum()),
        "delta_pp": 100*(recovered-lost)/len(y),
        "mcnemar_exato_p_exploratorio": float(binomtest(recovered, recovered+lost, 0.5).pvalue) if recovered+lost else 1.0,
    }


def validate_partition(df, ref, fold, name, pred_col, common):
    required = {"row_id", "fold", "y_true_id", "pred_v3_id", pred_col}
    if not required.issubset(df.columns):
        raise ValueError(f"{name}: faltam colunas {required-set(df.columns)}")
    if not df.row_id.is_unique or not df.fold.eq(fold).all():
        raise ValueError(f"{name}: row_id duplicado ou fold incorreto")
    ref_fold = ref.loc[ref.fold == fold, ["row_id", "fold", "y_true_id", "y_pred_id_v3"]].copy()
    if set(df.row_id) != set(ref_fold.row_id):
        raise ValueError(f"{name}: row_id nao cobre exatamente o fold {fold} do V3")
    ref_order = ref_fold.set_index("row_id")
    df_order = df.set_index("row_id")
    if not np.array_equal(df_order.loc[ref_order.index, "y_true_id"].to_numpy(), ref_order.y_true_id.to_numpy()):
        raise ValueError(f"{name}: y_true_id diverge do V3")
    if not np.array_equal(df_order.loc[ref_order.index, "pred_v3_id"].to_numpy(), ref_order.y_pred_id_v3.to_numpy()):
        raise ValueError(f"{name}: pred_v3_id diverge do V3")
    for c in ("y_true_id", "pred_v3_id", pred_col):
        if not df[c].isin([0,1,2]).all():
            raise ValueError(f"{name}: classe invalida {c}")
    if common:
        z = df[["prob_c1", "prob_c234", "prob_c5"]].to_numpy(float)
        if not np.isfinite(z).all() or (z < 0).any() or not np.allclose(z.sum(1),1,atol=1e-3):
            raise ValueError(f"{name}: probabilidades invalidas")
    return df


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v3",type=Path,default=Path("results/ensemble_majority_5_v3_norberto_oof.csv"))
    p.add_argument("--tapt",type=Path,default=Path("results/ep2/tapt_bertimbau/analise_cv/tapt_oof_completo.csv"))
    p.add_argument("--moberto-dir",type=Path,default=Path("results/ep2/moberto_512"))
    p.add_argument("--out",type=Path,default=Path("results/ep2/consenso_tapt_moberto"))
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    v3=pd.read_csv(a.v3,encoding="utf-8-sig")
    tapt=pd.read_csv(a.tapt,encoding="utf-8-sig")
    req={"row_id","fold","y_true_id","y_pred_id_v3"}
    if not req.issubset(v3.columns) or len(v3)!=20092 or not v3.row_id.is_unique:
        raise ValueError("Referencia V3 invalida")
    if sorted(v3.fold.unique()) != [0,1,2,3,4] or not np.isclose((v3.y_true_id==v3.y_pred_id_v3).mean(),0.475661955,atol=1e-7):
        raise ValueError("V3 nao corresponde ao benchmark OOF")
    if not tapt.row_id.is_unique or len(tapt)!=20092:
        raise ValueError("TAPT OOF deve ter 20.092 row_ids distintos")
    if set(tapt.row_id) != set(v3.row_id):
        raise ValueError("TAPT OOF nao cobre todos os row_ids do V3")
    out=[]
    for f in range(5):
        path=a.moberto_dir / f"fold{f}_seed{a.seed}" / "predicoes_fold.csv"
        meta=a.moberto_dir / f"fold{f}_seed{a.seed}" / "metricas.json"
        if not path.is_file():
            if meta.exists():
                raise ValueError(f"Fold {f}: metricas presentes mas predicoes ausentes")
            continue
        if not meta.is_file():
            raise ValueError(f"Fold {f}: falta metricas.json")
        m=json.loads(meta.read_text(encoding="utf-8-sig"))
        dm=pd.read_csv(path,encoding="utf-8-sig")
        dt=tapt[tapt.fold==f].copy()
        validate_partition(dm,v3,f,"moBERTo","pred_moberto_id",True)
        validate_partition(dt,v3,f,"TAPT","pred_tapt_id",True)
        if int(m.get('fold',-1))!=f or int(m.get('seed',-1))!=a.seed or int(m.get('n_val',-1))!=len(dm):
            raise ValueError(f"Fold {f}: metricas moBERTo nao correspondem ao CSV")
        y=dm.y_true_id.to_numpy(); pred=dm.pred_moberto_id.to_numpy()
        if not np.isclose((y==pred).mean(),float(m.get('accuracy',-1)),atol=1e-7):
            raise ValueError(f"Fold {f}: accuracy moBERTo diverge de metricas.json")
        merged=dm.merge(dt[["row_id","fold","pred_tapt_id"]],on=["row_id","fold"],validate="one_to_one")
        if len(merged)!=len(dm):
            raise ValueError(f"Fold {f}: TAPT e moBERTo sem correspondencia completa")
        v=merged.pred_v3_id.to_numpy(); t=merged.pred_tapt_id.to_numpy(); mob=merged.pred_moberto_id.to_numpy()
        consensus=(t==mob)&(t!=v)
        merged["pred_consenso_id"]=np.where(consensus,t,v)
        merged["consenso_mudou_v3"]=consensus
        out.append(merged[["row_id","fold","y_true_id","pred_v3_id","pred_tapt_id","pred_moberto_id","pred_consenso_id","consenso_mudou_v3"]])
    if not out:
        raise ValueError("Nenhum fold moBERTo encontrado em " + str(a.moberto_dir))
    data=pd.concat(out,ignore_index=True).sort_values("row_id")
    if not data.row_id.is_unique:
        raise ValueError("Duplicidade entre folds")
    all_fold=sorted(map(int,data.fold.unique()))
    result={"regra":"Se TAPT e moBERTo concordam em classe diferente da V3, substitui; caso contrario V3",
            "folds_avaliados":all_fold,"n_avaliado":len(data),
            "nota":"Fold 0 foi utilizado na descoberta da regra; nao tratar seus p-valores como confirmatorios. Mesmo a CV OOF tem limitacoes por reutilizacao de dados para selecao.",
            "por_fold":[]}
    def do_eval(df):
        y=df.y_true_id.to_numpy()
        return {"n":len(y),"v3":metrics(y,df.pred_v3_id.to_numpy()),
                "tapt":metrics(y,df.pred_tapt_id.to_numpy()),
                "moberto":metrics(y,df.pred_moberto_id.to_numpy()),
                "consenso":metrics(y,df.pred_consenso_id.to_numpy()),
                "contra_v3":paired(y,df.pred_v3_id.to_numpy(),df.pred_consenso_id.to_numpy()),
                "matriz_consenso":confusion_matrix(y,df.pred_consenso_id,labels=[0,1,2]).tolist()}
    for fold in all_fold:
        result["por_fold"].append({"fold":fold,**do_eval(data[data.fold==fold])})
    result["total_disponivel"] = do_eval(data)
    if len(all_fold)>1:
        result["folds_1_a_4"] = do_eval(data[data.fold!=0])
    if len(all_fold)==5:
        if len(data)!=20092 or set(data.row_id)!=set(v3.row_id):
            raise ValueError("CV dita completa mas faltam row_ids")
        result["status"]="CV COMPLETA"
    else:
        result["status"]="CV PARCIAL"
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/"resumo.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    data.to_csv(a.out/"predicoes_consenso.csv",index=False,encoding="utf-8-sig")
    lines=["# EP2 — Consenso TAPT + moBERTo como corretor do Ensemble V3","",
           "**Regra fixa:** sobrescreve V3 somente quando TAPT e moBERTo concordam entre si e discordam de V3.",
           "", "**Cuidado metodologico:** regra formulada apos observar o fold 0; ele nao e teste independente. Nenhum dado de teste1/teste2 foi utilizado.",
           "",f"Status: {result['status']} | folds: {all_fold} | n={len(data)}", "",
           "| Fold | N | V3 | TAPT | moBERTo | Consenso | Ganho consenso (p.p.) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for e in result['por_fold']:
        lines.append(f"| {e['fold']} | {e['n']} | {100*e['v3']['accuracy']:.3f}% | {100*e['tapt']['accuracy']:.3f}% | {100*e['moberto']['accuracy']:.3f}% | {100*e['consenso']['accuracy']:.3f}% | {e['contra_v3']['delta_pp']:+.3f} |")
    def summary_block(name,d):
        return [f"## {name}",f"V3: {d['v3']['accuracy']:.6f}; consenso: {d['consenso']['accuracy']:.6f}; delta: {d['contra_v3']['delta_pp']:+.3f} p.p.",
                f"Correcoes: {d['contra_v3']['corrigidos']}; perdas: {d['contra_v3']['perdidos']}; saldo: {d['contra_v3']['saldo']}; p exploratorio: {d['contra_v3']['mcnemar_exato_p_exploratorio']:.5g}.",""]
    lines += [""] + summary_block("Todos os folds disponiveis",result['total_disponivel'])
    if 'folds_1_a_4' in result:
        lines += summary_block("Apenas folds 1–4 (sem fold piloto)",result['folds_1_a_4'])
    lines += ["**Aviso:** este teste nao substitui McNemar oficial no teste1 rotulado, reservado para o metodo EP2 final congelado."]
    (a.out/"relatorio.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Fold(s):",all_fold, "n:",len(data))
    for e in result['por_fold']:
        print(f"fold {e['fold']}: V3={e['v3']['accuracy']:.4f}; moBERTo={e['moberto']['accuracy']:.4f}; consenso={e['consenso']['accuracy']:.4f}; saldo={e['contra_v3']['saldo']:+d}")
    print("Total:", result['total_disponivel']['contra_v3'])
    print("Resultados em",a.out)

if __name__=="__main__":
    main()
