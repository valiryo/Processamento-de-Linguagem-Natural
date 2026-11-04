#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agrega resultados OOF TAPT (1..5 folds), com checagem de alinhamento.

Nao carrega treino/teste1/teste2, nao escolhe thresholds nem hiperparametros.
Uso (raiz do repo): python src/ep2/avaliar_tapt_cv.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support

CLASSES = ['c1', 'c234', 'c5']
PRED_COLS = ['row_id', 'fold', 'y_true_id', 'pred_v3_id', 'pred_tapt_id', 'prob_c1', 'prob_c234', 'prob_c5']


def score(y, z):
    return {'accuracy': float(accuracy_score(y, z)), 'macro_f1': float(f1_score(y, z, labels=[0,1,2], average='macro'))}


def paired(y, old, new):
    old_ok, new_ok = old == y, new == y
    recovered = int((~old_ok & new_ok).sum())
    lost = int((old_ok & ~new_ok).sum())
    agree_both_correct = int((old_ok & new_ok).sum())
    agree_both_wrong = int((~old_ok & ~new_ok).sum())
    n = len(y)
    return {'ambos_acertam': agree_both_correct, 'somente_v3_acerta': lost,
            'somente_tapt_acerta': recovered, 'ambos_erram': agree_both_wrong,
            'recuperados': recovered, 'perdidos': lost,
            'saldo_acertos': recovered - lost, 'delta_accuracy_pp': 100 * (recovered - lost) / n,
            'mcnemar_exato_p_exploratorio': float(binomtest(recovered, recovered + lost).pvalue) if recovered + lost else 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folds-dir', type=Path, default=Path('results/ep2/tapt_bertimbau'))
    parser.add_argument('--v3', type=Path, default=Path('results/ensemble_majority_5_v3_norberto_oof.csv'))
    parser.add_argument('--out', type=Path, default=Path('results/ep2/tapt_bertimbau/analise_cv'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.v3.is_file():
        parser.error(f'OOF V3 nao encontrado: {args.v3}')
    v3 = pd.read_csv(args.v3, encoding='utf-8-sig')
    required_v3 = {'row_id', 'fold', 'y_true_id', 'y_pred_id_v3'}
    if not required_v3.issubset(v3.columns):
        raise ValueError(f'V3 sem colunas: {required_v3 - set(v3.columns)}')
    if len(v3) != 20092 or v3.row_id.nunique() != 20092:
        raise ValueError('V3 nao possui 20.092 row_ids distintos')
    v3 = v3.set_index('row_id', verify_integrity=True).sort_index()
    if not np.array_equal(v3.index.to_numpy(), np.arange(20092)):
        raise ValueError('V3 nao tem row_ids de 0 a 20091')
    if not np.isin(v3[['fold','y_true_id','y_pred_id_v3']], [0,1,2,3,4]).all().all():
        raise ValueError('Dados invalidos na referencia V3')
    if not np.isclose(accuracy_score(v3.y_true_id, v3.y_pred_id_v3), 0.475661955, atol=1e-7):
        raise ValueError('A referencia V3 nao confere com a OOF oficial')

    chunks = []
    found = []
    for fold in range(5):
        path = args.folds_dir / f'fold{fold}_seed{args.seed}' / 'predicoes_fold.csv'
        metrics = path.parent / 'metricas.json'
        if not path.is_file():
            if metrics.exists():
                raise ValueError(f'Fold {fold}: metricas presentes, mas CSV ausente em {path}')
            continue
        if not metrics.is_file():
            raise ValueError(f'Fold {fold}: CSV presente sem metricas.json: {path}')
        df = pd.read_csv(path, encoding='utf-8-sig')
        m = json.loads(metrics.read_text(encoding='utf-8-sig'))
        if not set(PRED_COLS).issubset(df.columns):
            raise ValueError(f'Fold {fold}: colunas faltando: {set(PRED_COLS)-set(df.columns)}')
        if df.row_id.duplicated().any() or not (df.fold == fold).all():
            raise ValueError(f'Fold {fold}: ids duplicados ou fold incorreto')
        if (m.get('fold') != fold or m.get('seed') != args.seed or m.get('n_val') != len(df)):
            raise ValueError(f'Fold {fold}: metricas nao conferem com o CSV')
        original_ids = v3.index[v3.fold == fold]
        if set(df.row_id) != set(original_ids):
            raise ValueError(f'Fold {fold}: falta/sobra linha comparado aos folds oficiais')
        compare = v3.loc[df.row_id.to_numpy()]
        for c_source, col in [('y_true_id','y_true_id'),('pred_v3_id','y_pred_id_v3'),('fold','fold')]:
            if not np.array_equal(df[c_source].to_numpy(), compare[col].to_numpy()):
                raise ValueError(f'Fold {fold}: divergencia entre {c_source} e V3')
        for c in ['y_true_id','pred_v3_id','pred_tapt_id']:
            if not df[c].isin([0,1,2]).all():
                raise ValueError(f'Fold {fold}: predicao ou rotulo invalido em {c}')
        ps = df[['prob_c1','prob_c234','prob_c5']].to_numpy()
        if not np.isfinite(ps).all() or not np.allclose(ps.sum(axis=1),1,atol=1e-3) or (ps < 0).any():
            raise ValueError(f'Fold {fold}: probabilidades invalidas')
        if not np.array_equal(df.pred_tapt_id.to_numpy(), ps.argmax(axis=1)):
            raise ValueError(f'Fold {fold}: predicao nao corresponde ao argmax de probabilidades')
        acc = accuracy_score(df.y_true_id, df.pred_tapt_id)
        if not np.isclose(acc, m['tapt_accuracy'],atol=1e-8):
            raise ValueError(f'Fold {fold}: acuracia nao confere com metricas.json')
        chunks.append(df)
        found.append(fold)
    if not chunks:
        parser.error('Nenhum fold encontrado. Rode o experimento 03 ou corrija --folds-dir')
    full = pd.concat(chunks, ignore_index=True).sort_values('row_id').reset_index(drop=True)
    if full.row_id.duplicated().any():
        raise ValueError('IDs duplicados entre folds')
    complete = found == list(range(5)) and len(full) == len(v3)
    if complete and not np.array_equal(full.row_id.to_numpy(),v3.index.to_numpy()):
        raise ValueError('OOF agregada nao cobre exatamente o treino inteiro')
    y, old, new = (full[c].to_numpy(dtype=int) for c in ['y_true_id','pred_v3_id','pred_tapt_id'])
    per_fold = []
    for fold, df in full.groupby('fold',sort=True):
        yy, oo, nn = (df[c].to_numpy(dtype=int) for c in ['y_true_id','pred_v3_id','pred_tapt_id'])
        r = paired(yy,oo,nn)
        per_fold.append({'fold':int(fold),'n':len(df),'v3_accuracy':score(yy,oo)['accuracy'],
                         'tapt_accuracy':score(yy,nn)['accuracy'], 'delta_accuracy_pp':r['delta_accuracy_pp'],
                         'ganhos':r['recuperados'],'perdas':r['perdidos']})
    per_class = []
    for cid, name in enumerate(CLASSES):
        mask = y == cid
        b, a = int((old[mask]==cid).sum()),int((new[mask]==cid).sum())
        r = paired(y[mask],old[mask],new[mask])
        per_class.append({'classe':name,'n':int(mask.sum()),'acertos_v3':b,'acertos_tapt':a,
                          'recall_v3':float(b/mask.sum()),'recall_tapt':float(a/mask.sum()),
                          'delta_recall_pp':float(100*(a-b)/mask.sum()),
                          'recuperados':r['recuperados'],'perdidos':r['perdidos']})
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(per_fold).to_csv(out/'por_fold.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(per_class).to_csv(out/'por_classe.csv',index=False,encoding='utf-8-sig')
    full.to_csv(out/('tapt_oof_completo.csv' if complete else 'tapt_oof_parcial.csv'),index=False,encoding='utf-8-sig')
    r = paired(y,old,new)
    results = {'status':'CV COMPLETA' if complete else 'PARCIAL',
               'folds_disponiveis':found,'n_total':len(full),
               'v3':score(y,old),'tapt':score(y,new),
               'comparacao_pareada':r,
               'matriz_confusao_tapt': confusion_matrix(y,new,labels=[0,1,2]).tolist(),
               'por_fold':per_fold,'por_classe':per_class,
               'aviso':'McNemar OOF exploratorio; o teste oficial do EP2 usa o teste1 rotulado, somente apos congelar o metodo.'}
    (out/'resumo.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    msg = [f'# TAPT — {results["status"]}',
           f'Folds disponiveis: {found} (n={len(full)}).',
           f'Acuracia V3: {results["v3"]["accuracy"]:.6f}.',
           f'Acuracia TAPT: {results["tapt"]["accuracy"]:.6f}.',
           f'Delta TAPT - V3: {r["delta_accuracy_pp"]:+.4f} pontos percentuais.',
           f'Acertos recuperados: {r["recuperados"]}; perdidos: {r["perdidos"]}.',
           f'McNemar exato (exploratorio): p={r["mcnemar_exato_p_exploratorio"]:.6f}.',
           '', '## Acuracia por fold',
           '| fold | n | V3 | TAPT | delta (p.p.) |', '|---|---:|---:|---:|---:|']
    for q in per_fold:
        msg.append(f'| {q["fold"]} | {q["n"]} | {q["v3_accuracy"]:.4%} | {q["tapt_accuracy"]:.4%} | {q["delta_accuracy_pp"]:+.3f} |')
    msg += ['', '## Recall por classe',
            '| classe | n | V3 | TAPT | delta (p.p.) |', '|---|---:|---:|---:|---:|']
    for q in per_class:
        msg.append(f'| {q["classe"]} | {q["n"]} | {q["recall_v3"]:.4%} | {q["recall_tapt"]:.4%} | {q["delta_recall_pp"]:+.3f} |')
    msg += ['', '**Nota:** esta analise nao usa teste1 nem teste2. O McNemar oficial sera no teste1 rotulado, com o metodo EP2 congelado.']
    (out/'relatorio.md').write_text('\n'.join(msg)+'\n',encoding='utf-8')
    print('\n'.join(msg)); print('\nArquivos salvos:',out)

if __name__ == '__main__':
    main()
