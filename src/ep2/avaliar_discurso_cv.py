"""Agrega as predições geradas por discurso_features.py em cinco folds.

Apenas avalia resultados OOF; não reentreina modelos nem seleciona regras por teste.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import binomtest

VARIANTS=['estrutural','discurso','completo','linear']

def load_csv(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def summary(pred, y, name):
    return {'modelo':name, 'accuracy':float(accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro')),
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
            'recall_by_class':{['c1','c234','c5'][i]:float(((pred==i)&(y==i)).sum()/((y==i).sum())) for i in [0,1,2]}}

def paired(a,b,y):
    corrected=int(((a!=y)&(b==y)).sum()); lost=int(((a==y)&(b!=y)).sum());
    return {'correcoes':corrected,'perdas':lost,'saldo':corrected-lost,
            'mudancas':int((a!=b).sum()),
            'mcnemar_exato_exploratorio_p':float(binomtest(min(corrected,lost),corrected+lost,.5).pvalue) if corrected+lost else 1.0}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold-dir',type=Path,default=Path('results/ep2/discurso'))
    parser.add_argument('--result-zip',type=Path,default=Path('results/old_results.zip'),help='Opcional: ZIP anterior com results/style_hgb_oof.csv')
    a=parser.parse_args()
    result=[]
    single=a.fold_dir/'oof_discursivo.csv'
    if single.exists():
        result=load_csv(single)
    else:
      for fold in range(5):
          path=a.fold_dir/f'fold{fold}'/'oof_discursivo.csv'
          r=load_csv(path)
          if len(r)!=[4019,4019,4019,4016,4019][fold]:raise ValueError(f'Fold {fold} tamanho {len(r)} inesperado')
          for row in r:
              if int(row['fold'])!=fold:raise ValueError('Fold divergente')
          result+=r
    result.sort(key=lambda row:int(row['row_id']))
    if len(result)!=20092 or [int(r['row_id']) for r in result]!=list(range(20092)):
        raise ValueError('Linhas incompletas, duplicadas ou fora de ordem')
    yy=np.array([int(r['y_true_id']) for r in result]);vv=np.array([int(r['pred_v3_id']) for r in result]);
    fd=np.array([int(r['fold']) for r in result]);pred={v:np.array([int(r[f'pred_{v}_id']) for r in result]) for v in VARIANTS}
    stats={'n':len(result),'status':'CV COMPLETA',
           'v3':summary(vv,yy,'Ensemble V3'),
           'modelos':{k:summary(p,yy,k) for k,p in pred.items()},
           'pareado':{k:paired(vv,p,yy) for k,p in pred.items()},
           'por_fold':[{ 'fold':f,'n':int((fd==f).sum()), 'v3':float(accuracy_score(yy[fd==f],vv[fd==f])),
                 **{k:float(accuracy_score(yy[fd==f],p[fd==f])) for k,p in pred.items()}} for f in range(5)]}
    style=None
    if a.result_zip.exists():
        with zipfile.ZipFile(a.result_zip) as z:
            with io.TextIOWrapper(z.open('results/style_hgb_oof.csv'),encoding='utf-8-sig',newline='') as f:
                style_rows=list(csv.DictReader(f))
        style=np.full(len(yy),-1,dtype=int)
        for r in style_rows:style[int(r['row_id'])]=int(r['y_pred_id'])
        assert (style!=-1).all()
        stats['style_original']=summary(style,yy,'style_hgb_EP1')
        # Análise complementar NÃO pré-especificada como método final. Não usar p-valor para selecionar regra.
        consensus=np.where((pred['completo']==style)&(pred['completo']!=vv), pred['completo'],vv)
        stats['consenso_exploratorio']={'modelo':summary(consensus,yy,'V3_corrigido_quando_style_e_discurso_concordam'),
              'pareado':paired(vv,consensus,yy)}
    out=a.fold_dir
    (out/'resumo_cv.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=['# EP2 — Experimento 06: validação cruzada completa','',
          '**Protocolo:** cinco folds fixos por grupo de texto; dados de desenvolvimento somente.',
          '', '| Modelo | Accuracy | Macro-F1 | Recall c1 | Recall c234 | Recall c5 |',
          '|---|---:|---:|---:|---:|---:|']
    for m in [stats['v3']]+list(stats['modelos'].values())+([stats['style_original']] if style is not None else []):
        rec=m['recall_by_class']
        rows.append(f'| {m["modelo"]} | {m["accuracy"]:.2%} | {m["macro_f1"]:.2%} | {rec["c1"]:.2%} | {rec["c234"]:.2%} | {rec["c5"]:.2%} |')
    rows.extend(['','## Por fold','','| Fold | V3 | Estrutural | Discurso | Completo | Linear |','|---|---:|---:|---:|---:|---:|'])
    for r in stats['por_fold']:
        rows.append(f'| {r["fold"]} | '+ ' | '.join(f'{r[k]:.2%}' for k in ['v3']+VARIANTS)+' |')
    rows+=['','## Complementaridade com o V3',
       f'- Modelo completo: {stats["pareado"]["completo"]["correcoes"]} erros corrigidos, {stats["pareado"]["completo"]["perdas"]} acertos perdidos quando substituído isoladamente.',
       '- A troca simples do V3 pelo classificador de atributos não é recomendada.',
       '- Regra de consenso adicional (se apresentada abaixo) é exploratória e foi avaliada com os mesmos folds; NÃO é evidência independente.',
       '- Método do EP2 continua sem estar congelado para teste1/teste2.']
    if style is not None:
        m=stats['consenso_exploratorio']['modelo'];comp=stats['consenso_exploratorio']['pareado']
        rows+=['',f'## Consenso exploratório com modelo de estilo do EP1',
               f'Accuracy={m["accuracy"]:.4%}; corrigidos={comp["correcoes"]}; perdidos={comp["perdas"]}; saldo={comp["saldo"]}.',
               'O classificador original style_hgb já era um experimento do EP1; portanto esta comparação não implica inovação inédita.']
    (out/'relatorio_cv.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('\n'.join(rows))

if __name__=='__main__':main()
