"""Relatório descritivo, exploratório, das características discursivas por classe."""
import argparse
import csv
from pathlib import Path
import numpy as np
from discurso_features import FEATURE_NAMES,read_train_xlsx


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',type=Path,default=Path('data/raw/train.xlsx'))
    ap.add_argument('--feature-cache',type=Path,default=Path('results/ep2/discurso/features.npy'))
    ap.add_argument('--out',type=Path,default=Path('results/ep2/discurso/analise_caracteristicas.csv'))
    a=ap.parse_args()
    _,y=read_train_xlsx(a.train);X=np.load(a.feature_cache)
    if X.shape!=(len(y),len(FEATURE_NAMES)):raise ValueError('Dimensões incorretas')
    means=np.stack([X[y==k].mean(axis=0) for k in [0,1,2]])
    pcts=np.stack([(X[y==k]>0).mean(axis=0) for k in [0,1,2]])
    sd=np.maximum(X.std(axis=0),.01)
    rows=[]
    for j,name in enumerate(FEATURE_NAMES):
        effect=(max(means[:,j])-min(means[:,j]))/sd[j]
        rows.append((name,*(f'{m:.5f}' for m in means[:,j]),
                     *(f'{100*p:.3f}' for p in pcts[:,j]),f'{effect:.5f}'))
    rows.sort(key=lambda r:-float(r[-1]))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['caracteristica','media_c1','media_c234','media_c5','presenca_c1_pct','presenca_c234_pct','presenca_c5_pct','amplitude_media_div_desvio_padrao']);w.writerows(rows)
    print('Pronto:',a.out)
if __name__=='__main__':main()
