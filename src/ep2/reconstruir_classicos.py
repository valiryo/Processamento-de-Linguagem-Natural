"""Reconstrói OOFs clássicos com configurações já escolhidas, somente em cópia limpa."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('modelo', choices=['baseline', 'svc'])
    a = p.parse_args()
    from reproduzir_ep2 import check_files
    check_files(data_only=True)
    if (ROOT / 'results/ep2/candidato_incerteza/congelamento.json').exists():
        raise ValueError('Reconstrução somente em cópia limpa, sem candidato congelado.')
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    output = ROOT / 'results' / ('baseline_oof.csv' if a.modelo == 'baseline' else 'tfidf_word_char_linearsvc_oof.csv')
    if output.exists():
        raise ValueError(f'Preservar OOF existente: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        if a.modelo == 'svc':
            from src.experiments import tfidf_word_char_linearsvc as svc
            svc.C_VALUES = [0.5]  # valor histórico final; nenhuma nova busca
            svc.main()
        else:
            from src.models.baseline import criar_baseline_oficial
            train = pd.read_excel(ROOT / 'data/raw/train.xlsx')
            folds = pd.read_csv(ROOT / 'data/splits/folds.csv')
            texts = train.resp_text.astype(str)
            predicted = np.empty(len(train), dtype=object)
            for f in range(5):
                val = folds.fold.to_numpy() == f
                if set(texts[val]) & set(texts[~val]):
                    raise ValueError('Grupos cruzam folds')
                model = criar_baseline_oficial().fit(texts[~val], train.clarity[~val])
                predicted[val] = model.predict(texts[val])
                print(f'Baseline fold {f}: concluído', flush=True)
            pd.DataFrame({'row_index': np.arange(len(train)), 'classe_real': train.clarity,
                          'predicao': predicted, 'fold': folds.fold}).to_csv(output, index=False)


if __name__ == '__main__':
    main()
