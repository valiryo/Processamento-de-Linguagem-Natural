"""Análise global/local e histórico verificável do EP2; somente dados de treino."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from gating_discursivo import ROOT, LABELS, dump, load_train, metrics, sha

sys.path.insert(0, str(ROOT))
from src.models.baseline import criar_baseline_oficial

OUT = ROOT / 'results/ep2/analise_candidato'


def main():
    texts, y, folds, v3, _, audit = load_train()
    OUT.mkdir(parents=True, exist_ok=True)
    tables = []
    def add(name, truth, pred, f, path):
        m = metrics(truth, pred)
        tables.append({'modelo': name, 'escopo': '5 folds' if len(truth) == len(y) else 'fold 0',
                       'n': len(truth), 'accuracy': m['accuracy'], 'macro_f1': m['macro_f1'],
                       'recall_c234': m['recall']['c234'], 'fonte': str(path),
                       **{f'fold_{k}': float((truth[f == k] == pred[f == k]).mean()) if (f == k).any() else None for k in range(5)}})
    add('V3 EP1', y, v3.y_pred_id_v3.to_numpy(), folds, 'results/ensemble_majority_5_v3_norberto_oof.csv')
    sources = {}
    for name, folder, col in [('TAPT', 'tapt_bertimbau', 'pred_tapt_id'), ('moBERTo', 'moberto_512', 'pred_moberto_id')]:
        paths = [ROOT / f'results/ep2/{folder}/fold{k}_seed42/predicoes_fold.csv' for k in range(5)]
        parts = []
        for f, path in enumerate(paths):
            d = pd.read_csv(path)
            if 'fold' not in d:
                d['fold'] = f
            parts.append(d)
            sources[str(path.relative_to(ROOT))] = sha(path)
        d = pd.concat(parts).sort_values('row_id')
        if not np.array_equal(d.row_id, np.arange(len(y))) or not np.array_equal(d.y_true_id, y) or not np.array_equal(d.fold, folds):
            raise ValueError(f'Histórico desalinhado: {name}')
        if col not in d:
            raise ValueError(f'{name}: colunas disponíveis {list(d.columns)}')
        add(name, y, d[col].to_numpy(int), folds, str(paths[0].parent.parent.relative_to(ROOT)))
    gate = pd.read_csv(ROOT / 'results/ep2/gating_discursivo/oof_completo.csv')
    for name in ['constante', 'incerteza', 'discurso']:
        add('Gate ' + name, y, gate['pred_' + name].to_numpy(), folds, 'results/ep2/gating_discursivo/oof_completo.csv')
    pd.DataFrame(tables).to_csv(OUT / 'experimentos_reproduzidos.csv', index=False)

    print('Ajustando baseline interpretável em todo o treino (análise descritiva)...', flush=True)
    model = criar_baseline_oficial().fit(texts, np.array(LABELS)[y])
    vec = model.named_steps['tfidf']
    clf = model.named_steps['classificador']
    terms = vec.get_feature_names_out()
    rows = []
    for c, label in enumerate(clf.classes_):
        for direction, ids in [('positivo', np.argsort(clf.coef_[c])[-20:][::-1]), ('negativo', np.argsort(clf.coef_[c])[:20])]:
            for rank, j in enumerate(ids, start=1):
                rows.append({'classe': label, 'direcao': direction, 'rank': rank, 'termo': terms[j], 'coeficiente': clf.coef_[c, j]})
    pd.DataFrame(rows).to_csv(OUT / 'importancia_global_baseline.csv', index=False)
    # Interpretação local usa modelo treinado FORA do fold 0.
    tr, va = folds != 0, folds == 0
    local_model = criar_baseline_oficial().fit(texts[tr], np.array(LABELS)[y[tr]])
    v = local_model.named_steps['tfidf']
    c = local_model.named_steps['classificador']
    base_pred = local_model.predict(texts[va])
    original_base = pd.read_csv(ROOT / 'results/baseline_oof.csv').sort_values('row_index')
    historic = original_base.predicao.to_numpy()[va]
    agreements = int((base_pred == historic).sum())
    print(f'Baseline fold 0: {agreements}/{int(va.sum())} predições iguais ao histórico', flush=True)
    ids = np.flatnonzero(va)
    local = []
    # Um caso curto por par de confusão do V3: exemplos para posterior leitura qualitativa.
    for true in range(3):
        for pred in range(3):
            if true == pred:
                continue
            eligible = [i for i in ids if y[i] == true and v3.y_pred_id_v3.iloc[i] == pred and 100 <= len(texts[i]) <= 1200]
            if not eligible:
                continue
            i = eligible[0]
            row = v.transform([texts[i]])
            scores = local_model.decision_function([texts[i]])[0]
            order = np.argsort(scores)[::-1]
            winner, runner = order[:2]
            effects = row.data * (c.coef_[winner, row.indices] - c.coef_[runner, row.indices])
            offset = c.intercept_[winner] - c.intercept_[runner]
            reconstructed = float(effects.sum() + offset)
            if not np.isclose(reconstructed, scores[winner] - scores[runner], atol=1e-10):
                raise ValueError('Contribuições locais não recompõem a margem')
            term_names = v.get_feature_names_out()
            relevant = np.argsort(np.abs(effects))[::-1][:12]
            local.append({'row_id': int(i), 'fold': 0, 'real': LABELS[true], 'v3': LABELS[pred],
                          'gate_incerteza': LABELS[int(gate.pred_incerteza.iloc[i])],
                          'baseline': str(c.classes_[winner]), 'segunda_baseline': str(c.classes_[runner]),
                          'texto': texts[i], 'margem_logit': reconstructed, 'intercepto_diferenca': float(offset),
                          'contribuicoes': [{'termo': str(term_names[row.indices[j]]), 'delta_logit': float(effects[j])} for j in relevant]})
    dump(OUT / 'interpretacao_local.json', local)
    # Diagnóstico por família de atributo e classe, descritivo, sem treinar/selecionar.
    disc = np.load(ROOT / 'results/ep2/gating_discursivo/features.npy')
    from discurso_features import FEATURE_NAMES
    stats = []
    for j, feature in enumerate(FEATURE_NAMES):
        for target, mask in [('todos', np.ones(len(y), dtype=bool))] + [(label, y == k) for k, label in enumerate(LABELS)]:
            stats.append({'atributo': feature, 'classe': target, 'media': float(disc[mask, j].mean()),
                          'mediana': float(np.median(disc[mask, j])), 'desvio': float(disc[mask, j].std())})
    pd.DataFrame(stats).to_csv(OUT / 'atributos_por_classe.csv', index=False)
    dump(OUT / 'manifest.json', {'train_sha256': sha(ROOT / 'data/raw/train.xlsx'),
        'script_sha256': sha(__file__), 'fontes_historicas': sources,
        'baseline_fold0_predicoes_iguais': agreements, 'baseline_fold0_n': int(va.sum()),
        'baseline_fold0': metrics(y[va], np.array([LABELS.index(s) for s in base_pred])),
        'test_labels_read': False,
        'limite': 'Coeficientes e contribuições explicam o baseline auxiliar, não os Transformers nem o gate. Associação não é causalidade.',
        'nao_localizados': ['Resultados brutos do experimento 06 (discurso) citado no guia', 'Código e resultados brutos do experimento 07 (limpeza) citado no guia']})
    print(pd.DataFrame(tables).to_string(index=False), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    with threadpool_limits(limits=2):
        main()
