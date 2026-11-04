"""Prepara candidato usando só treino. Congelamento/inferência são etapas explícitas.

Não abre o teste1 rotulado em nenhum comando. A preparação não lê teste1/teste2.
Inferência reutiliza componentes V3 já calculados, após validar sua procedência.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from gating_discursivo import (ROOT, BASES, CONFIG, LABELS, PROBS, VOTES,
                              ConditionalMixture, dump, load_train, sha, uncertainty)

OUT = ROOT / 'results/ep2/candidato_incerteza'
MODEL = OUT / 'modelo.json'
MANIFEST = OUT / 'candidato.json'
FREEZE = OUT / 'congelamento.json'


def predict_serialized(model, x, prob):
    z = (x - np.asarray(model['mean'])) / np.asarray(model['scale'])
    weights = softmax(np.column_stack([np.ones(len(x)), z]) @ np.asarray(model['weights']), axis=1)
    mix = np.einsum('nk,nkc->nc', weights, prob)
    return softmax(np.log(np.clip(mix, 1e-12, 1)) + np.asarray(model['bias']), axis=1)


def candidate_signature():
    paths = [ROOT / 'results/ep2/gating_discursivo/protocolo_pre_especificado.json',
             ROOT / 'results/ep2/gating_discursivo/resumo.json',
             ROOT / 'results/ep2/gating_discursivo/oof_completo.csv', Path(__file__)]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def prepare(check_only=False):
    _, y, _, v3, prob, audit = load_train()
    x, names = uncertainty(prob, v3[VOTES].to_numpy())
    if check_only:
        print('Integridade OK. Nenhum teste aberto e nenhum treinamento executado.')
        return
    if FREEZE.exists():
        raise ValueError('Candidato já congelado; não reabrir seleção após avaliação oficial.')
    signature = {**audit['sources_sha256'], **candidate_signature()}
    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if m['sources_sha256'] != signature or m['model_sha256'] != sha(MODEL):
            raise ValueError('Candidato existente tem fontes diferentes; preservar e usar nova versão explicitamente.')
        print('Candidato existente validado; nada sobrescrito.')
        return
    print('Ajustando gate final em 20.092 previsões OOF; CPU; sem dados de teste...', flush=True)
    fitted = ConditionalMixture().fit(x, prob, y)
    model = {'schema': 1, 'feature_names': names, 'experts': CONFIG['experts'],
             'weights': fitted.w.tolist(), 'bias': fitted.bias.tolist(),
             'mean': fitted.scaler.mean_.tolist(), 'scale': fitted.scaler.scale_.tolist(),
             'iterations': fitted.iterations}
    restored = predict_serialized(model, x, prob)
    if not np.allclose(restored, fitted.predict_proba(x, prob), atol=1e-12):
        raise ValueError('Serialização altera predições')
    dump(MODEL, model)
    metrics = json.loads((ROOT / 'results/ep2/gating_discursivo/resumo.json').read_text(encoding='utf-8'))
    dump(MANIFEST, {'status': 'CANDIDATO_NAO_CONGELADO', 'selected_variant': 'incerteza',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config': CONFIG,
        'selection_note': 'A hipótese discursiva falhou no critério pré-especificado. Incerteza foi destacada depois, pela ablação; escolha exploratória autorizada como candidato pelo usuário. Não alegar seleção pré-especificada desta variante.',
        'evaluation': metrics['models']['incerteza'], 'reference': metrics['models']['v3'],
        'final_fit_n': len(y), 'model_sha256': sha(MODEL), 'sources_sha256': signature,
        'evaluation_limit': 'Na avaliação cada gate usa aproximadamente 2.000 linhas do seu fold. Este candidato final é reajustado em todos os OOFs, como etapa de treinamento final. Sua generalização ainda não foi avaliada em teste independente.',
        'test_labels_read': False, 'test_inputs_read': False})
    print(f'Candidato preparado e serialização verificada: {MODEL}', flush=True)


def validate_candidate():
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for relative, expected in m['sources_sha256'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Fonte mudou: {relative}')
    if sha(MODEL) != m['model_sha256']:
        raise ValueError('Modelo alterado')
    return m


def freeze(confirm):
    if not confirm:
        raise ValueError('A escolha ainda está aberta. Use --confirmar-escolha-final somente após decidir encerrar o desenvolvimento.')
    m = validate_candidate()
    if FREEZE.exists():
        f = json.loads(FREEZE.read_text(encoding='utf-8'))
        if f['candidate_sha256'] != sha(MANIFEST) or f['model_sha256'] != sha(MODEL):
            raise ValueError('Congelamento existente incompatível')
        print('Congelamento existente validado.')
        return
    dump(FREEZE, {'status': 'CONGELADO', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
         'method': 'mistura condicional com 11 sinais de incerteza e interceptos de classe',
         'candidate_sha256': sha(MANIFEST), 'model_sha256': sha(MODEL),
         'test_labels_read_before_freeze': False, 'sources_sha256': m['sources_sha256']})
    print('Método congelado. Próxima etapa: inferir. Não usar o resultado de teste1 para voltar à seleção.')


def infer():
    if not FREEZE.exists():
        raise ValueError('Método não congelado. Nenhum conjunto de teste foi aberto.')
    validate_candidate()
    f = json.loads(FREEZE.read_text(encoding='utf-8'))
    if f['candidate_sha256'] != sha(MANIFEST) or f['model_sha256'] != sha(MODEL):
        raise ValueError('Congelamento não corresponde ao candidato')
    import inferir_v3_ep2 as historical
    from entrega_ep2 import carregar_planilha, predicoes_csv, comando_submeter
    manifest = historical.load_manifest()
    for component in historical.ORDER:
        if not historical.check_cache(component, manifest):
            raise ValueError(f'Componente histórico inválido/ausente: {component}; não será retreinado automaticamente')
    first = historical.dataset(historical.ORIGINAL_TEST1, 900)
    second = historical.dataset(historical.TEST2, 897)
    combined = first + second
    v3 = pd.read_csv(historical.COMP / historical.PARTS['ensemble'][1])
    arrays = [pd.read_csv(historical.COMP / (name + '_final.csv'))[PROBS].to_numpy(float) for name in BASES]
    for col in ['pred_svc_word_char', 'pred_baseline', 'y_pred_id']:
        p = np.full((1797, 3), (1 - CONFIG['hard_vote_probability']) / 2)
        p[np.arange(1797), v3[col].to_numpy(int)] = CONFIG['hard_vote_probability']
        arrays.append(p)
    tensor = np.stack(arrays, axis=1)
    if not np.isfinite(tensor).all() or (tensor < 0).any() or not np.allclose(tensor.sum(2), 1, atol=1e-5):
        raise ValueError('Probabilidades de inferência inválidas')
    x, names = uncertainty(tensor, v3[VOTES].to_numpy())
    model = json.loads(MODEL.read_text(encoding='utf-8'))
    if model['feature_names'] != names:
        raise ValueError('Ordem de atributos divergente')
    prob = predict_serialized(model, x, tensor)
    predicted = np.array(LABELS)[prob.argmax(1)]
    destination = OUT / 'final'
    destination.mkdir(exist_ok=True)
    expected_files = [destination / f'predicoes_{tag}.csv' for tag in ['test1', 'test2']] + [destination / 'test2_rotulado_GATE.xlsx']
    if any(p.exists() for p in expected_files):
        raise ValueError('Saídas finais já existem; preservar resultados, sem sobrescrever automaticamente')
    import hashlib
    for tag, start, end in [('test1', 0, 900), ('test2', 900, 1797)]:
        d = pd.DataFrame({'row_id': np.arange(end-start), 'y_pred': predicted[start:end],
                          'text_sha256': [hashlib.sha256(t.encode('utf-8')).hexdigest() for t in combined[start:end]]})
        d.to_csv(destination / f'predicoes_{tag}.csv', index=False)
        predicoes_csv(destination / f'predicoes_{tag}.csv', combined[start:end])
    comando_submeter(argparse.Namespace(test2=historical.TEST2, predicoes=destination / 'predicoes_test2.csv', saida=destination / 'test2_rotulado_GATE.xlsx'))
    dump(destination / 'manifest.json', {'freeze_sha256': sha(FREEZE), 'input_manifest_sha256': sha(historical.MANIFEST),
        'outputs_sha256': {p.name: sha(p) for p in expected_files}, 'n_test1': 900, 'n_test2': 897,
        'test_labels_read': False, 'backup_preserved': True})
    print('Inferência concluída. McNemar é uma etapa separada, após este congelamento.')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preparar', 'congelar', 'inferir'])
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--confirmar-escolha-final', action='store_true')
    a = parser.parse_args()
    with threadpool_limits(limits=2):
        if a.command == 'preparar':
            prepare(a.check_data)
        elif a.command == 'congelar':
            freeze(a.confirmar_escolha_final)
        else:
            infer()
