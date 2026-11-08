"""Reprodução auditável do gate congelado, sem seleção nem treinamento neural.

Os módulos científicos originais são importados sem alterar seus arquivos.
As novas saídas ficam em results/ep2/reproducao; a entrega original é protegida.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'src/ep2/artefatos_reproducao.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metadata():
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def check_files(data_only=False):
    m = metadata()
    entries = m['dados_externos'] if data_only else {**m['arquivos'], **m['dados_externos']}
    if data_only:
        entries = {**entries, 'data/splits/folds.csv': m['arquivos']['data/splits/folds.csv']}
    errors = []
    for name, digest in entries.items():
        p = ROOT / name
        if not p.is_file():
            errors.append(f'Ausente: {name}')
        elif sha(p) != digest:
            errors.append(f'SHA-256 divergente: {name}')
    if errors:
        raise ValueError('\n'.join(errors) + '\nExtraia os dois pacotes na mesma raiz e forneça os dados oficiais sem regravar as planilhas.')
    print(f'Integridade: {len(entries)} arquivos conferidos.', flush=True)
    return m


def check():
    m = check_files()
    import candidato_incerteza as c
    import inferir_v3_ep2 as h
    from gating_discursivo import load_train
    c.validate_candidate()
    f = json.loads(c.FREEZE.read_text(encoding='utf-8'))
    if f['status'] != 'CONGELADO' or f['candidate_sha256'] != sha(c.MANIFEST) or f['model_sha256'] != sha(c.MODEL):
        raise ValueError('Congelamento incompatível')
    load_train()  # folds, grupos, classes, OOFs, probabilidades e regra V3
    hm = h.load_manifest()
    for name in h.ORDER:
        if not h.check_cache(name, hm):
            raise ValueError(f'Cache histórico ausente ou inválido: {name}')
    h.dataset(h.ORIGINAL_TEST1, 900)
    h.dataset(h.TEST2, 897)
    print('Treino, folds, modelo congelado e sete caches íntegros. Gabarito de teste1 não lido.', flush=True)
    return m


def destination(value):
    p = (ROOT / value).resolve()
    p.relative_to((ROOT / 'results/ep2').resolve())
    protected = ['candidato_incerteza', 'inferencia_v3', 'final', 'gating_discursivo', 'entrega_grupo13', 'entrega_grupo13_revisada']
    for name in protected:
        original = (ROOT / 'results/ep2' / name).resolve()
        if p == original or original in p.parents or p in original.parents:
            raise ValueError(f'Destino protegido: {p}')
    return p


def validate_output(folder):
    from entrega_ep2 import carregar_planilha, predicoes_csv
    m = metadata()
    generated = folder / 'final'
    for name, digest in m['saidas_congeladas_sha256'].items():
        if sha(generated / name) != digest:
            raise ValueError(f'Saída diferente da referência congelada: {name}')
    raw, _, _ = carregar_planilha(ROOT / 'data/raw/test2.xlsx', 897, rotulado=False)
    result, _, _ = carregar_planilha(generated / 'test2_rotulado_GATE.xlsx', 897, rotulado=True)
    if [x[0] for x in raw] != [x[0] for x in result]:
        raise ValueError('Textos/ordem divergentes')
    predicted = predicoes_csv(generated / 'predicoes_test2.csv', [x[0] for x in raw])
    if predicted != [x[1] for x in result]:
        raise ValueError('XLSX difere do CSV')
    from collections import Counter
    counts = dict(Counter(predicted))
    if counts != m['test2_distribuicao']:
        raise ValueError('Distribuição diferente da referência')
    print(f'APROVADO: 897 linhas, 2 colunas, ordem preservada; {counts}. CSVs e XLSX idênticos em SHA-256 à entrega.')


def infer(folder):
    check()
    import candidato_incerteza as c
    if (folder / 'final').exists():
        validate_output(folder)
        print('Saídas existentes conferidas; nenhuma sobrescrita.')
        return
    folder.mkdir(parents=True, exist_ok=True)
    # Só o destino muda. MODEL/MANIFEST/FREEZE continuam apontando aos originais.
    c.OUT = folder
    c.infer()
    validate_output(folder)


def evaluate_oof(folder):
    check_files()
    import numpy as np
    import pandas as pd
    import gating_discursivo as g
    out = folder / 'oof'
    if out.exists():
        raise ValueError('Use --saida com uma pasta nova para recalcular todos os folds, sem retomada de resultados anteriores.')
    g.run(argparse.Namespace(out=out, check_data=False, fold=None))
    original = pd.read_csv(ROOT / 'results/ep2/gating_discursivo/oof_completo.csv')
    reproduced = pd.read_csv(out / 'oof_completo.csv')
    cols = ['row_id', 'fold', 'half', 'y_true_id', 'pred_v3', 'pred_constante', 'pred_incerteza', 'pred_discurso']
    if not np.array_equal(original[cols].to_numpy(), reproduced[cols].to_numpy()):
        raise ValueError('As predições OOF reproduzidas diferem do histórico; não substituir a entrega.')
    ref = json.loads((ROOT / 'results/ep2/gating_discursivo/resumo.json').read_text(encoding='utf-8'))
    actual = json.loads((out / 'resumo.json').read_text(encoding='utf-8'))
    if actual['models'] != ref['models']:
        raise ValueError('Métricas diferentes do histórico')
    print('APROVADO: três variantes, cinco folds e 20.092 predições OOF reproduzidos.')


def mcnemar(folder, ep1):
    check()
    validate_output(folder)
    from entrega_ep2 import comando_mcnemar
    m = metadata()
    truth = ROOT / 'data/raw/test1-rotulado-para-teste-estatistico.xlsx'
    if sha(truth) != m['mcnemar_externo']['gabarito_sha256']:
        raise ValueError(f'Gabarito externo de McNemar divergente: {truth}')
    # GitHub entrega LF; a cópia histórica Windows usa CRLF. Dois hashes exatos,
    # previamente verificados como os mesmos dados, sem aceitar alterações livres.
    accepted = {m['mcnemar_externo']['ep1_sha256'], m['mcnemar_externo']['ep1_sha256_lf']}
    if sha(ep1) not in accepted:
        raise ValueError(f'Referência externa do EP1 divergente: {ep1}')
    output = folder / 'mcnemar_reproduzido.json'
    comando_mcnemar(argparse.Namespace(metodo_congelado=True, test1=truth,
        original_test1=ROOT / 'data/raw/test1.xlsx', predicoes_ep1=ep1,
        predicoes_ep2=folder / 'final/predicoes_test1.csv', saida=output))
    reference = json.loads((ROOT / 'results/ep2/candidato_incerteza/final/mcnemar_oficial.json').read_text(encoding='utf-8'))
    if json.loads(output.read_text(encoding='utf-8')) != reference:
        raise ValueError('McNemar diferente do histórico')
    print('APROVADO: McNemar reproduzido; nenhuma seleção foi reaberta.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('comando', choices=['conferir-dados', 'conferir', 'inferir', 'validar-saida', 'oof', 'mcnemar'])
    p.add_argument('--saida', type=Path, default=Path('results/ep2/reproducao'))
    p.add_argument('--ep1', type=Path, default=ROOT / 'results/ensemble_majority_5_v3_norberto_final.csv')
    a = p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        folder = destination(a.saida)
        if a.comando == 'conferir-dados': check_files(data_only=True)
        elif a.comando == 'conferir': check()
        elif a.comando == 'inferir': infer(folder)
        elif a.comando == 'validar-saida': validate_output(folder)
        elif a.comando == 'oof': evaluate_oof(folder)
        else: mcnemar(folder, a.ep1.resolve())


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError, ImportError) as exc:
        print(f'ERRO: {exc}', file=sys.stderr)
        raise SystemExit(1)
