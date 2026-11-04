#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduz inferência do Ensemble V3 do EP1 em teste1 + teste2.

Coloque SOMENTE este arquivo em src/ep2/ do repositório original.
Os scripts históricos em src/experiments/ são carregados sem alterações.

Uso (PowerShell, na raiz):
  python src/ep2/inferir_v3_ep2.py conferir
  python src/ep2/inferir_v3_ep2.py preparar
  python src/ep2/inferir_v3_ep2.py rodar
  python src/ep2/inferir_v3_ep2.py separar
  python src/ep2/entrega_ep2.py submeter

Os dois conjuntos são previstos juntos numa única execução por modelo;
nenhum rótulo do teste1 rotulado é lido por este programa.

AVISO ACADÊMICO: reproduzir o V3 do EP1 NÃO constitui método inovador
para o EP2 e não assegura melhora estatística sobre a submissão anterior.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src' / 'experiments'
RAW = ROOT / 'data' / 'raw'
WORK = ROOT / 'results' / 'ep2' / 'inferencia_v3'
COMP = WORK / 'componentes'
FINAL = ROOT / 'results' / 'ep2' / 'final'
INPUT = WORK / 'entrada_combinada.xlsx'
MANIFEST = WORK / 'manifest.json'
TRAIN = RAW / 'train.xlsx'
ORIGINAL_TEST1 = RAW / 'test1.xlsx'  # teste1 SEM rótulos
TEST2 = RAW / 'test2.xlsx'
N1, N2 = 900, 897
N = N1 + N2
LABELS = ('c1', 'c234', 'c5')

# Nome lógico -> nome exato do script histórico -> CSV original de saída.
PARTS = {
    'baseline': ('baseline_final.py', 'baseline_final.csv'),
    'svc': ('tfidf_word_char_linearsvc_final.py', 'tfidf_word_char_linearsvc_final.csv'),
    'bert512': ('bertimbau_512_final.py', 'bertimbau_512_final.csv'),
    'soft256': ('bertimbau_duplicate_soft_labels_final.py', 'bertimbau_duplicate_soft_labels_256_final.csv'),
    'ordinal256': ('bertimbau_ordinal_256_final.py', 'bertimbau_ordinal_256_final.csv'),
    'norberto': ('norberto_base_1024_final.py', 'norberto_base_1024_final.csv'),
    'ensemble': ('ensemble_v3_final.py', 'ensemble_majority_5_v3_norberto_final.csv'),
}
ORDER = tuple(PARTS)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f'Arquivo obrigatório ausente: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def require_files() -> None:
    for path in (TRAIN, ORIGINAL_TEST1, TEST2):
        if not path.is_file():
            raise FileNotFoundError(f'Não encontrei {path}. Verifique data/raw/')
    for name, (script, _) in PARTS.items():
        if not (SRC / script).is_file():
            raise FileNotFoundError(f'Falta script {name}: {SRC / script}')
    if not (ROOT / 'src' / 'models' / 'baseline.py').is_file():
        raise FileNotFoundError('Falta src/models/baseline.py, usado pelo baseline original.')


def dataset(path: Path, n_expected: int):
    import pandas as pd
    df = pd.read_excel(path).reset_index(drop=True)
    if len(df) != n_expected:
        raise ValueError(f'{path}: {len(df)} linhas, esperadas {n_expected}')
    if 'resp_text' not in df.columns:
        raise ValueError(f'{path}: não tem resp_text')
    if df['resp_text'].isna().any() or not df['resp_text'].astype(str).str.strip().all():
        raise ValueError(f'{path}: há textos vazios')
    if 'clarity' in df and df['clarity'].notna().any():
        raise ValueError(f'{path}: clarity está preenchido! Use apenas arquivos NÃO rotulados na inferência.')
    return df['resp_text'].astype(str).tolist()


def source_signature() -> dict:
    require_files()
    return {
        'train_sha256': sha_file(TRAIN),
        'test1_sha256': sha_file(ORIGINAL_TEST1),
        'test2_sha256': sha_file(TEST2),
        'scripts_sha256': {name: sha_file(SRC / f) for name, (f, _) in PARTS.items()},
    }


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        raise FileNotFoundError('Execute primeiro: python src/ep2/inferir_v3_ep2.py preparar')
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if m['signature'] != source_signature():
        raise ValueError('Os arquivos de entrada ou scripts mudaram desde preparar; não misture predições. '
                         'Faça backup dos resultados e execute preparar --reiniciar.')
    if not INPUT.is_file():
        raise FileNotFoundError(f'Arquivo intermediário ausente: {INPUT}')
    if sha_file(INPUT) != m['input_sha256']:
        raise ValueError('A planilha intermediária foi modificada!')
    return m


def prepare(reset: bool = False) -> None:
    require_files()
    if MANIFEST.exists() and not reset:
        m = load_manifest()
        print(f'Já preparado: {INPUT} ({m["n"]} registros).')
        return
    if MANIFEST.exists() and reset:
        print('AVISO: reiniciando. Os CSVs antigos serão apagados antes de novas execuções.')
        for _, (_, filename) in PARTS.items():
            (COMP / filename).unlink(missing_ok=True)
        for name in PARTS:
            (COMP / (name + '.meta.json')).unlink(missing_ok=True)
        (COMP / 'ensemble_majority_5_v2_final.csv').unlink(missing_ok=True)
        for name in ('predicoes_test1_ep2.csv', 'predicoes_test2_ep2.csv'):
            (FINAL / name).unlink(missing_ok=True)
    import pandas as pd
    first = dataset(ORIGINAL_TEST1, N1)
    second = dataset(TEST2, N2)
    WORK.mkdir(parents=True, exist_ok=True)
    COMP.mkdir(parents=True, exist_ok=True)
    # Escreve somente entradas; NÃO transfere os rótulos do teste1 rotulado.
    pd.DataFrame({'resp_text': first + second, 'clarity': [None] * N}).to_excel(INPUT, index=False)
    m = {
        'signature': source_signature(),
        'input_sha256': sha_file(INPUT),
        'n': N, 'n_test1': N1, 'n_test2': N2,
        'hashes_test1': [sha_text(t) for t in first],
        'hashes_test2': [sha_text(t) for t in second],
    }
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Preparado: {N1} textos de teste1 + {N2} textos de teste2 = {N} textos.')
    print(f'Entrada conjunta: {INPUT}')
    print('Nenhum rótulo verdadeiro dos conjuntos de teste foi lido.')


def validate_csv(path: Path, n: int = N, *, require_probs: bool = False) -> None:
    import pandas as pd
    if not path.is_file():
        raise FileNotFoundError(f'Saída ausente: {path}')
    df = pd.read_csv(path)
    columns = {'row_id', 'y_pred_id'}
    if require_probs:
        columns |= {'prob_c1', 'prob_c234', 'prob_c5'}
    if not columns.issubset(df.columns):
        raise ValueError(f'{path.name}: faltam colunas {sorted(columns - set(df.columns))}')
    if len(df) != n or df['row_id'].tolist() != list(range(n)):
        raise ValueError(f'{path.name}: linhas/ordem de row_id inválidas')
    if not set(df['y_pred_id'].tolist()).issubset({0, 1, 2}):
        raise ValueError(f'{path.name}: y_pred_id inválido')
    if require_probs and (df[['prob_c1', 'prob_c234', 'prob_c5']].isna().any().any()):
        raise ValueError(f'{path.name}: probabilidades incompletas')


def expected_meta(name: str, m: dict, csvpath: Path) -> dict:
    meta = {'input_sha256': m['input_sha256'],
            'script_sha256': m['signature']['scripts_sha256'][name],
            'csv_sha256': sha_file(csvpath)}
    if name == 'ensemble':
        meta['dependencies_sha256'] = {dep: sha_file(COMP / PARTS[dep][1])
                                       for dep in ORDER[:-1]}
    return meta


def check_cache(name: str, m: dict) -> bool:
    script, csvname = PARTS[name]
    csvpath = COMP / csvname
    meta = COMP / (name + '.meta.json')
    if not csvpath.is_file() or not meta.is_file():
        return False
    try:
        info = json.loads(meta.read_text(encoding='utf-8'))
        if info != expected_meta(name, m, csvpath):
            return False
        validate_csv(csvpath, require_probs=(name == 'norberto'))
        return True
    except (ValueError, KeyError, OSError):
        return False


def run_child(name: str) -> None:
    """Um único componente em processo isolado para liberar a GPU entre modelos."""
    if name not in PARTS:
        raise ValueError(f'Componente desconhecido: {name}')
    m = load_manifest()
    filename, outputname = PARTS[name]
    scriptpath = SRC / filename
    spec = importlib.util.spec_from_file_location(f'ep2_original_{name}', scriptpath)
    if spec is None or spec.loader is None:
        raise ImportError(f'Não foi possível carregar {scriptpath}')
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(mod)
    # APENAS configuração de caminhos/quantidade; parâmetros de treinamento e votação intactos.
    mod.RESULTS_DIR = COMP
    if name != 'ensemble':
        mod.TEST_PATH = INPUT
        mod.EXPECTED_TEST_ROWS = N
        mod.OUTPUT_PATH = COMP / outputname
        for key in ('TRAIN_OUTPUT_DIR', 'OUTPUT_DIR'):
            if hasattr(mod, key):
                setattr(mod, key, ROOT / 'output' / 'ep2' / 'inferencia_v3' / name)
    else:
        mod.EXPECTED_N = N
        paths = {
            'BERT512_PATH': 'bert512', 'SOFT256_PATH': 'soft256',
            'SVC_PATH': 'svc', 'BASELINE_PATH': 'baseline',
            'ORDINAL_PATH': 'ordinal256', 'NORBERTO_PATH': 'norberto',
        }
        for field, dep in paths.items():
            setattr(mod, field, COMP / PARTS[dep][1])
        mod.V2_OUTPUT_PATH = COMP / 'ensemble_majority_5_v2_final.csv'
        mod.V3_OUTPUT_PATH = COMP / outputname
    print(f'Executando script original: {filename} (n={N})', flush=True)
    mod.main()
    csvpath = COMP / outputname
    validate_csv(csvpath, require_probs=(name == 'norberto'))
    info = expected_meta(name, m, csvpath)
    (COMP / (name + '.meta.json')).write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'OK: {csvpath}', flush=True)


def run_all(one: str | None, force: bool = False) -> None:
    m = load_manifest()
    names = [one] if one else list(ORDER)
    if one == 'ensemble':
        missing = [x for x in ORDER[:-1] if not check_cache(x, m)]
        if missing:
            raise ValueError('Antes do ensemble, conclua os componentes: ' + ', '.join(missing))
    for name in names:
        if not force and check_cache(name, m):
            print(f'[PULANDO] {name}: CSV validado para a entrada atual.', flush=True)
            continue
        if name == 'ensemble':
            missing = [x for x in ORDER[:-1] if not check_cache(x, m)]
            if missing:
                raise ValueError('Faltam componentes válidos: ' + ', '.join(missing))
        print(f'\n=== {name} ===', flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '_executar', name]
        subprocess.run(command, cwd=ROOT, check=True)
    print('\nConcluído. Agora execute: python src/ep2/inferir_v3_ep2.py separar')


def split() -> None:
    m = load_manifest()
    if not check_cache('ensemble', m):
        raise ValueError('Ensemble não está pronto/válido. Execute "rodar" primeiro.')
    import pandas as pd
    df = pd.read_csv(COMP / PARTS['ensemble'][1])
    if len(df) != N or 'y_pred' not in df:
        raise ValueError('O arquivo final de ensemble está incompleto.')
    true_ids = df['y_pred_id'].astype(int).tolist()
    preds = df['y_pred'].astype(str).tolist()
    if any(p != LABELS[i] for p, i in zip(preds, true_ids)):
        raise ValueError('Inconsistência y_pred e y_pred_id no arquivo do ensemble.')
    FINAL.mkdir(parents=True, exist_ok=True)
    for tag, offset, hashes in (
        ('test1', 0, m['hashes_test1']),
        ('test2', N1, m['hashes_test2']),
    ):
        path = FINAL / f'predicoes_{tag}_ep2.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['row_id', 'y_pred', 'text_sha256'])
            for i, text_hash in enumerate(hashes):
                writer.writerow([i, preds[offset + i], text_hash])
        print(f'Gerado: {path} ({len(hashes)} predições reais)', flush=True)
    print('\nAgora: python src/ep2/entrega_ep2.py submeter')
    print('AVISO: estas predições reproduzem o V3 do EP1; não demonstram inovação no EP2.')


def status() -> None:
    m = load_manifest()
    print(f'Entrada preparada: {N1} + {N2} = {N} instâncias')
    for name in ORDER:
        print(f'  {name:12s}: ' + ('PRONTO' if check_cache(name, m) else 'PENDENTE'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('conferir', help='Confere apenas os arquivos necessários')
    prep = sub.add_parser('preparar', help='Monta uma entrada não rotulada para 900+897 textos')
    prep.add_argument('--reiniciar', action='store_true', help='Descarta resultados anteriores e prepara novamente')
    run = sub.add_parser('rodar', help='Treina os seis modelos originais e reproduz a votação V3')
    run.add_argument('--componente', choices=ORDER)
    run.add_argument('--forcar', action='store_true', help='Refaz o componente mesmo se houver CSV válido')
    sub.add_parser('status', help='Lista componentes concluídos')
    sub.add_parser('separar', help='Gera CSVs para test1 e test2 no formato do entrega_ep2.py')
    hidden = sub.add_parser('_executar', help=argparse.SUPPRESS)
    hidden.add_argument('name', choices=ORDER)
    args = parser.parse_args()
    try:
        if args.cmd == 'conferir':
            require_files()
            print('Sete scripts originais encontrados, além de train.xlsx, test1.xlsx e test2.xlsx.')
            print('Modelo original preservado: 6 componentes + votação V3.')
        elif args.cmd == 'preparar':
            prepare(args.reiniciar)
        elif args.cmd == 'rodar':
            run_all(args.componente, args.forcar)
        elif args.cmd == 'status':
            status()
        elif args.cmd == 'separar':
            split()
        else:
            run_child(args.name)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, subprocess.CalledProcessError) as e:
        print(f'ERRO: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
