"""Empacota código e artefatos reais para reprodução; preserva grupo13.zip e método."""
from pathlib import Path
import hashlib
import json
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from reproduzir_ep2 import sha

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'results/ep2/entrega_grupo13'


def pack(out=DEFAULT_OUT):
    out = Path(out)
    candidate = ROOT / 'results/ep2/candidato_incerteza'
    m = json.loads((candidate / 'candidato.json').read_text(encoding='utf-8'))
    # Os hashes científicos são apenas conferidos, nunca recalculados para aceitar alterações.
    for name, expected in m['sources_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Fonte científica alterada: {name}')
    if sha(candidate / 'modelo.json') != m['model_sha256']:
        raise ValueError('Modelo congelado alterado')
    artifacts = {ROOT / n for n in m['sources_sha256'] if not n.startswith('src') and not n.startswith('data\\raw')}
    artifacts.update([ROOT / 'results/baseline_oof.csv', ROOT / 'results/tfidf_word_char_linearsvc_oof.csv'])
    cache = ROOT / 'results/ep2/inferencia_v3'
    artifacts.update(p for p in cache.rglob('*') if p.is_file())
    # Históricos e interpretação: permitem recalcular tabelas/atributos sem treino neural.
    for folder in ['gating_discursivo', 'analise_candidato']:
        artifacts.update(p for p in (ROOT / 'results/ep2' / folder).iterdir() if p.is_file())
    for folder in ['tapt_bertimbau', 'moberto_512']:
        artifacts.update((ROOT / 'results/ep2' / folder).glob('fold*_seed42/predicoes_fold.csv'))
    for name in ['bertimbau_duplicate_soft_labels_256_cv.csv', 'bertimbau_ordinal_256_cv.csv', 'norberto_base_1024_cv.csv']:
        artifacts.add(ROOT / 'results' / name)
    frozen = [candidate / n for n in ['modelo.json', 'candidato.json', 'congelamento.json', 'final/mcnemar_oficial.json']]
    code = sorted((ROOT / 'src').rglob('*.py'))
    manifest = {
        'schema': 1,
        'ambiente': 'Windows x64, CPython 3.11.9; reprodução congelada em CPU',
        'arquivos': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(code + frozen) | artifacts)},
        'dados_externos': {f'data/raw/{n}': sha(ROOT / 'data/raw' / n) for n in ['train.xlsx', 'test1.xlsx', 'test2.xlsx']},
        'mcnemar_externo': {
            'gabarito_sha256': sha(ROOT / 'data/raw/test1-rotulado-para-teste-estatistico.xlsx'),
            'ep1_sha256': sha(ROOT / 'results/ensemble_majority_5_v3_norberto_final.csv'),
            'ep1_sha256_lf': hashlib.sha256((ROOT / 'results/ensemble_majority_5_v3_norberto_final.csv').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
            'ep1_url': 'https://raw.githubusercontent.com/valiryo/Processamento-de-Linguagem-Natural/9461ced46632d90ebf185a5bbd35b6f166247248/results/ensemble_majority_5_v3_norberto_final.csv'},
        'saidas_congeladas_sha256': json.loads((candidate / 'final/manifest.json').read_text(encoding='utf-8'))['outputs_sha256'],
        'test2_distribuicao': {'c1': 296, 'c234': 273, 'c5': 328},
        'artefatos_zip': [p.relative_to(ROOT).as_posix() for p in sorted(artifacts)],
        'nota': 'Não contém planilhas oficiais brutas, checkpoints neurais ou CSV final individual de teste1. Caches intermediários contêm as saídas dos componentes para 900+897 textos; são entradas da reprodução, nunca a planilha de submissão.'}
    path = ROOT / 'src/ep2/artefatos_reproducao.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    docs = [ROOT / n for n in ['README_ENTREGA.md', 'REPRODUCAO_EP2.md', 'AUDITORIA_FINAL_EP2.md']]
    sources = code + frozen + docs + [path] + sorted(ROOT.glob('requirements*.txt')) + sorted((ROOT / 'src/ep2').glob('*.md'))
    # Preserva os bytes das fontes. Git pode normalizar LF/CRLF e invalidar hashes históricos.
    with ZipFile(out / 'codigo_grupo13.zip', 'w', ZIP_DEFLATED) as z:
        for p in sorted(set(sources)):
            z.write(p, p.relative_to(ROOT).as_posix())
        evidence = ROOT / 'results/ep2/auditoria_2026-11-08'
        for p in sorted(list(evidence.glob('*.json')) + list(evidence.glob('*.log'))):
            if p.name == 'pacotes_finais.json':
                continue  # Hash do próprio ZIP fica externo, evitando autorreferência.
            z.write(p, 'auditoria_2026-11-08/' + p.name)
    with ZipFile(out / 'artefatos_reproducao_grupo13.zip', 'w', ZIP_DEFLATED) as z:
        for p in sorted(artifacts):
            z.write(p, p.relative_to(ROOT).as_posix())
    for p in docs:
        shutil.copy2(p, out / p.name)
    delivery = out / 'manifest_entrega.json'
    current = json.loads(delivery.read_text(encoding='utf-8'))
    # Preserve a proveniência dos documentos, cujo conteúdo não foi refeito.
    current.setdefault('fontes_documentos_antes_auditoria_sha256', dict(current['fontes_sha256']))
    current['fontes_sha256'] = {n: sha(ROOT / n) for n in current['fontes_sha256']}
    for n in ['codigo_grupo13.zip', 'artefatos_reproducao_grupo13.zip', *[p.name for p in docs]]:
        current['arquivos_sha256'][n] = sha(out / n)
    current['auditoria_reproducao'] = {'data_local': '2026-11-08', 'manifest': 'src/ep2/artefatos_reproducao.json',
        'manifest_sha256': sha(path), 'publicacao_desta_revisao': 'PENDENTE; nenhum push realizado pela auditoria'}
    current['auditoria_reproducao']['documentos_principais_preservados'] = True
    delivery.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({n: (out / n).stat().st_size for n in ['grupo13.zip', 'codigo_grupo13.zip', 'artefatos_reproducao_grupo13.zip']}, indent=2))


if __name__ == '__main__':
    pack()
