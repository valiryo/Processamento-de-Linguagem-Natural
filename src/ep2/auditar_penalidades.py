"""Audita a entrega congelada sem treinar ou selecionar modelos.

O baseline é avaliado apenas em treino/OOF e teste1 já liberado.
Teste2 serve exclusivamente para validar o formato e a identidade da entrega.
--remote-main-observado registra uma consulta externa feita separadamente.
"""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pypdf import PdfReader

from gating_discursivo import ROOT, LABELS, dump, sha
from finalizar_grupo13 import audit_final
import inferir_v3_ep2 as cached


def main(args):
    out = ROOT / 'results/ep2/auditoria_penalidades'
    out.mkdir(parents=True, exist_ok=True)
    package = ROOT / 'results/ep2/entrega_grupo13'
    final = audit_final()
    manifest = json.loads((package / 'manifest_entrega.json').read_text(encoding='utf-8'))
    for name, expected in manifest['arquivos_sha256'].items():
        if sha(package / name) != expected:
            raise ValueError(f'Arquivo alterado desde a entrega: {name}')
    with ZipFile(package / 'grupo13.zip') as z:
        if z.testzip() is not None or sorted(z.namelist()) != sorted(manifest['zip_principal_conteudo']):
            raise ValueError('ZIP inválido ou com arquivos inesperados')
        for name in z.namelist():
            import hashlib
            if hashlib.sha256(z.read(name)).hexdigest() != manifest['arquivos_sha256'][name]:
                raise ValueError(f'Conteúdo interno do ZIP divergente: {name}')
        sheet = pd.read_excel(io.BytesIO(z.read('test2_rotulado_grupo13.xlsx')))
        report = PdfReader(io.BytesIO(z.read('Relatorio_EP2_grupo13.pdf')))
        slides = PdfReader(io.BytesIO(z.read('Apresentacao_EP2_grupo13.pdf')))
    original = pd.read_excel(ROOT / 'data/raw/test2.xlsx')
    if sheet.shape != (897, 2) or sheet.columns.tolist() != ['resp_text', 'clarity']:
        raise ValueError('Formato incorreto da planilha dentro do ZIP')
    if sheet.resp_text.tolist() != original.resp_text.tolist() or not sheet.clarity.isin(LABELS).all():
        raise ValueError('Textos, ordem ou rótulos inválidos')
    report_text = '\n'.join(p.extract_text() or '' for p in report.pages)
    headings = [int(n) for n in re.findall(r'^(\d{1,2})\. ', report_text, re.M)]
    if headings != list(range(1, 11)):
        raise ValueError('Relatório não contém as dez seções esperadas')
    with ZipFile(package / 'codigo_grupo13.zip') as z:
        if z.testzip() is not None:
            raise ValueError('Pacote de código corrompido')
        if not {'src/ep2/gating_discursivo.py', 'src/ep2/candidato_incerteza.py'}.issubset(z.namelist()):
            raise ValueError('Pacote não contém os scripts do modelo')
        if any(n.endswith('.xlsx') or n.endswith('/predicoes_test1.csv') for n in z.namelist()):
            raise ValueError('Pacote de código contém planilha ou predições individuais de teste1')
    train = pd.read_excel(ROOT / 'data/raw/train.xlsx')
    folds = pd.read_csv(ROOT / 'data/splits/folds.csv')
    baseline = pd.read_csv(ROOT / 'results/baseline_oof.csv').sort_values('row_index')
    gate = pd.read_csv(ROOT / 'results/ep2/gating_discursivo/oof_completo.csv').sort_values('row_id')
    y, f = train.clarity.to_numpy(), folds.fold.to_numpy()
    if baseline.row_index.tolist() != list(range(len(y))) or baseline.classe_real.tolist() != y.tolist() or baseline.fold.tolist() != f.tolist():
        raise ValueError('Baseline OOF desalinhado')
    if gate.row_id.tolist() != list(range(len(y))) or gate.fold.tolist() != f.tolist() or np.array(LABELS)[gate.y_true_id].tolist() != y.tolist():
        raise ValueError('Gate OOF desalinhado')
    majority = np.empty(len(y), dtype=object)
    rows = []
    for fold in range(5):
        tr, va = f != fold, f == fold
        chosen = pd.Series(y[tr]).value_counts().idxmax()
        majority[va] = chosen
        rows.append({'fold': fold, 'n': int(va.sum()), 'classe_majoritaria_do_treino': chosen,
            'majoritario': float((majority[va] == y[va]).mean()),
            'baseline_oficial': float((baseline.predicao.to_numpy()[va] == y[va]).mean()),
            'ep2': float((np.array(LABELS)[gate.pred_incerteza.to_numpy()[va]] == y[va]).mean())})
    averages = {name: float(np.mean([r[name] for r in rows])) for name in ['majoritario', 'baseline_oficial', 'ep2']}
    pooled = {'majoritario': float((majority == y).mean()), 'baseline_oficial': float((baseline.predicao.to_numpy() == y).mean()),
              'ep2': float((np.array(LABELS)[gate.pred_incerteza] == y).mean())}
    cache_manifest = cached.load_manifest()
    if not cached.check_cache('baseline', cache_manifest):
        raise ValueError('Cache do baseline final não validado')
    # O arquivo combina inferências, mas só as primeiras 900 linhas de teste1 são avaliadas.
    b = pd.read_csv(cached.COMP / 'baseline_final.csv', nrows=900)
    if b.row_id.tolist() != list(range(900)):
        raise ValueError('Baseline final desalinhado')
    t1 = pd.read_excel(ROOT / 'data/raw/test1-rotulado-para-teste-estatistico.xlsx')
    from hashlib import sha256
    if [sha256(str(t).encode('utf-8')).hexdigest() for t in t1.resp_text] != cache_manifest['hashes_test1']:
        raise ValueError('Textos do baseline final desalinhados')
    test1 = {'majoritario': float((t1.clarity == train.clarity.value_counts().idxmax()).mean()),
             'baseline_oficial': float((b.y_pred.to_numpy() == t1.clarity.to_numpy()).mean()), 'ep2': final['acuracia_ep2']}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-l', 'HEAD'], cwd=ROOT, text=True, encoding='utf-8')
    large = []
    for line in tree.splitlines():
        info, path = line.split('\t', 1)
        values = info.split()
        if values[1] == 'blob' and values[3].isdigit() and int(values[3]) > 100*1024*1024:
            large.append({'path': path, 'bytes': int(values[3])})
    result = {'data_auditoria': '2026-11-04', 'zip_e_hashes_validos': True, 'test2_897x2_textos_ordem_labels_validos': True,
        'rotulos_individuais_test1_ausentes_dos_pacotes': True, 'relatorio_secoes': headings,
        'relatorio_paginas': len(report.pages), 'apresentacao_pdf_paginas': len(slides.pages),
        'por_fold': rows, 'media_aritmetica_cinco_folds': averages, 'oof_global': pooled, 'test1': test1,
        'comparacao_test2_realizada': False, 'desempenho_test2': 'DESCONHECIDO; gabarito indisponível',
        'congelamento_e_artefatos_verificados': True, 'head_local': head, 'main_remoto_observado': args.remote_main_observado,
        'blobs_git_maiores_100_MiB': large, 'envio_no_prazo': 'NÃO COMPROVADO; limite oficial 08/11',
        'codigo': 'Pacote complementar disponível localmente; conferir publicação/envio ao professor',
        'limite_auditoria': 'Artefatos e código disponíveis não comprovam ações externas ao registro desta sessão nem garantem desempenho no teste2.',
        'fontes_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), package / 'grupo13.zip', package / 'codigo_grupo13.zip', ROOT / 'data/raw/train.xlsx', ROOT / 'data/raw/test2.xlsx', ROOT / 'data/raw/test1-rotulado-para-teste-estatistico.xlsx']}}
    dump(out / 'auditoria.json', result)
    lines = ['# Auditoria das penalidades — Grupo 13', '', 'Data: 04/11/2026. Modelo congelado preservado; nenhuma nova seleção ou alteração de predições.', '',
        '| Conjunto | Majoritário | Baseline oficial | EP2 |', '|---|---:|---:|---:|']
    for title, metrics in [('Média dos 5 folds', averages), ('OOF global', pooled), ('Teste1', test1)]:
        lines.append('| '+title+' | '+' | '.join(f'{100*metrics[name]:.4f}%' for name in ['majoritario', 'baseline_oficial', 'ep2'])+' |')
    lines += ['', '## Verificações', '',
        '- ZIP principal íntegro: relatório PDF, apresentação PDF e somente a planilha do teste2.',
        '- Planilha dentro do ZIP: 897 linhas, duas colunas, classes válidas, todos os textos na ordem original.',
        '- Relatório com as dez seções; apresentação em PDF de nove páginas; código complementar íntegro.',
        '- Predições individuais do teste1 ausentes dos dois pacotes; métricas agregadas de McNemar constam no relatório como exigido.',
        '- Comparação de desempenho somente em treino/OOF e teste1. Teste2 usado para inferência/validação de formato, sem seleção por desempenho.',
        '- Fontes, modelo e saídas correspondem aos hashes do congelamento e da entrega.', '', '## Pontos não garantidos', '',
        '- Desempenho do teste2 desconhecido: não é possível garantir ausência das penalidades baseadas nessa acurácia.',
        '- Envio no prazo não comprovado: entregar relatório, código, apresentação e XLSX até 08/11 pelo responsável do EP1.',
        f'- HEAD local: `{head}`. Main remoto observado separadamente: `{args.remote_main_observado}`.',
        f'- O commit local contém {len(large)} arquivos maiores que 100 MiB; não assumir que um push normal publicará o código.',
        '- GitHub bloqueia arquivos maiores que 100 MiB em Git normal: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github',
        '- O código pode ser disponibilizado pelo pacote complementar; confirmar canal aceito pelo professor. A versão remota deve incluir o EP2 se for o meio de disponibilização escolhido.',
        '- McNemar p=0,734342 não demonstra superioridade; isso afeta o critério positivo de 5%, não constitui uma das penalidades de 20% citadas.',
        '', '## Arquivos grandes no commit', '']
    lines += [f"- {r['path']}: {r['bytes']/1024**2:.1f} MiB" for r in large]
    (out / 'auditoria.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'medias_5_folds': averages, 'oof': pooled, 'test1': test1, 'arquivos_git_grandes': len(large), 'relatorio': str(out / 'auditoria.md')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-main-observado', default='não consultado')
    main(parser.parse_args())
