"""Documenta e empacota resultados congelados; não ajusta modelo ou parâmetros.

Executar somente após candidato_incerteza.py congelar/inferir e McNemar.
O ZIP principal contém exclusivamente relatório PDF, apresentação PDF e teste2.
O código é disponibilizado em um arquivo complementar separado.
"""
from __future__ import annotations

from collections import Counter
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from xml.sax.saxutils import escape
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from candidato_incerteza import validate_candidate
from documentar_candidato import build_content, pct
from entrega_ep2 import carregar_planilha, conferir_original, predicoes_csv
from gating_discursivo import ROOT, dump, sha

BASE = ROOT / 'results/ep2/candidato_incerteza'
FINAL = BASE / 'final'
OUT = ROOT / 'results/ep2/entrega_grupo13'
TITLE = 'EP2 — Classificação de clareza em respostas e-SIC — Grupo 13'


def decimal_pt(value, digits=4):
    return f'{value:.{digits}f}'.replace('.', ',')


def audit_final():
    validate_candidate()
    freeze = json.loads((BASE / 'congelamento.json').read_text(encoding='utf-8'))
    if freeze['status'] != 'CONGELADO' or freeze['model_sha256'] != sha(BASE / 'modelo.json') or freeze['candidate_sha256'] != sha(BASE / 'candidato.json'):
        raise ValueError('Congelamento incompatível')
    manifest = json.loads((FINAL / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['freeze_sha256'] != sha(BASE / 'congelamento.json'):
        raise ValueError('Inferência incompatível com congelamento')
    for name, digest in manifest['outputs_sha256'].items():
        if sha(FINAL / name) != digest:
            raise ValueError(f'Saída de inferência alterada: {name}')
    truth, _, _ = carregar_planilha(ROOT / 'data/raw/test1-rotulado-para-teste-estatistico.xlsx', 900, rotulado=True)
    conferir_original(ROOT / 'data/raw/test1.xlsx', truth)
    texts = [r[0] for r in truth]
    y = np.asarray([r[1] for r in truth])
    ep1 = np.asarray(predicoes_csv(ROOT / 'results/ensemble_majority_5_v3_norberto_final.csv', texts))
    ep2 = np.asarray(predicoes_csv(FINAL / 'predicoes_test1.csv', texts))
    old_sheet = pd.read_excel(ROOT / 'entrega/test1_resultado_final.xlsx')
    if old_sheet.resp_text.astype(str).tolist() != texts or old_sheet.clarity.tolist() != ep1.tolist():
        raise ValueError('Predições históricas não correspondem à planilha entregue no EP1')
    a, b, c, d = [int(mask.sum()) for mask in [(ep1 == y) & (ep2 == y), (ep1 == y) & (ep2 != y), (ep1 != y) & (ep2 == y), (ep1 != y) & (ep2 != y)]]
    p = float(binomtest(min(b, c), b+c, .5).pvalue) if b+c else 1.
    stats = json.loads((FINAL / 'mcnemar_oficial.json').read_text(encoding='utf-8'))
    expected = {'ambos_acertam': a, 'somente_ep1_acerta': b, 'somente_ep2_acerta': c, 'ambos_erram': d}
    if stats['contingencia'] != expected or not np.isclose(stats['mcnemar_p_exato_bicaudal'], p, atol=1e-14):
        raise ValueError('McNemar não foi reproduzido')
    if stats['acuracia_ep1'] != (a+b)/900 or stats['acuracia_ep2'] != (a+c)/900:
        raise ValueError('Acurácias oficiais inconsistentes')
    original, _, _ = carregar_planilha(ROOT / 'data/raw/test2.xlsx', 897, rotulado=False)
    submission, _, _ = carregar_planilha(FINAL / 'test2_rotulado_GATE.xlsx', 897, rotulado=True)
    if [r[0] for r in original] != [r[0] for r in submission]:
        raise ValueError('Teste2 fora da ordem original')
    predicted = predicoes_csv(FINAL / 'predicoes_test2.csv', [r[0] for r in original])
    if predicted != [r[1] for r in submission]:
        raise ValueError('Planilha diferente do CSV')
    backups = json.loads((ROOT / 'results/ep2/validacao_2026-11-03.json').read_text(encoding='utf-8'))['hashes_sha256']
    for relative, digest in backups.items():
        if relative.startswith('results/ep2/final/') and sha(ROOT / relative) != digest:
            raise ValueError(f'Backup alterado: {relative}')
    result = {**stats, 'estatistica_mcnemar_exato_min_b_c': min(b, c), 'p_scipy_independente': p,
        'acertos_ep1': a+b, 'acertos_ep2': a+c, 'test1_alinhado': True, 'ep1_confere_planilha_entregue': True,
        'test2_linhas': 897, 'test2_colunas': ['resp_text', 'clarity'], 'test2_textos_e_ordem_preservados': True,
        'test2_classes': dict(Counter(predicted)), 'backup_preservado': True,
        'method_unchanged_after_test': True, 'freeze_sha256': sha(BASE / 'congelamento.json'),
        'model_sha256': sha(BASE / 'modelo.json')}
    return result


def content(stats):
    old, summary = build_content()
    blocks = []
    section = 0
    def p(s): blocks.append(('paragraph', s))
    def table(headers, rows): blocks.append(('table', (headers, rows)))
    for kind, value in old:
        if kind == 'heading':
            if value.startswith('EP2'):
                value = TITLE
            else:
                section = int(value.split('.')[0])
            value = value.replace('2. Estratégia principal e estado da escolha', '2. Estratégia principal escolhida').replace('5. Configuração do candidato', '5. Configuração final').replace('7. Resultados de desenvolvimento', '7. Resultados dos modelos desenvolvidos')
            blocks.append((kind, value))
            if section == 5:
                p('Os componentes reutilizados mantêm as configurações históricas abaixo. Não houve nova seleção desses hiperparâmetros com o teste1 ou teste2. Os scripts completos integram o pacote de código.')
                table(['Componente-base', 'Configuração histórica'], [
                    ['BERTimbau 512', 'Contexto 512; learning rate 3e-5; 2 épocas; batch físico 4; acumulação 4'],
                    ['BERTimbau soft-label', 'Contexto 256; learning rate 3e-5; 2 épocas; batch físico 8; acumulação 2; alvos por distribuição de duplicatas do treino'],
                    ['BERTimbau ordinal', 'Contexto 256; learning rate 3e-5; 2 épocas; batch físico 8; acumulação 2; dois limiares ordenados'],
                    ['NorBERTo', 'Contexto 1024; learning rate 2e-5; 2 épocas; batch físico 2; acumulação 8'],
                    ['LinearSVC', 'C=0,5; class_weight=balanced; TF-IDF word (1,2) e char_wb (3,5)'],
                    ['Baseline', 'TfidfVectorizer padrão; LogisticRegression com class_weight=balanced']])
                p('Nos quatro componentes neurais, o batch efetivo é 16, seed=42, weight decay=0,01 e precisão FP16. Esses componentes foram treinados em todo o treino para a inferência final; as probabilidades já estavam disponíveis em cache e foram reutilizadas pelo gate.')
            elif section == 8:
                table(['Estratégia', 'Conjunto', 'Acertos', 'Acurácia'], [
                    ['EP1 — Ensemble V3', 'teste1 (900)', str(stats['acertos_ep1']), pct(stats['acuracia_ep1'])],
                    ['EP2 — gate de incerteza', 'teste1 (900)', str(stats['acertos_ep2']), pct(stats['acuracia_ep2'])]])
                q = stats['contingencia']
                table(['', 'EP2 correto', 'EP2 errado'], [['EP1 correto', str(q['ambos_acertam']), str(q['somente_ep1_acerta'])], ['EP1 errado', str(q['somente_ep2_acerta']), str(q['ambos_erram'])]])
                p(f"O EP2 tem ganho numérico de quatro acertos (+{decimal_pt(stats['delta_ep2_menos_ep1_pp'])} ponto percentual). Há b={q['somente_ep1_acerta']} casos em que só o EP1 acerta e c={q['somente_ep2_acerta']} em que só o EP2 acerta. Foi aplicado McNemar exato bicaudal, usando a distribuição Binomial(b+c, 0,5); estatística min(b,c) = {stats['estatistica_mcnemar_exato_min_b_c']}. O p-valor é {decimal_pt(stats['mcnemar_p_exato_bicaudal'], 10)}.")
                p('Como p > 0,05, não foi demonstrada superioridade estatisticamente significativa do EP2. Isso não demonstra equivalência dos métodos; significa que estes dados não sustentam a alegação de melhoria pelo teste aplicado. Não foi efetuada nova seleção ou ajuste após observar o resultado.')
                p('A arquitetura e os pesos do gate foram congelados antes de gerar suas predições finais e abrir o gabarito do teste1. As probabilidades dos componentes-base já existiam em cache; a combinação final do EP2 foi produzida somente após esse congelamento. A comparação usa as predições históricas reais do EP1, conferidas contra a planilha entregue, e os mesmos 900 textos/rótulos. O p-valor foi recalculado com uma implementação independente, scipy.stats.binomtest, e coincide com o resultado do script de entrega.')
                p('O teste2 foi utilizado somente para inferência, com 897 rótulos: 296 c1, 273 c234 e 328 c5. O XLSX contém exclusivamente resp_text e clarity; cada texto e sua ordem foram comparados com o original. A acurácia do teste2 é desconhecida, pois seu gabarito não está disponível. A distribuição de previsões não foi usada para modificar o método.')
            elif section == 10:
                p('Repositório-base: https://github.com/valiryo/Processamento-de-Linguagem-Natural. O código do EP2 está organizado em src/ep2. O arquivo complementar codigo_grupo13.zip reúne os scripts, o modelo do gate e registros de auditoria, sem dados brutos, ambientes virtuais ou checkpoints neurais. Os artefatos gerados permanecem separados do código no controle de versão.')
                p('Para reproduzir o desenvolvimento: obter o repositório-base e os dados oficiais; utilizar Python 3.11.9 e as dependências abaixo. Os CSVs OOF históricos referenciados nos manifests são entradas necessárias. O pacote complementar inclui o gate final já congelado: nesse caso, não executar novamente preparar ou congelar. Para recalcular esses artefatos desde o início, usar uma cópia de trabalho separada, sem os arquivos candidato.json, modelo.json e congelamento.json previamente fornecidos; preservar o pacote original.')
                p('A inferência requer os caches históricos validados de results/ep2/inferencia_v3. Se eles não estiverem disponíveis, a sequência inferir_v3_ep2.py preparar, rodar e separar reconstrói os componentes com os scripts do EP1, incluindo treinamentos neurais. Essa reconstrução pode ser demorada e apresentar variação numérica; não é a execução rápida a partir dos caches originais. O teste1 rotulado permanece reservado à avaliação do método escolhido, nunca ao ajuste de parâmetros.')
                table(['Etapa', 'Comando PowerShell, na raiz'], [
                    ['Integridade', '.\\.venv\\Scripts\\python.exe src/ep2/gating_discursivo.py --check-data'],
                    ['Validação do gate', '.\\.venv\\Scripts\\python.exe src/ep2/gating_discursivo.py'],
                    ['Interpretação', '.\\.venv\\Scripts\\python.exe src/ep2/analisar_candidato.py'],
                    ['Treinamento final, em reprodução limpa', '.\\.venv\\Scripts\\python.exe src/ep2/candidato_incerteza.py preparar'],
                    ['Congelamento, em reprodução limpa', '.\\.venv\\Scripts\\python.exe src/ep2/candidato_incerteza.py congelar --confirmar-escolha-final'],
                    ['Inferência com caches íntegros', '.\\.venv\\Scripts\\python.exe src/ep2/candidato_incerteza.py inferir'],
                    ['Conferência e documentos', '.\\.venv\\Scripts\\python.exe src/ep2/finalizar_grupo13.py']])
                p('A reprodução deve preservar as escolhas congeladas; não usar o teste1 para refazer seleção. No workspace da entrega, as saídas já existem e não devem ser apagadas para repetir comandos. Para reprodução do desenvolvimento em pasta alternativa, usar --out results/ep2/gating_reproducao. O teste estatístico é executado por entrega_ep2.py mcnemar com --metodo-congelado e os caminhos dos CSVs, conforme README_ENTREGA.md.')
                table(['Dependência', 'Versão verificada'], [[name, importlib.metadata.version(name)] for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'openpyxl', 'torch', 'transformers', 'reportlab', 'python-docx', 'pypdf']])
                p('Foram verificados o gradiente analítico por diferenças finitas, o isolamento de grupos, a cobertura das 20.092 linhas de avaliação, a serialização do modelo, o bloqueio antes do congelamento e a exportação com dados sintéticos. A entrega real foi adicionalmente conferida quanto a hashes, quantidade de linhas, classes e ordem dos textos. Os PDFs e o DOCX possuem verificações estruturais.')
                p('Rastreabilidade: os hashes SHA-256 completos do modelo, do congelamento, das fontes e das saídas estão em candidato.json, congelamento.json e manifest_entrega.json. A geração deste relatório confere o modelo e as previsões contra esses registros. O arquivo grupo13.zip contém relatório PDF, apresentação PDF e a planilha do teste2; as predições individuais do teste1 ficam fora desse ZIP.')
            continue
        if section in (8, 10):
            continue
        if kind == 'paragraph':
            if value.startswith('RASCUNHO TÉCNICO'):
                value = 'ACH2118 — Introdução ao Processamento de Língua Natural — EACH/USP. Grupo 13. Modelo final: mistura de especialistas condicionada à incerteza. O método foi congelado antes da avaliação oficial; o ganho no teste1 não foi estatisticamente significativo.'
            elif value.startswith('Integrantes transcritos'):
                value = 'Participação efetiva dos quatro integrantes confirmada pelo grupo. Não há desistências informadas.'
            elif value.startswith('A tarefa prevê'):
                value = 'A tarefa prevê c1, c234 ou c5 a partir de resp_text. O modelo final do EP2 é uma mistura condicional de especialistas: os pesos variam com a incerteza das predições. Ele reutiliza Transformers em português e classificadores TF-IDF do EP1, substituindo a decisão final por votação por uma combinação aprendida. A hipótese foi selecionada apenas no desenvolvimento; o teste1 foi reservado para a comparação estatística após o congelamento.'
                value += ' Chamamos de gate o mecanismo que calcula esses pesos. As predições out-of-fold (OOF) são obtidas para exemplos não usados no treinamento do respectivo modelo-base. O V3 continua sendo uma das entradas; portanto, as sete entradas não correspondem a sete modelos-base independentes.'
            elif value.startswith('Os textos são convertidos'):
                value = 'Os textos de entrada são convertidos em string, como nos scripts históricos; foi encontrada uma célula numérica no índice 5026 (indexação iniciada em zero). Os arquivos originais e os rótulos são preservados. Os OOFs são alinhados por row_id e conferidos contra folds, classes verdadeiras e textos do treino. Não se usam metadados de conflito de rótulos como atributos do gate.'
                value += ' Os Transformers usam seus tokenizadores com truncamento no limite de contexto. Os componentes TF-IDF aplicam a conversão para minúsculas definida nos vetorizadores. O gate opera sobre as saídas desses componentes e não aplica nova limpeza aos textos.'
            elif value.startswith('A padronização dos sinais'):
                value = value.replace('é ajustada somente na metade de calibração', 'é ajustada somente na metade de calibração durante a avaliação; no ajuste final, usa todos os OOFs de treino')
            elif value.startswith('A contribuição técnica'):
                value = 'A contribuição técnica investigada é a seleção condicional de especialistas com calibração separada da avaliação. Ela difere da regra determinística do EP1. Porém, o ganho do componente condicional sobre pesos constantes é de apenas cinco acertos OOF; a ablação não oferece evidência forte de vantagem do mecanismo adaptativo. A contribuição é incremental e sua utilidade deve ser interpretada à luz desses limites.'
            elif value.startswith('Configuração candidata:'):
                value = value.replace('Configuração candidata:', 'Configuração final:')
            elif value.startswith('Na validação, cada gate'):
                value = value.replace('no candidato final', 'no modelo final').replace('Ainda não existe avaliação independente desse candidato final.', 'A avaliação em teste1, realizada após o congelamento, é apresentada na seção 8.')
            elif value.startswith('O guia menciona'):
                continue
            elif value.startswith('O V3 foi reproduzido'):
                value = 'O V3 foi reproduzido pelas funções originais de votação. Os novos arquivos estão em src/ep2 e results/ep2, com hashes SHA-256 das fontes e dos modelos. A etapa de desenvolvimento e avaliação do gate reutilizou os OOFs existentes e não exigiu novos treinamentos neurais nem alteração dos scripts históricos. Os experimentos anteriores do EP2, como TAPT, moBERTo e especialistas binários, envolveram treinamento neural e aparecem como histórico na seção 7.'
            elif value.startswith('O treino tem '):
                value = value.replace('20092', '20.092').replace('18432', '18.432')
            elif value.startswith('O gate de incerteza acerta '):
                value = value.replace('9586/', '9.586/').replace('9557/', '9.557/')
            elif value.startswith('Os casos foram amostrados'):
                value = value.replace('O grupo deve revisar essa leitura. ', '')
            value = value.replace('O candidato usa 11 sinais', 'O modelo final usa 11 sinais')
            if value.startswith('Caso row_id='):
                import re
                value = re.sub(r'(?<=: )(\d+)\.(\d{6})(?=\.)', r'\1,\2', value)
                value = value.replace('Caso row_id=', 'Exemplo de validação, row_id=')
        if kind == 'table':
            headers, rows = value
            value = ([s.replace('Recall candidato', 'Recall EP2') for s in headers], rows)
            if 'Cinco maiores coeficientes positivos' in headers:
                import re
                value = (headers, [[re.sub(r'(\d+)\.(\d{3})(?=\))', r'\1,\2', str(s)) for s in row] for row in rows])
        blocks.append((kind, value))
    pos = next(i for i, (kind, value) in enumerate(blocks) if kind == 'heading' and value.startswith('7.'))
    blocks.insert(pos+1, ('paragraph', 'As métricas globais abaixo são calculadas sobre as 20.092 predições de desenvolvimento concatenadas, e não pela média simples dos cinco folds. Macro-F1 é a média do F1 das três classes; recall mede a proporção de exemplos reais da classe recuperados. No gate, a avaliação também mantém separadas as metades de calibração e avaliação dentro de cada fold.'))
    return blocks, summary


def write_report(blocks):
    from docx import Document
    from docx.shared import Pt, Cm
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
    styles = getSampleStyleSheet()
    styles['Heading2'].keepWithNext = True
    styles.add(ParagraphStyle('Body13', fontName='Helvetica', fontSize=10, leading=13, spaceAfter=6, allowWidows=0, allowOrphans=0))
    styles.add(ParagraphStyle('Cell13', fontName='Helvetica', fontSize=8.5, leading=11, splitLongWords=True))
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Pt(50)
    section.top_margin, section.bottom_margin = Pt(40), Pt(45)
    section.footer_distance = Pt(20)
    doc.core_properties.title = TITLE
    doc.core_properties.author = 'Grupo 13'
    doc.styles['Normal'].font.name = 'Arial'
    doc.styles['Normal'].font.size = Pt(10)
    doc.styles['Normal'].paragraph_format.space_after = Pt(6)
    doc.styles['Normal'].paragraph_format.widow_control = True
    doc.styles['Heading 1'].paragraph_format.keep_with_next = True
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('ACH2118 | EP2 | Grupo 13 | Página ').font.size = Pt(8)
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    flow, md = [], []
    for kind, value in blocks:
        if kind == 'heading':
            doc.add_heading(value, level=1)
            flow.append(Paragraph(escape(value), styles['Heading2']))
            md.extend(['', '## ' + value, ''])
        elif kind == 'paragraph':
            doc.add_paragraph(value)
            flow.append(Paragraph(escape(value), styles['Body13']))
            md.extend([value, ''])
        else:
            headers, rows = value
            t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Light Shading Accent 1'
            widths = [140, 355] if len(headers) == 2 else [495/len(headers)] * len(headers)
            if headers[0] == 'Integrante': widths = [300, 195]
            t.autofit = False
            for column, width in zip(t.columns, widths): column.width = Pt(width)
            for cell, s in zip(t.rows[0].cells, headers): cell.text = s
            for row in rows:
                for cell, s in zip(t.add_row().cells, row): cell.text = str(s)
            for i, row in enumerate(t.rows):
                properties = row._tr.get_or_add_trPr()
                properties.append(OxmlElement('w:cantSplit'))
                if i == 0: properties.append(OxmlElement('w:tblHeader'))
                for cell, width in zip(row.cells, widths):
                    cell.width = Pt(width)
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.keep_with_next = i < len(t.rows)-1
                        for run in paragraph.runs: run.font.size = Pt(8.5)
            md.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
            md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows); md.append('')
            pdf_rows = [[Paragraph(escape(str(s)), styles['Cell13']) for s in row] for row in [headers]+rows]
            t = Table(pdf_rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dfeaf2')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            flow.append(KeepTogether([t, Spacer(1, 9)]))
    doc.save(OUT / 'Relatorio_EP2_grupo13.docx')
    (OUT / 'Relatorio_EP2_grupo13.md').write_text('\n'.join(md), encoding='utf-8')
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.grey)
        canvas.drawString(50, 28, 'ACH2118 | EP2 | Grupo 13 | Método congelado')
        canvas.drawRightString(A4[0]-50, 28, str(document.page))
    SimpleDocTemplate(str(OUT / 'Relatorio_EP2_grupo13.pdf'), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=40, bottomMargin=45).build(flow, onFirstPage=footer, onLaterPages=footer)


def write_slides(stats):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    slides = [
        ('Clareza em respostas e-SIC', ['EP2 · ACH2118 · EACH/USP · Grupo 13', 'Arthur Yang Tung · Kevin Tamayose', 'Augusto Prado Yassuda · Filipe Valeriano Batista de Oliveira', 'Modelo final: mistura condicionada à incerteza.'], 35),
        ('Dados e referência', ['20.092 exemplos; classes c1, c234 e c5.', '18.432 textos distintos; 291 grupos com rótulos conflitantes.', 'V3: 47,5662% de acurácia OOF.', 'Cinco folds preservados, agrupados por texto.'], 45),
        ('Evolução do método', ['EP1: cinco votos e desempate seletivo com NorBERTo.', 'EP2: pesos aprendidos a partir de entropia, margem e votos.', 'Sete entradas: quatro distribuições neurais e três votos codificados.', 'Gate softmax + interceptos de classe; regularização L2.'], 65),
        ('Calibração e avaliação separadas', ['Especialistas treinados fora de cada fold avaliado.', 'Dentro do fold: calibrar em A, avaliar em B; depois inverter.', 'Grupos de texto separados; scaler ajustado só na calibração.', 'Limite: modelos e hipóteses já foram selecionados nesses folds.'], 65),
        ('Resultados OOF e ablação', ['V3: 47,5662% · macro-F1 47,3933%', 'Pesos constantes: 47,6856% · macro-F1 47,4845%', 'Gate de incerteza: 47,7105% · macro-F1 47,5200%', 'Gate com discurso: 47,4020% · macro-F1 47,2061%'], 60),
        ('O tamanho do ganho', ['+29 acertos contra V3; 332 correções e 303 perdas.', 'Ganho em quatro folds; queda no fold 3.', 'Apenas +5 acertos sobre a mistura constante.', 'Recall c234 caiu de 37,2246% para 37,0641%.'], 45),
        ('Interpretação e erros', ['Baseline auxiliar: coeficientes globais e contribuições locais.', 'Caso 4: processo sem prazo; c234 confundida com c5.', 'Caso 76: restrição legal e orientação; fronteira c1/c234.', 'Caso 113: conteúdo em anexo, ausente da entrada.', 'A análise lexical não explica causalmente os Transformers.'], 75),
        ('Teste1: comparação oficial', [f"EP1: {stats['acertos_ep1']}/900 = {pct(stats['acuracia_ep1'])}", f"EP2: {stats['acertos_ep2']}/900 = {pct(stats['acuracia_ep2'])}", 'Só EP1 acerta: 37; só EP2 acerta: 41.', 'McNemar exato bicaudal: p = 0,734342.', 'Ganho numérico; sem superioridade significativa a 5%.'], 70),
        ('Conclusões e entrega', ['Gate de incerteza congelado antes da avaliação oficial.', 'Atributos discursivos não melhoraram o resultado.', 'Ganho pequeno; contribuição condicional ainda pouco sustentada.', 'Teste2: 897 rótulos, XLSX validado; acurácia desconhecida.', 'Código, protocolo, ablações e resultados rastreáveis.'], 50),
    ]
    path = OUT / 'Apresentacao_EP2_grupo13.pdf'
    c = canvas.Canvas(str(path), pagesize=(960, 540)); c.setTitle(TITLE)
    style = ParagraphStyle('Slide13', fontName='Helvetica', fontSize=22, leading=29, textColor=colors.HexColor('#25364c'))
    title_style = ParagraphStyle('Title13', fontName='Helvetica-Bold', fontSize=30, leading=34, textColor=colors.HexColor('#12243b'))
    duration = sum(s[2] for s in slides)
    notes = ['# Roteiro — Grupo 13', '', f'Tempo planejado: {duration//60}min{duration%60:02d}s. Ensaiar para respeitar o limite de dez minutos.', '']
    for i, (title, lines, seconds) in enumerate(slides, 1):
        c.setFillColor(colors.HexColor('#f7f9fc')); c.rect(0, 0, 960, 540, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#126b74')); c.rect(0, 524, 960, 16, fill=1, stroke=0)
        p = Paragraph(escape(title), title_style); _, ht = p.wrap(864, 90); p.drawOn(c, 48, 490-ht)
        y = 410
        for line in lines:
            p = Paragraph(escape(line), style); _, h = p.wrap(850, 400); p.drawOn(c, 52, y-h); y -= h+20
        if y < 50: raise ValueError(f'Slide {i} excede a área útil')
        c.setFont('Helvetica', 10); c.setFillColor(colors.HexColor('#68798a'))
        c.drawString(48, 24, 'ACH2118 | EP2 | Grupo 13')
        c.drawRightString(912, 24, f'{i:02d} / {len(slides):02d}'); c.showPage()
        notes += [f'## Slide {i} — {title} ({seconds}s)', '', *['- '+s for s in lines], '']
    c.save()
    (OUT / 'Roteiro_grupo13.md').write_text('\n'.join(notes), encoding='utf-8')


def package(stats):
    submission = OUT / 'test2_rotulado_grupo13.xlsx'
    shutil.copyfile(FINAL / 'test2_rotulado_GATE.xlsx', submission)
    names = ['Relatorio_EP2_grupo13.pdf', 'Apresentacao_EP2_grupo13.pdf', submission.name]
    with ZipFile(OUT / 'grupo13.zip', 'w', ZIP_DEFLATED) as z:
        for name in names: z.write(OUT / name, name)
    # Código complementar, sem saídas de teste1, dados brutos ou modelos gigantes.
    sources = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'src/ep2').glob('*.md')) + [ROOT / 'requirements.txt']
    for p in [BASE / 'modelo.json', BASE / 'candidato.json', BASE / 'congelamento.json',
              ROOT / 'results/ep2/gating_discursivo/protocolo_pre_especificado.json',
              ROOT / 'results/ep2/gating_discursivo/resumo.json',
              ROOT / 'results/ep2/gating_discursivo/auditoria.json',
              ROOT / 'results/ep2/gating_discursivo/oof_completo.csv',
              ROOT / 'results/ep2/validacao_2026-11-03.json',
              FINAL / 'mcnemar_oficial.json']:
        sources.append(p)
    sources += sorted((ROOT / 'results/ep2/gating_discursivo').glob('fold*_calibracao.json'))
    sources += sorted((ROOT / 'results/ep2/analise_candidato').glob('*.csv'))
    sources += sorted((ROOT / 'results/ep2/analise_candidato').glob('*.json'))
    with ZipFile(OUT / 'codigo_grupo13.zip', 'w', ZIP_DEFLATED) as z:
        for path in sources: z.write(path, path.relative_to(ROOT).as_posix())
        z.write(OUT / 'README_ENTREGA.md', 'README_ENTREGA.md')
    files = names + ['grupo13.zip', 'codigo_grupo13.zip', 'Relatorio_EP2_grupo13.docx', 'Relatorio_EP2_grupo13.md', 'Roteiro_grupo13.md', 'README_ENTREGA.md']
    dump(OUT / 'manifest_entrega.json', {'grupo': 13, 'metodo': 'gate de incerteza congelado',
        'validacao': stats, 'zip_principal_conteudo': names, 'codigo_publicado_remotamente': False,
        'arquivos_sha256': {name: sha(OUT / name) for name in files},
        'fontes_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), FINAL / 'mcnemar_oficial.json', FINAL / 'manifest.json', BASE / 'congelamento.json', ROOT / 'src/ep2/documentar_candidato.py']}})


def main(report_only=False):
    stats = audit_final()
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, _ = content(stats)
    write_report(blocks)
    if report_only:
        if not (OUT / 'Apresentacao_EP2_grupo13.pdf').is_file() or not (OUT / 'Roteiro_grupo13.md').is_file():
            raise FileNotFoundError('A revisão somente do relatório requer apresentação e roteiro existentes')
    else:
        write_slides(stats)
    readme = '''# Entrega EP2 — Grupo 13

Método final: gate de incerteza, congelado antes de suas predições finais e do McNemar. As probabilidades dos componentes-base já existiam em cache.
Integrantes confirmados: Arthur Yang Tung, Kevin Tamayose, Augusto Prado Yassuda e Filipe Valeriano Batista de Oliveira.

## Arquivos

- `grupo13.zip`: relatório PDF, apresentação PDF e XLSX de teste2 com 897 linhas.
- `codigo_grupo13.zip`: código complementar; disponibilizar também no prazo. Sobrepor à raiz do repositório-base, sem mudar os caminhos.
- `Relatorio_EP2_grupo13.docx`: versão editável fora do ZIP principal.
- `Roteiro_grupo13.md`: roteiro de nove slides, aproximadamente 8min30s; ensaiar antes da apresentação.
- `manifest_entrega.json`: hashes dos arquivos e validações.

Nenhum arquivo foi enviado ao professor ou publicado no GitHub automaticamente. O código atualizado deve ser disponibilizado pelo grupo. A planilha que deve ser entregue é `test2_rotulado_grupo13.xlsx`, dentro de `grupo13.zip`; não usar o XLSX de backup do V3.

## Resultado oficial

Teste1: EP1 = 434/900 (48,2222%); EP2 = 438/900 (48,6667%). McNemar exato bicaudal: p=0,7343420330, estatística min(b,c)=37. Ganho numérico de quatro acertos, SEM superioridade estatisticamente significativa. O método não será ajustado usando esse resultado. Teste2: acurácia desconhecida.

## Reprodução

Repositório-base: https://github.com/valiryo/Processamento-de-Linguagem-Natural
Usar os dados oficiais em `data/raw/` e os folds/OOFs/caches originais indicados nos manifests. O pacote de código não inclui dados brutos ou checkpoints. A falta de caches exige reconstrução pelos scripts históricos, com possível custo de GPU.
Executar PowerShell na raiz. Em uma cópia limpa, a sequência é:

```powershell
.\\.venv\\Scripts\\python.exe src/ep2/gating_discursivo.py --check-data
.\\.venv\\Scripts\\python.exe src/ep2/gating_discursivo.py
.\\.venv\\Scripts\\python.exe src/ep2/analisar_candidato.py
.\\.venv\\Scripts\\python.exe src/ep2/candidato_incerteza.py preparar
.\\.venv\\Scripts\\python.exe src/ep2/candidato_incerteza.py congelar --confirmar-escolha-final
.\\.venv\\Scripts\\python.exe src/ep2/candidato_incerteza.py inferir
.\\.venv\\Scripts\\python.exe src/ep2/entrega_ep2.py mcnemar --metodo-congelado --predicoes-ep2 results/ep2/candidato_incerteza/final/predicoes_test1.csv --saida results/ep2/candidato_incerteza/final/mcnemar_oficial.json
.\\.venv\\Scripts\\python.exe src/ep2/finalizar_grupo13.py
```

No workspace atual, não repetir inferência nem apagar saídas: todas essas etapas já foram concluídas. A preparação recusa retreinar um candidato congelado. O modelo e o congelamento fornecidos no pacote devem ser usados tal como estão; em uma reprodução a partir do pacote congelado, pular `preparar` e `congelar`.

Os hashes identificam precisamente dados, códigos e modelos utilizados. O candidato.json original conserva seu estado histórico de preparação; o congelamento.json posterior é o registro vigente da decisão final. Os antigos rascunhos e o guia de desenvolvimento são registros históricos, não a entrega final.
'''
    # A instrução auditada da raiz prevalece sobre o texto histórico embutido.
    audited_readme = ROOT / 'README_ENTREGA.md'
    if audited_readme.is_file():
        readme = audited_readme.read_text(encoding='utf-8')
    (OUT / 'README_ENTREGA.md').write_text(readme, encoding='utf-8')
    package(stats)
    if (ROOT / 'src/ep2/empacotar_reproducao.py').is_file() and audited_readme.is_file():
        from empacotar_reproducao import pack
        pack(OUT)
    print(json.dumps({'entrega': str(OUT / 'grupo13.zip'), 'codigo': str(OUT / 'codigo_grupo13.zip'), 'mcnemar_p': stats['mcnemar_p_exato_bicaudal'], 'test2_linhas': stats['test2_linhas']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--somente-relatorio', action='store_true')
    parser.add_argument('--saida', type=Path, help='Pasta alternativa para os documentos e pacotes')
    args = parser.parse_args()
    if args.saida:
        OUT = args.saida.resolve()
        OUT.relative_to(ROOT)
    main(report_only=args.somente_relatorio)
