"""Gera rascunhos rastreáveis (MD/DOCX/PDF e slides PDF); sem consultar testes."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import sys
from xml.sax.saxutils import escape

import pandas as pd

from gating_discursivo import ROOT, dump, sha

OUT = ROOT / 'results/ep2/documentacao_2026-11-03'
GATE = ROOT / 'results/ep2/gating_discursivo'
ANALYSIS = ROOT / 'results/ep2/analise_candidato'


def pct(x):
    return f'{100 * x:.4f}%'.replace('.', ',')


def build_content():
    summary = json.loads((GATE / 'resumo.json').read_text(encoding='utf-8'))
    audit = json.loads((GATE / 'auditoria.json').read_text(encoding='utf-8'))
    hist = pd.read_csv(ANALYSIS / 'experimentos_reproduzidos.csv')
    local = json.loads((ANALYSIS / 'interpretacao_local.json').read_text(encoding='utf-8'))
    global_weights = pd.read_csv(ANALYSIS / 'importancia_global_baseline.csv')
    m = summary['models']
    blocks = []
    def h(text): blocks.append(('heading', text))
    def p(text): blocks.append(('paragraph', text))
    def table(headers, rows): blocks.append(('table', (headers, rows)))
    h('EP2 — Classificação de clareza em respostas e-SIC')
    p('RASCUNHO TÉCNICO — 03/11/2026. Candidato em desenvolvimento; não submeter como relatório concluído. Por decisão do usuário, o teste1 rotulado permanece fechado. Não há método EP2 congelado, McNemar oficial nem novas predições de teste do candidato.')
    h('1. Integrantes efetivos do grupo')
    p('Integrantes transcritos do relatório do EP1; o grupo deve confirmar a participação efetiva e informar seu número antes da entrega. Não foi presumida nenhuma desistência.')
    table(['Integrante', 'Número USP'], [['Arthur Yang Tung', '14559819'], ['Kevin Tamayose', '14669711'], ['Augusto Prado Yassuda', '12543210'], ['Filipe Valeriano Batista de Oliveira', '14570701']])
    h('2. Estratégia principal e estado da escolha')
    p('A tarefa prevê c1, c234 ou c5 a partir de resp_text. O V3 do EP1 combina Transformers em português e classificadores TF-IDF. A proposta avaliada no EP2 aprende uma mistura condicional de especialistas: os pesos variam com a incerteza das predições. O gate de incerteza é o candidato atual, ainda sem escolha final. O V3 permanece como alternativa de segurança.')
    p('Há sete entradas na mistura: distribuições de BERTimbau 512, BERTimbau soft-label 256, BERTimbau ordinal e NorBERTo; votos do LinearSVC, do baseline e do V3. Para os três votos, a distribuição auxiliar é 0,6 na classe prevista e 0,2 nas demais. Esses valores são uma codificação fixa, não probabilidades calibradas dos classificadores.')
    p('Para atributos padronizados z, os pesos são softmax([1,z] W). A distribuição m é a soma das distribuições dos especialistas ponderadas por esses pesos. A saída é softmax(log(m) + b), com três interceptos de classe b. W e b minimizam entropia cruzada com regularização L2. O candidato usa 11 sinais: quatro entropias normalizadas, quatro margens entre as duas maiores probabilidades e três frações de votos dos cinco componentes originais.')
    p('A hipótese original acrescentava 53 atributos estruturais e discursivos. Ela não passou no critério pré-especificado: precisava superar o V3 e as duas ablações, com ganho em pelo menos três folds. O destaque posterior da variante de incerteza surgiu da ablação e é uma decisão exploratória, registrada separadamente em candidato.json. Não foi apresentado como aprovação pelo critério original.')
    p('A contribuição técnica investigada é a seleção condicional de especialistas com calibração separada da avaliação. Ela difere da regra determinística do EP1, mas o ganho do componente condicional sobre pesos constantes é de apenas cinco acertos; não há evidência suficiente para prometer inovação valorizada pelo professor ou melhoria confirmada.')
    h('3. Pré-processamento')
    p('Os textos são convertidos em string, como nos scripts históricos; foi encontrada uma célula numérica na linha de índice 5026. Não se alteram rótulos, textos originais ou folds. Os OOFs são alinhados por row_id e conferidos contra folds, classes verdadeiras e textos do treino. Não se usam metadados de conflito de rótulos como atributos.')
    p('A padronização dos sinais do gate é ajustada somente na metade de calibração. No experimento discursivo, as regras fixas medem 27 atributos estruturais e 26 marcadores lexicais. A normalização interna desses marcadores não substitui o texto nem a chave dos grupos; vetores iguais de contagens não são tratados como duplicatas do documento.')
    h('4. Parâmetros e alternativas avaliadas')
    p('Não foi executada nova busca em grade de hiperparâmetros neurais. Foi feita uma ablação com configurações fixas, sem tentativa de otimizar p-valores. Os parâmetros abaixo foram registrados antes da execução em protocolo_pre_especificado.json.')
    table(['Parâmetro', 'Valor/alternativas'], [['Atributos do gate', 'constante; 11 de incerteza; 11 + 53 discursivos'], ['L2 do gate / interceptos', '0,02 / 0,02'], ['Otimização', 'L-BFGS-B, até 300 iterações, ftol=1e-10'], ['Divisão interna', '2 partes por grupo, StratifiedGroupKFold, seed=42'], ['Distribuição auxiliar do voto', '(0,6; 0,2; 0,2), permutada pela classe prevista'], ['Normalização de atributos', 'StandardScaler ajustado apenas na calibração']])
    h('5. Configuração do candidato')
    p('Configuração candidata: gate de incerteza com 11 atributos e os mesmos hiperparâmetros da seção 4. Esses valores são configurações fixas avaliadas, não ótimos demonstrados por busca exaustiva. O arquivo modelo.json contém os pesos, interceptos e estatísticas de padronização ajustados em todos os 20.092 OOFs. A serialização foi verificada contra a predição em memória.')
    p('Na validação, cada gate foi ajustado em aproximadamente 2.000 linhas; no candidato final, usa-se todo o conjunto OOF. Essa diferença de tamanho de calibração e o treinamento final dos especialistas em todo o treino podem mudar o comportamento no teste. Ainda não existe avaliação independente desse candidato final.')
    h('6. Procedimento experimental e reprodutibilidade')
    p(f"O treino tem {audit['n']} exemplos, {audit['unique_texts']} textos distintos, {audit['duplicate_groups']} grupos repetidos e {audit['conflicting_groups']} grupos com rótulos conflitantes. As classes têm 6.347, 6.853 e 6.892 ocorrências. Os cinco folds históricos são preservados; textos idênticos não cruzam folds.")
    p('Para cada fold externo f, os especialistas foram treinados nos outros quatro folds. Somente os OOFs de f são usados para o gate dessa avaliação. O próprio f é dividido em duas partes com grupos de texto: calibra-se em A e avalia-se em B, depois calibra-se em B e avalia-se em A. Assim, nem o gate nem os especialistas treinam com os rótulos das linhas em que são avaliados. Treinar o gate nos OOFs dos outros quatro folds seria inadequado, pois esses especialistas poderiam ter aprendido com f.')
    p('Esse protocolo preserva todas as 20.092 linhas de avaliação com uma previsão por linha. Ele não elimina o otimismo decorrente da escolha histórica dos modelos-base e das múltiplas hipóteses exploradas nesses folds. As métricas são de desenvolvimento. A comparação com V3 é pareada; a ausência de vazamento direto nesta nova calibração não transforma o desenvolvimento em teste confirmatório.')
    p('O V3 foi reproduzido pelas funções originais de votação. Os novos arquivos estão em src/ep2 e results/ep2. Fontes, modelos e configurações possuem SHA-256 nos manifests. Não houve novo treinamento neural, download de modelos nem alteração dos scripts históricos.')
    h('7. Resultados de desenvolvimento')
    table(['Modelo', 'Acurácia', 'Macro-F1', 'Recall c234'], [[r.modelo, pct(r.accuracy), pct(r.macro_f1), pct(r.recall_c234)] for r in hist.itertuples()])
    table(['Fold', 'V3', 'Constante', 'Incerteza', 'Discurso'], [[str(f)] + [pct(next(r['accuracy'] for r in summary['folds'] if r['fold'] == f and r['model'] == name)) for name in ['v3', 'constante', 'incerteza', 'discurso']] for f in range(5)])
    p(f"O gate de incerteza acerta {m['incerteza']['correct']}/20.092 contra {m['v3']['correct']}/20.092 do V3: +29 acertos, ou +0,1443 ponto percentual. Corrige 332 erros e perde 303 acertos. Melhora em quatro folds e piora no fold 3. O recall de c234 passa de {pct(m['v3']['recall']['c234'])} para {pct(m['incerteza']['recall']['c234'])}; portanto, a motivação de recuperar a classe intermediária não foi confirmada.")
    p('A mistura constante acerta 9.581 exemplos, apenas cinco a menos que o gate de incerteza. Essa comparação limita a atribuição do ganho à adaptação dos pesos. Já a inclusão dos sinais discursivos reduz o resultado para 9.524 acertos, 33 abaixo do V3, e não foi promovida.')
    table(['Classe', 'Recall V3', 'Recall candidato'], [[c, pct(m['v3']['recall'][c]), pct(m['incerteza']['recall'][c])] for c in ['c1', 'c234', 'c5']])
    p('Histórico adicional: especialista binário c234 e classificadores de fronteiras adjacentes foram executados apenas no fold 0, com 47,2754% e 44,3643%, respectivamente. O primeiro não mudou nenhuma previsão do V3. Os resultados de TAPT e moBERTo da tabela foram recalculados diretamente dos CSVs dos cinco folds.')
    p('O guia menciona 42,53% para atributos discursivos isolados e 43,56% para limpeza de texto. Os resultados brutos desses dois ensaios não foram localizados no repositório nem nos ZIPs inspecionados; esses números não entram na tabela de resultados reproduzidos. O script discursivo existe, mas o script da limpeza não foi localizado. O consenso TAPT+moBERTo possui resultados locais e deve ser citado como experimento exploratório, pois foi concebido após observar o fold 0.')
    h('8. Evolução em relação ao EP1 e significância')
    table(['Método', 'Conjunto', 'Acurácia / estatística'], [['EP1 V3', 'teste1, 900 exemplos', 'PENDENTE — gabarito não aberto'], ['EP2 candidato', 'teste1, 900 exemplos', 'PENDENTE — sem inferência final'], ['McNemar exato bicaudal', 'EP1 versus EP2 no teste1', 'PENDENTE — nenhum p-valor oficial calculado']])
    p('Após a decisão definitiva: registrar congelamento; produzir 897 predições do teste2 e 900 do teste1 original; validar alinhamento; só então abrir o teste1 rotulado e compará-lo com as predições históricas reais do EP1. Relatar acurácias, tabela 2x2, discordâncias, estatística exata min(b,c), p-valor bicaudal e direção do efeito. Não reabrir a seleção após conhecer esse resultado.')
    p('Não foi demonstrada superioridade estatística. A acurácia do teste2 permanece desconhecida. O XLSX existente em results/ep2/final/test2_rotulado_EP2.xlsx é backup do V3 e não representa o novo candidato.')
    h('9. Interpretação global, local e análise de erros')
    p('Conforme o modelo de relatório do professor, usa-se o baseline TF-IDF + regressão logística como classificador auxiliar interpretável. Os coeficientes abaixo provêm de ajuste em todo o treino e descrevem associação lexical. Eles não são explicações causais do gate nem dos Transformers. Nomes próprios, órgãos e temas entre os maiores coeficientes sugerem dependência de contexto institucional, que pode não generalizar.')
    table(['Classe', 'Cinco maiores coeficientes positivos'], [[c, '; '.join(f'{r.termo} ({r.coeficiente:.3f})' for r in global_weights.loc[(global_weights.classe == c) & (global_weights.direcao == 'positivo')].head(5).itertuples())] for c in ['c1', 'c234', 'c5']])
    p('A interpretação local usa um baseline ajustado fora do fold 0. Suas 4.019 previsões coincidem com o histórico. Para cada palavra, calcula-se TF-IDF × (coeficiente da classe prevista - coeficiente da segunda classe), somando a diferença entre interceptos. A soma de todas as contribuições recompõe numericamente a margem entre os dois logits. O JSON disponibiliza os 12 termos de maior efeito absoluto por caso; esses 12 sozinhos não precisam recompor a margem completa.')
    selected = {r['row_id']: r for r in local}
    for i, explanation in [
        (4, 'A resposta informa processo em análise, mas sem prazo conclusivo, e remete a um endereço externo. Essa combinação de conteúdo parcial e orientação ajuda a formular uma hipótese para a confusão com c5, sem demonstrar a causa do erro. O termo institucional planejamento favorece c5 no baseline em +0,6365 de margem contra c234.'),
        (76, 'O texto combina restrição de escopo da LAI, informação pessoal e indicação de outro departamento. Uma resposta que orienta o cidadão pode também parecer uma negativa; o baseline fica muito próximo da fronteira c1/c234. O efeito de solicitação é +0,2080 para c1 contra c234, sem implicar que o termo determine clareza.'),
        (113, 'A resposta diz que o conteúdo segue em anexo. Como o anexo não está em resp_text, faltam evidências que podem justificar o rótulo c5. A limitação é da entrada disponível, além de uma possível falha de classificação. Não foi acessado o documento anexado.')]:
        r = selected[i]
        p(f"Caso row_id={i}, fold 0: verdadeiro={r['real']}; V3={r['v3']}; gate={r['gate_incerteza']}; baseline={r['baseline']}. Margem do baseline contra {r['segunda_baseline']}: {r['margem_logit']:.6f}. {explanation}")
    p('Os casos foram amostrados por pares de confusão e lidos para elaborar hipóteses qualitativas; não constituem avaliação humana independente ou representativa do corpus. O grupo deve revisar essa leitura. Outros três casos e suas contribuições estão em interpretacao_local.json. Duplicatas conflitantes limitam classificadores determinísticos, mas não demonstram que a acurácia global de 47–48% seja um teto da tarefa.')
    h('10. Código e reprodução')
    p('Repositório: https://github.com/valiryo/Processamento-de-Linguagem-Natural. Os novos arquivos estão apenas no workspace local; não houve commit ou push. Antes da entrega, o grupo deve disponibilizar o código. A apresentação PDF preparada tem nove slides e roteiro de aproximadamente nove minutos, ainda com a seção oficial pendente.')
    p('Executar na raiz com o Python da .venv. Sequência já executada: gating_discursivo.py --check-data; gating_discursivo.py --fold 0; gating_discursivo.py; analisar_candidato.py; candidato_incerteza.py preparar; test_gating_discursivo.py -v; documentar_candidato.py. Os comandos completos e as etapas futuras estão em src/ep2/CONTINUACAO_EP2.md.')
    table(['Dependência', 'Versão verificada'], [[name, importlib.metadata.version(name)] for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'openpyxl', 'torch', 'transformers', 'reportlab', 'python-docx', 'pypdf']])
    p('Pendências antes da entrega: decidir e congelar o método; inferência do candidato; avaliação McNemar; preencher esta seção com resultados reais; confirmar número e integrantes do grupo; revisar texto e slides; gerar grupoXX.zip com relatório, apresentação PDF e somente a planilha de teste2. Não empacotar como entrega final enquanto essas pendências permanecerem.')
    return blocks, summary


def write_report(blocks):
    from docx import Document
    from docx.shared import Pt
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.pagesizes import A4
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyEP2', fontName='Helvetica', fontSize=9, leading=13, spaceAfter=7, alignment=TA_LEFT))
    styles.add(ParagraphStyle('CellEP2', fontName='Helvetica', fontSize=8, leading=10))
    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(10)
    md = []
    flow = []
    for kind, content in blocks:
        if kind == 'heading':
            md.extend(['', '## ' + content, ''])
            doc.add_heading(content, level=1)
            flow.append(Paragraph(escape(content), styles['Heading2']))
        elif kind == 'paragraph':
            md.extend([content, ''])
            doc.add_paragraph(content)
            flow.append(Paragraph(escape(content), styles['BodyEP2']))
        else:
            headers, rows = content
            md.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
            md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
            md.append('')
            table = doc.add_table(rows=1, cols=len(headers))
            table.style = 'Light Shading Accent 1'
            for cell, text in zip(table.rows[0].cells, headers): cell.text = text
            for row in rows:
                for cell, text in zip(table.add_row().cells, row): cell.text = str(text)
            data = [[Paragraph(escape(str(s)), styles['CellEP2']) for s in row] for row in [headers] + rows]
            widths = [165, 330] if len(headers) == 2 else [495 / len(headers)] * len(headers)
            t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e1eaf1')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey),
                ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            flow.extend([t, Spacer(1, 10)])
    (OUT / 'Relatorio_EP2_RASCUNHO.md').write_text('\n'.join(md), encoding='utf-8')
    doc.save(OUT / 'Relatorio_EP2_RASCUNHO.docx')
    def footer(canvas, document):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(50, 28, 'EP2 | RASCUNHO | método não congelado | teste1 fechado')
        canvas.drawRightString(A4[0] - 50, 28, str(document.page))
    SimpleDocTemplate(str(OUT / 'Relatorio_EP2_RASCUNHO.pdf'), pagesize=A4,
                      leftMargin=50, rightMargin=50, topMargin=40, bottomMargin=45).build(flow, onFirstPage=footer, onLaterPages=footer)


def write_slides(summary):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    slides = [
        ('Clareza em respostas e-SIC', ['EP2 · ACH2118 · EACH/USP', 'Candidato: mistura de especialistas condicionada à incerteza.', 'Rascunho: teste1 fechado; método ainda não congelado.'], 35),
        ('O desafio e a referência', ['20.092 exemplos; classes c1, c234 e c5.', '18.432 textos distintos; 291 grupos com rótulos conflitantes.', 'V3: 47,5662% de acurácia OOF; recall c234 = 37,2246%.', 'Cinco folds preservados, agrupados por texto.'], 55),
        ('O que mudou na hipótese', ['V3: cinco votos e desempate seletivo com NorBERTo.', 'Proposta: pesos aprendidos conforme entropias, margens e frações de votos.', '7 entradas: 4 distribuições neurais + 3 votos codificados.', '53 sinais discursivos adicionais foram avaliados e rejeitados.'], 65),
        ('Como evitamos vazamento na calibração', ['Para cada fold f, especialistas treinados fora de f.', 'Dentro de f: calibrar em A e avaliar em B; depois inverter.', 'A e B separam grupos de texto; o scaler também usa só calibração.', 'A seleção histórica com estes folds continua sendo uma limitação.'], 65),
        ('Resultados e ablação', ['V3: 47,5662% · macro-F1 47,3933%', 'Mistura constante: 47,6856% · macro-F1 47,4845%', 'Gate de incerteza: 47,7105% · macro-F1 47,5200%', 'Gate discursivo: 47,4020% · macro-F1 47,2061%'], 70),
        ('O ganho é pequeno', ['Incerteza: +29 acertos contra V3; 332 correções e 303 perdas.', 'Melhora em quatro folds; piora no fold 3.', 'Apenas +5 acertos contra a mistura constante.', 'Recall c234 caiu para 37,0641%; motivação original não confirmada.'], 55),
        ('Interpretação e erros', ['Baseline auxiliar: coeficientes globais por classe e contribuições locais exatas.', 'Caso 4: processo em análise e sem prazo; c234 confundida com c5.', 'Caso 76: restrição legal e orientação; fronteira c1/c234.', 'Caso 113: resposta em anexo, indisponível ao classificador.', 'Associação lexical não explica causalmente o ensemble.'], 80),
        ('Comparação oficial: pendente', ['Teste1: 900 exemplos reservados para McNemar após congelamento.', 'Não há p-valor oficial nem superioridade estatística demonstrada.', 'Teste2: 897 rótulos ainda não produzidos pelo candidato.', 'O XLSX existente é backup V3, sem alegação de inovação.'], 45),
        ('Decisão atual e próximos passos', ['Gate de incerteza preparado como candidato; V3 preservado.', 'Congelar somente após a decisão final do grupo.', 'Inferir, executar McNemar, atualizar relatório e slides.', 'Entregar relatório + apresentação PDF + teste2 XLSX em grupoXX.zip.', 'Código e números rastreáveis; nenhuma melhora foi fabricada.'], 50),
    ]
    c = canvas.Canvas(str(OUT / 'Apresentacao_EP2_RASCUNHO.pdf'), pagesize=(960, 540))
    c.setTitle('EP2 — Candidato de incerteza — RASCUNHO')
    style = ParagraphStyle('slide', fontName='Helvetica', fontSize=22, leading=30, textColor=colors.HexColor('#25364c'))
    script = ['# Roteiro da apresentação — rascunho', '', f'Duração planejada: {sum(s[2] for s in slides)} segundos (aproximadamente 8min40s). Ensaiar; atualizar resultados oficiais antes de apresentar.', '']
    for i, (title, lines, seconds) in enumerate(slides, 1):
        c.setFillColor(colors.HexColor('#f7f9fc')); c.rect(0, 0, 960, 540, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#126b74')); c.rect(0, 524, 960, 16, fill=1, stroke=0)
        c.setFillColor(colors.HexColor('#12243b')); c.setFont('Helvetica-Bold', 30); c.drawString(48, 466, title)
        y = 405
        for line in lines:
            para = Paragraph(escape(line), style)
            _, height = para.wrap(850, 400)
            para.drawOn(c, 52, y-height)
            y -= height + 22
        if y < 55:
            raise ValueError(f'Slide {i} excede a área útil')
        c.setFont('Helvetica', 10); c.setFillColor(colors.HexColor('#68798a'))
        c.drawString(48, 24, 'ACH2118 | RASCUNHO — revisão e avaliação oficial pendentes')
        c.drawRightString(912, 24, f'{i:02d} / {len(slides):02d}')
        c.showPage()
        script += [f'## Slide {i} — {title} ({seconds}s)', '', *['- '+line for line in lines], '']
    c.save()
    (OUT / 'Roteiro_apresentacao.md').write_text('\n'.join(script), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, summary = build_content()
    write_report(blocks)
    write_slides(summary)
    files = ['Relatorio_EP2_RASCUNHO.md', 'Relatorio_EP2_RASCUNHO.docx', 'Relatorio_EP2_RASCUNHO.pdf', 'Apresentacao_EP2_RASCUNHO.pdf', 'Roteiro_apresentacao.md']
    dump(OUT / 'manifest.json', {'status': 'RASCUNHOS_NAO_SUBMETER', 'test_labels_read': False,
        'sources_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [GATE / 'resumo.json', GATE / 'auditoria.json', ANALYSIS / 'experimentos_reproduzidos.csv', ANALYSIS / 'interpretacao_local.json', ANALYSIS / 'importancia_global_baseline.csv', Path(__file__)]},
        'outputs_sha256': {name: sha(OUT / name) for name in files}})
    print(f'Rascunhos gerados em {OUT}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
