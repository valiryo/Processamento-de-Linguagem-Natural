#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EP2 PLN — validação, exportação do teste2 e McNemar após congelamento.

Coloque ESTE arquivo em src/ep2/ e execute da raiz do repositório.
Não treina modelos, não escolhe hiperparâmetros e NÃO inventa predições.

Dependências: apenas a biblioteca padrão do Python 3.10+.
Lê as duas colunas dos XLSX oficiais do EP2 e preserva os demais
componentes do teste2.xlsx ao preencher a coluna clarity.

Entrada das predições:
  CSV com row_id,y_pred[,text_sha256] e 900 ou 897 linhas,
  conforme o conjunto. Labels: c1, c234, c5. row_id: 0,1,...,n-1.
  O CSV original do V3 do EP1 também é aceito (tem colunas extras).

IMPORTANTE: não usar teste1/teste2 para seleção de modelo.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

CLASSES = frozenset({'c1', 'c234', 'c5'})
NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
M = '{' + NS + '}'
SHEET = 'xl/worksheets/sheet1.xml'
SHARED = 'xl/sharedStrings.xml'
ET.register_namespace('', NS)
ET.register_namespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
ET.register_namespace('mc', 'http://schemas.openxmlformats.org/markup-compatibility/2006')
ET.register_namespace('x14ac', 'http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac')


def falhar(msg: str) -> None:
    raise ValueError(msg)


def shared_strings(arquivo: ZipFile) -> list[str]:
    if SHARED not in arquivo.namelist():
        return []
    root = ET.fromstring(arquivo.read(SHARED))
    return [''.join(t.text or '' for t in si.iter(M+'t'))
            for si in root.findall(M+'si')]


def conteudo_celula(cel: ET.Element, compartilhadas: list[str]) -> str:
    tipo = cel.get('t')
    if tipo == 'inlineStr':
        return ''.join(t.text or '' for t in cel.iter(M+'t'))
    node = cel.find(M+'v')
    if node is None or node.text is None:
        return ''
    valor = node.text
    if tipo == 's':
        pos = int(valor)
        if not 0 <= pos < len(compartilhadas):
            falhar(f'Índice inválido em sharedStrings: {pos}')
        return compartilhadas[pos]
    return valor


def carregar_planilha(caminho: Path, quantidade: int, rotulado: bool):
    """Retorna linhas originais e o XML da primeira aba.

    Contrato da disciplina: primeira aba, colunas resp_text e clarity,
    linhas consecutivas sem fórmulas nas duas colunas.
    """
    if not caminho.is_file():
        raise FileNotFoundError(f'Planilha não encontrada: {caminho}')
    with ZipFile(caminho, 'r') as z:
        if SHEET not in z.namelist():
            falhar(f'A planilha precisa ter a primeira aba em {SHEET}: {caminho}')
        raiz = ET.fromstring(z.read(SHEET))
        comp = shared_strings(z)

    dimension = raiz.find(M+'dimension')
    if dimension is None or dimension.get('ref') != f'A1:B{quantidade+1}':
        falhar(f'Intervalo inválido em {caminho}: esperado A1:B{quantidade+1}; '
               f'encontrado {dimension.get("ref") if dimension is not None else "sem dimensão"}')
    dados = raiz.find(M+'sheetData')
    if dados is None:
        falhar('Arquivo XLSX sem sheetData')
    encontrados = {}
    for linha in dados.findall(M+'row'):
        r = linha.get('r', '')
        if not r.isdigit() or int(r) < 1 or int(r) > quantidade+1:
            falhar(f'Linha fora do intervalo: {r!r}')
        indice = int(r)
        if indice in encontrados:
            falhar(f'Linha duplicada: {indice}')
        celulas = {}
        for cel in linha.findall(M+'c'):
            referencia = cel.get('r', '')
            if referencia not in (f'A{indice}', f'B{indice}'):
                falhar(f'Coluna/célula inesperada: {referencia!r}')
            if referencia[0] in celulas:
                falhar(f'Célula duplicada: {referencia}')
            celulas[referencia[0]] = cel
        encontrados[indice] = (linha, celulas)

    if set(encontrados) != set(range(1, quantidade+2)):
        faltando = set(range(1, quantidade+2)) - set(encontrados)
        falhar(f'Linhas ausentes em {caminho}: {sorted(faltando)[:10]}')
    cabecalho = [conteudo_celula(encontrados[1][1][c], comp) for c in ('A','B')]
    if cabecalho != ['resp_text', 'clarity']:
        falhar(f'Colunas inválidas em {caminho}: {cabecalho!r}')

    linhas = []
    for indice in range(2, quantidade+2):
        _, celulas = encontrados[indice]
        if 'A' not in celulas:
            falhar(f'Texto ausente na linha {indice}')
        texto = conteudo_celula(celulas['A'], comp)
        rotulo = conteudo_celula(celulas['B'], comp).strip() if 'B' in celulas else ''
        if not texto.strip():
            falhar(f'Texto vazio na linha {indice}')
        if rotulado and rotulo not in CLASSES:
            falhar(f'Rótulo inválido na linha {indice}: {rotulo!r}')
        if not rotulado and rotulo:
            falhar(f'O conjunto deveria estar sem rótulos; linha {indice}: {rotulo!r}')
        linhas.append((texto, rotulo))
    return linhas, raiz, encontrados


def conferir_original(caminho: Path, rotulado: list[tuple[str,str]]) -> None:
    original, _, _ = carregar_planilha(caminho, 900, rotulado=False)
    for i, (antes, agora) in enumerate(zip(original, rotulado)):
        if antes[0] != agora[0]:
            falhar(f'test1 rotulado fora da ordem original na linha {i+2} (row_id={i})')


def predicoes_csv(caminho: Path, textos: list[str]) -> list[str]:
    if not caminho.is_file():
        raise FileNotFoundError(f'CSV de predições não encontrado: {caminho}')
    with caminho.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {'row_id', 'y_pred'} <= set(reader.fieldnames):
            falhar(f'O CSV precisa de row_id,y_pred: {caminho}')
        entradas = list(reader)
    if len(entradas) != len(textos):
        falhar(f'Quantidade incorreta de predições em {caminho}: '
               f'{len(entradas)}; esperadas {len(textos)}')
    saida = []
    for i, (entrada, texto) in enumerate(zip(entradas, textos)):
        if (entrada['row_id'] or '').strip() != str(i):
            falhar(f'row_id incorreto na linha {i+2}: {entrada["row_id"]!r} (esperado {i})')
        classe = (entrada['y_pred'] or '').strip()
        if classe not in CLASSES:
            falhar(f'Classe inválida na linha {i+2}: {classe!r}')
        hash_texto = (entrada.get('text_sha256') or '').strip()
        if hash_texto and hash_texto.lower() != sha256(texto.encode('utf-8')).hexdigest():
            falhar(f'Hash incorreto do texto na linha {i+2}; confira o alinhamento!')
        saida.append(classe)
    return saida


def salvar_json(caminho: Path, resultado: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(resultado, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def comando_validar(a):
    t1, _, _ = carregar_planilha(a.test1, 900, rotulado=True)
    t2, _, _ = carregar_planilha(a.test2, 897, rotulado=False)
    if a.original_test1:
        conferir_original(a.original_test1, t1)
    resultado = {
        'test1': {'linhas': 900, 'classes': dict(sorted(Counter(r[1] for r in t1).items()))},
        'test2': {'linhas': 897, 'rotulos_preenchidos': 0},
        'test1_alinhado_original': bool(a.original_test1),
        'nota': 'Nenhuma avaliação nem seleção de modelo foi realizada.'
    }
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


def comando_template(a):
    linhas, _, _ = carregar_planilha(a.test2, 897, rotulado=False)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    with a.saida.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['row_id', 'y_pred', 'text_sha256'])
        for i, (texto, _) in enumerate(linhas):
            writer.writerow([i, '', sha256(texto.encode('utf-8')).hexdigest()])
    print(f'TEMPLATE GERADO (SEM PREDIÇÕES, NÃO SUBMETER): {a.saida}')


def comando_submeter(a):
    linhas, raiz, mapa = carregar_planilha(a.test2, 897, rotulado=False)
    preditos = predicoes_csv(a.predicoes, [r[0] for r in linhas])
    for num_linha, valor in enumerate(preditos, start=2):
        row, colunas = mapa[num_linha]
        if 'B' in colunas:
            cel = colunas['B']
            for child in list(cel):
                cel.remove(child)
        else:
            cel = ET.Element(M+'c', {'r': f'B{num_linha}'})
            row.append(cel)
        cel.set('t', 'inlineStr')
        filho_is = ET.SubElement(cel, M+'is')
        ET.SubElement(filho_is, M+'t').text = valor
    xml = ET.tostring(raiz, encoding='utf-8', xml_declaration=True)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    if a.saida.resolve() == a.test2.resolve():
        falhar('A saída não pode sobrescrever o test2.xlsx original.')
    with ZipFile(a.test2, 'r') as entrada, ZipFile(a.saida, 'w') as saida:
        for membro in entrada.infolist():
            conteudo = xml if membro.filename == SHEET else entrada.read(membro.filename)
            saida.writestr(membro, conteudo)
    conferido, _, _ = carregar_planilha(a.saida, 897, rotulado=True)
    if [r[0] for r in conferido] != [r[0] for r in linhas]:
        falhar('Atenção: os textos foram alterados na exportação!')
    if [r[1] for r in conferido] != preditos:
        falhar('Atenção: os rótulos foram alterados na exportação!')
    print(f'SUBMISSÃO VALIDADA: {a.saida} — 897 linhas, 2 colunas.')
    print('Distribuição:', dict(sorted(Counter(preditos).items())))


def mcnemar_exato(a: int, b: int) -> float:
    n = a+b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def comando_mcnemar(a):
    if not a.metodo_congelado:
        falhar('Para evitar uso indevido do teste1, informe --metodo-congelado '
               'após decidir o modelo EP2 sem olhar os rótulos do teste1.')
    linhas, _, _ = carregar_planilha(a.test1, 900, rotulado=True)
    conferir_original(a.original_test1, linhas)
    textos = [r[0] for r in linhas]
    verdade = [r[1] for r in linhas]
    ep1 = predicoes_csv(a.predicoes_ep1, textos)
    ep2 = predicoes_csv(a.predicoes_ep2, textos)
    ambos = sum(v1==v2==y for v1,v2,y in zip(ep1,ep2,verdade))
    so_ep1 = sum(v1==y and v2!=y for v1,v2,y in zip(ep1,ep2,verdade))
    so_ep2 = sum(v1!=y and v2==y for v1,v2,y in zip(ep1,ep2,verdade))
    nenhum = 900 - ambos - so_ep1 - so_ep2
    p = mcnemar_exato(so_ep1, so_ep2)
    resultado = {
        'metodo_ep2_congelado': True,
        'total': 900,
        'acuracia_ep1': (ambos+so_ep1)/900,
        'acuracia_ep2': (ambos+so_ep2)/900,
        'delta_ep2_menos_ep1_pp': 100*(so_ep2-so_ep1)/900,
        'contingencia': {
            'ambos_acertam': ambos,
            'somente_ep1_acerta': so_ep1,
            'somente_ep2_acerta': so_ep2,
            'ambos_erram': nenhum
        },
        'mcnemar_p_exato_bicaudal': p,
        'ep2_superior_e_significativo': so_ep2>so_ep1 and p<0.05,
        'alerta': 'Não mudar o método após conhecer este resultado.'
    }
    salvar_json(a.saida, resultado)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='comando', required=True)
    v = sub.add_parser('validar', help='Valida a estrutura dos arquivos oficiais')
    v.add_argument('--test1', type=Path, default=Path('data/raw/test1-rotulado-para-teste-estatistico.xlsx'))
    v.add_argument('--test2', type=Path, default=Path('data/raw/test2.xlsx'))
    v.add_argument('--original-test1', type=Path, default=Path('data/raw/test1.xlsx'))
    v.set_defaults(func=comando_validar)
    t = sub.add_parser('template', help='Gera CSV vazio com hashes de alinhamento (NÃO submeter)')
    t.add_argument('--test2', type=Path, default=Path('data/raw/test2.xlsx'))
    t.add_argument('--saida', type=Path, default=Path('results/ep2/final/template_test2_NAO_SUBMETER.csv'))
    t.set_defaults(func=comando_template)
    s = sub.add_parser('submeter', help='Cria XLSX final a partir de 897 predições reais')
    s.add_argument('--test2', type=Path, default=Path('data/raw/test2.xlsx'))
    s.add_argument('--predicoes', type=Path, default=Path('results/ep2/final/predicoes_test2_ep2.csv'))
    s.add_argument('--saida', type=Path, default=Path('results/ep2/final/test2_rotulado_EP2.xlsx'))
    s.set_defaults(func=comando_submeter)
    m = sub.add_parser('mcnemar', help='Compara EP1 × EP2 após congelar o método')
    m.add_argument('--test1', type=Path, default=Path('data/raw/test1-rotulado-para-teste-estatistico.xlsx'))
    m.add_argument('--original-test1', type=Path, default=Path('data/raw/test1.xlsx'))
    m.add_argument('--predicoes-ep1', type=Path, default=Path('results/ensemble_majority_5_v3_norberto_final.csv'))
    m.add_argument('--predicoes-ep2', type=Path, default=Path('results/ep2/final/predicoes_test1_ep2.csv'))
    m.add_argument('--saida', type=Path, default=Path('results/ep2/final/mcnemar_oficial.json'))
    m.add_argument('--metodo-congelado', action='store_true')
    m.set_defaults(func=comando_mcnemar)
    args = p.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, KeyError, ET.ParseError) as exc:
        print(f'ERRO: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
