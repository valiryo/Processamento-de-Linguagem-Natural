"""EP2 / ACH2118: classificador de sinais discursivos e estruturais de respostas e-SIC.

Executar na raiz do repositório: python src/ep2/discurso_features.py
Requer: numpy, scipy, scikit-learn (não requer GPU).
O acesso .xlsx usa ZIP/XML da biblioteca padrão para não requerer openpyxl.
Não utiliza teste1 nem teste2; mantém os cinco folds congelados do EP1.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LABEL2ID = {'c1': 0, 'c234': 1, 'c5': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
TOKEN_RE = re.compile(r'\b[\wÀ-ÿ]+(?:[-’][\wÀ-ÿ]+)*\b', re.UNICODE)
SENTENCE_RE = re.compile(r'[.!?]+(?=\s|$)')
URL_RE = re.compile(r'\b(?:https?://|www\.)\S+', re.I)
MAIL_RE = re.compile(r'[\w.+-]+@[\w.-]+\.[a-z]{2,}', re.I)
DATE_RE = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
PCT_RE = re.compile(r'\b\d+(?:[,.]\d+)?\s*%')
PHONE_RE = re.compile(r'\b(?:0?800\s*\d|\(?\d{2}\)?\s*\d{4,5}[-\s]?\d{4})\b')
ENUM_RE = re.compile(r'(?:^|[\n\r])\s*(?:[0-9]{1,2}[).:-]|[-*•])\s')

# Regras concebidas manualmente, sem inspeção de rótulos ou instâncias do teste.
# Texto é normalizado para letras minúsculas e sem diacríticos.
LEXICAL_PATTERNS = {
  'saudacao': r'\b(?:prezad[oa]s?|senhor[ae]s?|car[oa]s?|bom dia|boa tarde|boa noite)\b',
  'despedida': r'\b(?:atenciosamente|cordialmente|respeitosamente|sem mais|abracos|cumprimentos)\b',
  'agradecimento': r'\b(?:agradec\w*|obrigad\w*|gratos?)\b',
  'informar': r'\b(?:informam\w*|esclarec\w*|comunicam\w*|respondem\w*|ressaltam\w*|notificam\w*)\b',
  'resposta_explicita': r'\b(?:segue\w*|sao os seguintes|a resposta e|em resposta|conforme solicitado|em atendimento|em atencao)\b',
  'negacao': r'\b(?:nao|nenhum[ao]s?|inexiste\w*|impossiv\w*|jamais|nunca|sem informacao)\b',
  'indisponibilidade': r'\b(?:nao (?:dispomos|possuimos|temos|localizamos|consta|ha)|informac\w* indisponiv\w*|nao foi possivel|inexistencia de dados|sem registro)\b',
  'justificativa': r'\b(?:porque|pois|portanto|devido|em virtude|em razao|tendo em vista|uma vez que|visto que|dessa forma|deste modo|assim sendo)\b',
  'causalidade': r'\b(?:motivo|causa|razao|justific\w*|explica\w*|fundament\w*)\b',
  'orientacao': r'\b(?:acesse|consulte|clique|procure|solicite|entre em contato|devera|pode ser obtid\w*|orientam\w*|recomendam\w*)\b',
  'encaminhamento': r'\b(?:encaminh\w*|redirecion\w*|competencia de outro|orgao responsavel|canal apropriado|setor responsavel|ouvidoria)\b',
  'prazo': r'\b(?:prazo|dias uteis|dias corridos|ate \d+ dias|a partir de|dentro de|contados? da|vencimento)\b',
  'anexo': r'\b(?:anex\w*|arquivo\w*|planilha\w*|documento\w* junt\w*|seguem os arquivos|segue arquivo)\b',
  'legislacao': r'\b(?:lei|decreto|portaria|resolucao|normativa|dispositiv\w*|legislacao|sumula|jurisprudencia|artigo|art\.)\b',
  'transparencia': r'\b(?:transparencia|acesso a informacao|e-sic|sic|informacao ao cidadao|lai|12\.527)\b',
  'recurso': r'\b(?:recurso|recorrer|recursal|interpo\w*|instancia|contest\w*|decisao)\b',
  'sigilo': r'\b(?:sigilo|sigilos\w*|confidencia\w*|dados pessoais|privacidade|reservad\w*|restrit\w*)\b',
  'incerteza': r'\b(?:talvez|possivelmente|aproximad\w*|estim\w*|na medida do possivel|eventualmente|provavel\w*)\b',
  'condicional': r'\b(?:caso|se|desde que|contanto que|na hipotese|quando necessario)\b',
  'contraste': r'\b(?:porem|contudo|entretanto|todavia|no entanto|embora|apesar de)\b',
  'referencia_externa': r'\b(?:site|pagina|portal|endereco eletronico|internet|link|plataforma|sistema|consultar no)\b',
  'restricao_escopo': r'\b(?:nao compete|nao e competencia|nao se enquadra|nao e atribui\w*|fora do escopo|natureza distinta)\b',
  'detalhamento': r'\b(?:detalh\w*|especific\w*|abaixo|a seguir|primeiramente|segundo|terceiro|relacao|tabela)\b',
  'procedimento': r'\b(?:procedimento|passo a passo|preench\w*|formulario|cadastro|protocolo|registr\w*|senha)\b',
  'dados_quantitativos': r'\b(?:total|quantidade|numero|percentual|valores|tabela|estatistica|serie historica|periodo)\b',
  'cortesia': r'\b(?:disposicao|colocamo-nos|apreciamos|contato|atendimento|esclarecimentos adicionais)\b',
}
LEX_RE = {k: re.compile(v) for k, v in LEXICAL_PATTERNS.items()}

CORE_FEATURES = [
 'log_caracteres', 'log_palavras', 'log_frases', 'log_paragrafos',
 'media_caracteres_palavra', 'diversidade_lexical', 'proporcao_palavras_longa',
 'comprimento_medio_frase', 'repeticao_trigramas', 'maior_palavra',
 'digitos_100_palavras', 'maiusculas_100_palavras',
 'virgulas_100_palavras', 'pontos_100_palavras', 'pontos_interrogacao_100_palavras',
 'dois_pontos_100_palavras', 'ponto_virgula_100_palavras', 'aspas_100_palavras',
 'barras_100_palavras', 'hifens_100_palavras',
 'qtd_urls', 'qtd_emails', 'qtd_datas', 'qtd_percentuais', 'qtd_telefones',
 'qtd_itens_enumerados', 'log_tamanho_primeiro_paragrafo',
]
DISCOURSE_FEATURES = list(LEXICAL_PATTERNS.keys())
FEATURE_NAMES = CORE_FEATURES + DISCOURSE_FEATURES


def strip_accents(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))


def read_train_xlsx(path: Path) -> tuple[list[str], np.ndarray]:
    """Read two-column original workbook with shared strings, preserving row order."""
    with zipfile.ZipFile(path) as z:
        shared = []
        with z.open('xl/sharedStrings.xml') as stream:
            for _, el in ET.iterparse(stream, events=('end',)):
                if el.tag == NS + 'si':
                    shared.append(''.join(node.text or '' for node in el.iter(NS + 't')))
                    el.clear()
        rows = []
        with z.open('xl/worksheets/sheet1.xml') as stream:
            for _, el in ET.iterparse(stream, events=('end',)):
                if el.tag == NS + 'row':
                    cells = {}
                    for c in el.findall(NS + 'c'):
                        col = re.match(r'[A-Z]+', c.attrib['r']).group()
                        val = c.find(NS + 'v')
                        inline = c.find(NS + 'is')
                        if val is not None:
                            cells[col] = shared[int(val.text)] if c.attrib.get('t') == 's' else (val.text or '')
                        elif inline is not None:
                            cells[col] = ''.join(t.text or '' for t in inline.iter(NS + 't'))
                    rows.append(cells)
                    el.clear()
    if not rows or rows[0].get('A') != 'resp_text' or rows[0].get('B') != 'clarity':
        raise ValueError('Esperadas colunas A=resp_text, B=clarity na planilha original.')
    texts, targets = [], []
    for row in rows[1:]:
        if row.get('B') not in LABEL2ID or not row.get('A'):
            raise ValueError(f'Dado inválido no índice {len(texts)}: {row.get("B")!r}')
        texts.append(row['A'])
        targets.append(LABEL2ID[row['B']])
    return texts, np.asarray(targets, dtype=np.int64)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def features(text: str) -> np.ndarray:
    tokens = TOKEN_RE.findall(text)
    lower = strip_accents(text.lower())
    words = [strip_accents(t.lower()) for t in tokens]
    n = max(len(words), 1)
    char_count = len(text)
    n_sentences = len(SENTENCE_RE.findall(text))
    paras = [p for p in re.split(r'\n\s*\n|\n\s*(?=[\-•\d]+[).])', text) if p.strip()]
    if not paras:
        paras = [text]
    trigrams = list(zip(words, words[1:], words[2:]))
    core = [
        math.log1p(char_count), math.log1p(len(words)), math.log1p(n_sentences), math.log1p(len(paras)),
        sum(map(len, words)) / n, len(set(words))/n, sum(len(w) >= 10 for w in words)/n,
        len(words)/max(n_sentences, 1),
        (len(trigrams)-len(set(trigrams)))/max(len(trigrams), 1),
        max((len(w) for w in words), default=0),
        100*sum(ch.isdigit() for ch in text)/n,
        100*sum(1 for t in tokens if len(t)>1 and t.isupper())/n,
        100*text.count(',')/n, 100*text.count('.')/n, 100*text.count('?')/n,
        100*text.count(':')/n, 100*text.count(';')/n,
        100*(text.count('"')+text.count('“')+text.count('”'))/n,
        100*text.count('/')/n, 100*text.count('-')/n,
        len(URL_RE.findall(text)), len(MAIL_RE.findall(text)), len(DATE_RE.findall(text)),
        len(PCT_RE.findall(text)), len(PHONE_RE.findall(text)), len(ENUM_RE.findall(text)),
        math.log1p(len(paras[0])),
    ]
    counts = [100*len(p.findall(lower))/n for p in LEX_RE.values()]
    return np.array(core+counts, dtype=np.float32)


def get_model(kind: str):
    if kind in ('structural', 'discursive', 'completo'):
        return HistGradientBoostingClassifier(max_iter=110, learning_rate=.055,
            max_leaf_nodes=13, max_depth=4, min_samples_leaf=50,
            l2_regularization=10., early_stopping=False, random_state=42)
    if kind == 'linear':
        return make_pipeline(StandardScaler(), LogisticRegression(
            C=.2, max_iter=500, solver='lbfgs', random_state=42))
    raise ValueError(kind)


def evaluate(pred: np.ndarray, y: np.ndarray) -> dict:
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro')),
            'matriz_confusao': confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
            'recall': [float(((pred==c)&(y==c)).sum()/max((y==c).sum(),1)) for c in (0,1,2)]}


def save_csv(path: Path, columns: list[str], records: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.writer(f);writer.writerow(columns);writer.writerows(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=Path('data/raw/train.xlsx'))
    p.add_argument('--folds', type=Path, default=Path('data/splits/folds.csv'))
    p.add_argument('--v3', type=Path, default=Path('results/ensemble_majority_5_v3_norberto_oof.csv'))
    p.add_argument('--out', type=Path, default=Path('results/ep2/discurso'))
    p.add_argument('--only-fold', type=int, default=None, choices=range(5))
    p.add_argument('--check-data', action='store_true')
    p.add_argument('--feature-cache', type=Path, default=None, help='Apenas para desenvolvimento: arquivo .npy calculado previamente com a mesma ordem e as mesmas features.')
    args=p.parse_args(); t0=perf_counter()
    texts,y=read_train_xlsx(args.train)
    foldrows=load_csv(args.folds); v3rows=load_csv(args.v3)
    N=len(texts)
    assert N==20092==len(y)==len(foldrows)==len(v3rows), 'Contagens divergentes.'
    folds=np.empty(N,dtype=int);v3=np.empty(N,dtype=int)
    for i,(fr,vr) in enumerate(zip(foldrows,v3rows)):
        assert int(fr['row_index'])==i==int(vr['row_id']), f'Índices divergentes na linha {i}'
        assert int(fr['fold'])==int(vr['fold']), f'Fold divergente na linha {i}'
        assert int(vr['y_true_id'])==int(y[i]), f'Rótulo divergente na linha {i}'
        folds[i]=int(fr['fold']);v3[i]=int(vr['y_pred_id_v3'])
    assert sorted(set(folds))==[0,1,2,3,4]
    for f in range(5):
        a={texts[i] for i in range(N) if folds[i]==f}
        b={texts[i] for i in range(N) if folds[i]!=f}
        assert not (a&b), f'Vazamento de textos idênticos no fold {f}'
    print(f'Alinhamento e agrupamento OK: {N} respostas / V3={accuracy_score(y,v3):.6f}.',flush=True)
    if args.check_data:return
    if args.feature_cache is not None:
        X = np.load(args.feature_cache)
        if X.shape != (N,len(FEATURE_NAMES)):
            raise ValueError(f'Cache de características incompatível: {X.shape}')
    else:
        outx = []
        for i,s in enumerate(texts):
            outx.append(features(s))
            if (i+1)%2500 == 0: print(f'Features {i+1}/{N} em {perf_counter()-t0:.1f}s',flush=True)
        X=np.vstack(outx)
    print(f'Características discursivas extraídas: {X.shape} em {perf_counter()-t0:.1f}s',flush=True)
    assert np.isfinite(X).all()
    args.out.mkdir(parents=True,exist_ok=True)
    if args.feature_cache is None: np.save(args.out/'features.npy',X)
    variants={'estrutural': (list(range(len(CORE_FEATURES))), 'structural'),
              'discurso': (list(range(len(CORE_FEATURES),len(FEATURE_NAMES))), 'discursive'),
              'completo': (list(range(len(FEATURE_NAMES))), 'completo'),
              'linear': (list(range(len(FEATURE_NAMES))), 'linear')}
    pred_by_variant={name:np.full(N,-1,dtype=np.int8) for name in variants}
    prob_main=np.full((N,3),np.nan,dtype=np.float32)
    folds_to_run=[args.only_fold] if args.only_fold is not None else list(range(5))
    per_fold=[]
    for fold in folds_to_run:
        tr=folds!=fold; va=folds==fold
        for name,(subset,kind) in variants.items():
            model=get_model(kind)
            model.fit(X[tr][:,subset],y[tr])
            pred=model.predict(X[va][:,subset]); pred_by_variant[name][va]=pred
            if name=='completo':prob_main[va]=model.predict_proba(X[va][:,subset])
            met=evaluate(pred,y[va]);per_fold.append({'fold':fold,'modelo':name,'n':int(va.sum()),**met})
            print(f'fold={fold} variante={name} accuracy={met["accuracy"]:.5f} macroF1={met["macro_f1"]:.5f}',flush=True)
    target=np.isin(folds,folds_to_run)
    records=[]
    for i in np.flatnonzero(target):
        records.append((i,int(folds[i]),int(y[i]),int(v3[i]),
                       *(int(pred_by_variant[name][i]) for name in variants),
                       *(f'{prob_main[i,j]:.7f}' for j in range(3))))
    args.out.mkdir(parents=True,exist_ok=True)
    save_csv(args.out/'oof_discursivo.csv',
             ['row_id','fold','y_true_id','pred_v3_id']+[f'pred_{name}_id' for name in variants]+['prob_c1','prob_c234','prob_c5'],records)
    metrics={'status':'CV COMPLETA' if args.only_fold is None else 'FOLD PILOTO',
             'n':int(target.sum()),'folds':folds_to_run,'features':FEATURE_NAMES,
             'referencia_v3':evaluate(v3[target],y[target]),
             'modelos':{name:evaluate(pred[target],y[target]) for name,pred in pred_by_variant.items()},
             'por_fold':per_fold,'tempo_segundos':round(perf_counter()-t0,2),
             'observacao':'Predicoes OOF, mas a seleção entre estas alternativas com os mesmos folds causa otimismo; sem teste1/teste2.'}
    preds=pred_by_variant['completo']; correction=np.logical_and(preds!=v3,target)
    metrics['comparacao_com_v3']={'mudancas':int(correction.sum()),
        'corrigidos':int(((y==preds)&(y!=v3)&target).sum()),
        'acertos_perdidos':int(((y==v3)&(y!=preds)&target).sum())}
    (args.out/'metricas.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=['# EP2 — Características discursivas e estruturais', '',
        'Validação: 5 folds originais agrupados por texto; sem teste1 nem teste2.',
        '', '| Modelo | Accuracy | Macro-F1 | Recall c1 | Recall c234 | Recall c5 |',
        '|---|---:|---:|---:|---:|---:|']
    for label,metric in [('Ensemble V3',metrics['referencia_v3'])]+[(v,metrics['modelos'][v]) for v in variants]:
        rows.append(f'| {label} | {metric["accuracy"]:.4%} | {metric["macro_f1"]:.4%} | '+
                    ' | '.join(f'{x:.2%}' for x in metric['recall'])+' |')
    rows+=['','## Ablação e limitações',
           '- Estrutural: tamanho, pontuação, formatação e densidade lexical.',
           '- Discurso: marcadores definidos previamente para justificativa, orientação, negativa, referência legal etc.',
           '- Completo: união das duas famílias (mesmo HistGradientBoosting).',
           '- Linear: mesmo conjunto completo com regressão logística; modelo interpretável de controle.',
           '- Classificadores apenas de características: não representam a integração neural por si só.',
           '- Qualquer escolha com base nesses folds requer confirmação independente; não usar os testes para afinar regras.',
           '', '## Comparação com V3',str(metrics['comparacao_com_v3'])]
    (args.out/'relatorio.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('Finalizado',args.out, 'em',round(perf_counter()-t0,1),'s',flush=True)


if __name__=='__main__':main()
