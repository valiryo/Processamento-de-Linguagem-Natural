# EP1 de PLN — Resumo do trabalho e próximos passos

**Estado do projeto:** 30/09/2026
**Tarefa:** classificação ternária da clareza de respostas de e-SIC (`c1`, `c234`, `c5`)
**Métrica principal:** Accuracy (Macro-F1 como apoio)
**Base:** `progresso_atual.md` (29/09/2026), atualizado com os quatro experimentos posteriores (três hipóteses baratas e o funil do Transformer da seção 7.3).

---

## 1. Situação em uma frase

O **Ensemble V3 (47,5662% de Accuracy OOF)** continua sendo o melhor resultado oficial. Quatro novas abordagens foram testadas nesta rodada (embeddings congelados, atributos estilísticos + GBM, kNN e um Transformer jurídico fine-tunado) e **nenhuma agregou valor**; o último foi barrado no fold 0 pelo critério go/no-go. A recomendação é **congelar o V3** e partir para o pipeline final de treino em 100% dos dados.

---

## 2. Protocolo (inalterado)

- 20.092 instâncias; classes balanceadas (31,6% / 34,1% / 34,3%).
- Folds congelados em `data/splits/folds.csv` (`StratifiedGroupKFold`, grupo = `resp_text`). **Nunca recriar.**
- Seed 42; todo candidato salva OOF com `row_id`; comparações sempre por `row_id`.
- `test.xlsx` não é usado para desenvolvimento.
- Oracle é apenas diagnóstico, nunca resultado.

---

## 3. Resultados consolidados

| Experimento                       |                     Accuracy | Situação                          |
| --------------------------------- | ---------------------------: | --------------------------------- |
| Baseline TF-IDF + LR              |                       45,18% | Componente do ensemble            |
| BERTimbau 256 / 384 / 512         | 46,29% / 46,16% / **46,50%** | 512 é o melhor BERT individual    |
| Soft labels 256                   |                       46,41% | Componente do ensemble            |
| Ordinal corrigido                 |                       44,97% | Fraco isolado, útil no ensemble   |
| LinearSVC word+char               |                       44,41% | Fraco isolado, útil no ensemble   |
| NorBERTo 1024                     |                       46,36% | Desempate em c234 (V3)            |
| Head+Tail, Label smoothing        |              45,94% / 46,21% | Encerrados                        |
| Albertina, SetFit, XLM-R (fold 0) |         ~45% / 42,7% / 44,8% | Encerrados                        |
| Legal-BERTimbau 512 (fold 0)      |                       46,88% | NO-GO (saldo c234 −77); encerrado |
| Ensemble V2                       |                      47,382% | Superado                          |
| **Ensemble V3**                   |                 **47,5662%** | **Melhor oficial**                |

**Composição do V3:** votação majoritária de 5 modelos (BERT512, soft labels, LinearSVC, baseline LR, ordinal corrigido), com desempate 2-2-1 por hierarquia fixa (V2). Nos empates 2-2-1 em que `c234` é uma das líderes, usa-se a probabilidade do NorBERTo entre as duas líderes.

---

## 4. O que foi feito nesta rodada

Seguindo a seção 45 do relatório (representações diferentes e baratas), foram testadas três hipóteses baratas (4.1 a 4.3) e, em seguida, o funil de um novo Transformer fine-tunado (4.4), todas com o mesmo protocolo de folds e diagnóstico contra o V3.

| #   | Método                                                         |                       Accuracy OOF |       Macro-F1 | Saldo vs V3 em c234 |
| --- | -------------------------------------------------------------- | ---------------------------------: | -------------: | ------------------: |
| 1   | Embeddings congelados por chunks (multilingual-e5) + LR        | 42,40% (base) / **42,83%** (large) | 42,67% (large) |            negativo |
| 2   | 59 atributos estilísticos + SVD(TF-IDF) + HistGradientBoosting |                             44,04% |         43,81% |                −235 |
| 3   | kNN cosseno sobre TF-IDF (k=25, peso sim⁴)                     |                             42,82% |         42,55% |                −352 |

### 4.1 Embeddings congelados (e5)

- Texto inteiro dividido em chunks de 254 tokens (até 6); atributos = `[1º chunk ‖ média dos chunks]`.
- Trocar base por large rendeu só +0,4 p.p.
- A regra exploratória de desempate com este modelo **não melhorou** o V3.
- Interpretação: encoders de recuperação capturam o assunto do texto, não a clareza.

### 4.2 Atributos estilísticos + GBM

- Comprimento, legibilidade (Flesch-PT), citações legais, enumerações, URLs, marcadores de cortesia etc.
- Disagreement com o V3 de 34,6%, mas saldo negativo em **todas** as classes.
- Standalone equivalente ao LinearSVC.

### 4.3 kNN por memorização de quase-duplicatas

Hipótese: como os folds só separam textos _idênticos_, quase-duplicatas poderiam vazar sinal útil.

| Faixa de similaridade |     n |    kNN |     V3 |
| --------------------- | ----: | -----: | -----: |
| [0,95, 1,00)          | 3.313 | 47,93% | 48,02% |
| [0,80, 0,95)          | 2.448 | 44,20% | 47,10% |
| [0,50, 0,80)          | 5.163 | 42,46% | 47,38% |
| [0,00, 0,50)          | 9.168 | 40,80% | 47,63% |

- Mesmo na faixa mais similar o kNN só empata com o V3: **a hipótese está refutada**.
- Das 20.092 linhas, 2.123 (10,57%) estão em grupos de texto repetido; o **teto de acurácia nelas é 63,02%**, o que evidencia **ruído de anotação**.

### 4.4 Legal-BERTimbau 512 (funil da seção 7.3) — NO-GO no fold 0

- **Modelo:** `rufimelo/Legal-BERTimbau-base` (BERTimbau com pré-treino continuado em textos jurídicos em português).
- **Hipótese:** respostas de e-SIC citam leis, artigos e decretos; um encoder com domínio jurídico poderia errar de forma diferente do BERTimbau e do NorBERTo, sobretudo em `c234`.
- **Configuração:** idêntica à do BERTimbau 512 (LR 3e-5, 2 épocas, batch 4×4, FP16, seed 42), de modo que só o checkpoint muda. Script: `legal_bertimbau_funnel.py`.
- **Critério go/no-go, declarado antes do resultado:** Accuracy ≥ 0,450 **e** saldo em `c234` ≥ +1 contra o V3 (saldo = novo certo/V3 errado − V3 certo/novo errado).

| Critério              | Resultado | Status    |
| --------------------- | --------: | --------- |
| Accuracy no fold 0    |    0,4688 | ok        |
| Saldo em `c234` vs V3 |   **−77** | **falha** |
| **Veredito**          |           | **NO-GO** |

- Os folds 1–4 **não foram executados**, conforme o funil da seção 40.
- A Accuracy é competitiva (o V3 faz 0,4728 no fold 0), mas o modelo corrige menos erros do V3 em `c234` do que o V3 corrige dos dele. Isso o torna inútil como complemento, que é a justificativa do NorBERTo.
- Hipótese **não verificada** para o resultado: por ser inicializado a partir do BERTimbau, seus erros tendem a ser correlacionados com os do BERT512 já presente no ensemble.
- Este é um resultado de **fold 0 (piloto)**, não de CV completa.

---

## 5. Conclusões

1. **Diversidade de erros entre Transformers fine-tunados vale mais que features externas.** Os únicos ganhos reais (V2 → V3) vieram de modelos com erros complementares _e_ competitivos (NorBERTo).
2. **Modelos fracos de natureza diferente só adicionam ruído.** Disagreement alto não basta; o saldo (novo certo / V3 errado menos o inverso) precisa ser favorável, especialmente em `c234`.
3. **Há um teto imposto pelos dados.** Com 291 grupos de texto idêntico e rótulos conflitantes (63% de teto nos duplicados), 47–48% parece próximo do limite prático desta anotação.
4. **Risco de overfitting no OOF.** Muitos experimentos sobre o mesmo OOF tornam ganhos de décimos de ponto pouco confiáveis. O V3 permanece congelado.
5. **Accuracy individual competitiva não garante complementaridade.** O Legal-BERTimbau chegou a 46,88% no fold 0, mas teve saldo −77 em `c234`. O critério go/no-go pré-declarado evitou gastar horas de GPU nos folds 1–4.

---

## 6. Linhas encerradas (não reabrir sem evidência nova)

Head+Tail · Label smoothing · Albertina · SetFit · XLM-R · Legal-BERTimbau (NO-GO no fold 0) · Embeddings congelados (E5) · Atributos estilísticos + GBM · kNN TF-IDF · Grids de pesos/regras de ensemble.

---

## 7. Próximos passos

### 7.1 Prioridade alta — pipeline final (seção 47 do relatório)

1. **Congelar o V3** como solução final.
2. Escrever um script de produção que treina, em **100% de `train.xlsx`**, os seis componentes:
   - BERTimbau 512 (LR 3e-5, 2 épocas, batch efetivo 16)
   - BERTimbau 256 com soft labels (alvos calculados no treino completo)
   - TF-IDF word+char + LinearSVC (C=0,5)
   - Baseline TF-IDF + LR (`class_weight="balanced"`, configuração original)
   - BERTimbau ordinal corrigido (predição = nº de limiares com prob > 0,5)
   - NorBERTo 1024 (com `FixedGASTrainer`, FP16, seed reiniciada antes do modelo)
3. Gerar previsões para o teste e aplicar exatamente: Majority-5 V2 → regra V3 (empates 2-2-1 com `c234` entre as líderes → probabilidade do NorBERTo entre as duas líderes).
4. **Não recalibrar nada** com o teste.

**Cuidados de implementação**

- Sem folds: definir de antemão o número de épocas (as mesmas dos CVs), sem early stopping com o teste.
- Reutilizar `row_id`/ordem do teste para alinhar as previsões de cada componente.
- Salvar previsões e probabilidades por componente para auditoria.
- Custo estimado: várias horas de GPU (NorBERTo e BERT512 são os mais caros); rodar um componente por vez e salvar cada saída.

### 7.2 Prioridade média — relatório escrito

- Incluir os três resultados negativos desta rodada como **ablação** (tabela da seção 4).
- Destacar a análise de ruído (duplicatas conflitantes, teto de 63%) como limite do problema.
- Reportar Albertina, SetFit, XLM-R e Legal-BERTimbau como resultados de fold 0 apenas.
- Documentar o critério go/no-go pré-declarado (Accuracy ≥ 0,450 e saldo em `c234` ≥ +1) e o NO-GO do Legal-BERTimbau como exemplo de decisão sem ajuste posterior.
- Justificar o V3 pela complementaridade, com os números de saldo por classe.

### 7.3 Novo Transformer fine-tunado — executado e encerrado (NO-GO)

O funil da seção 40 (`tokenização → benchmark curto → fold 0 → complementaridade vs V3 → go/no-go → folds 1–4`) foi seguido com o Legal-BERTimbau e parou no fold 0: Accuracy 0,4688, mas saldo em `c234` de −77 (detalhes na seção 4.4). **Não executar os folds 1–4.**

Não há outro candidato Transformer planejado. Para tentar um, reutilize `legal_bertimbau_funnel.py` com `--model` e `--tag` novos e o mesmo critério pré-declarado, sem alterá-lo depois de ver o resultado. Dado o histórico (Albertina, XLM-R e agora o Legal-BERTimbau), o retorno esperado é baixo.

---

## 8. Checklist antes de encerrar

- [ ] V3 marcado como versão final (scripts e OOFs preservados)
- [ ] Script de treino em 100% dos dados escrito e testado em amostra pequena
- [ ] Previsões do teste geradas pelos 6 componentes
- [ ] Regra V2 + V3 aplicada sem ajustes
- [ ] Relatório com ablações e limitações (ruído de anotação)
- [ ] Resultados negativos documentados, nenhum omitido
