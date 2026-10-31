**UNIVERSIDADE DE SÃO PAULO**

**ACH2118 – Introdução ao Processamento de Língua Natural**

**RELATÓRIO – EP1**

_Classificação ternária de clareza em respostas e-SIC_

**Modelo final: Ensemble V3 (majority-5 + desempate seletivo com NorBERTo)**

**Integrantes: Arthur Yang Tung (14559819); Kevin Tamayose (14669711); Augusto Prado Yassuda (12543210); Filipe Valeriano Batista de Oliveira (14570701)**

**Repositório:** [**https://github.com/valiryo/Processamento-de-Linguagem-Natural**](https://github.com/valiryo/Processamento-de-Linguagem-Natural)

Outubro de 2026

**Estrutura do documento.** O relatório segue os nove itens solicitados no modelo oficial do EP1. Para registrar o trabalho experimental sem violar a orientação de apresentar, na seção de resultado, apenas o modelo final, os experimentos alternativos são documentados em apêndices.

# 1. Integrantes efetivos do grupo

| **Integrante**                       | **Número USP** |
| ------------------------------------ | -------------- |
| Arthur Yang Tung                     | 14559819       |
| Kevin Tamayose                       | 14669711       |
| Augusto Prado Yassuda                | 12543210       |
| Filipe Valeriano Batista de Oliveira | 14570701       |

# 2. Estratégia principal escolhida

**Tarefa.** A tarefa é uma classificação ternária da coluna resp_text na variável clarity, com classes c1, c234 e c5 (mapeamento interno 0, 1 e 2, respectivamente). A métrica principal adotada foi Accuracy; Macro-F1 foi acompanhado como métrica complementar, em especial para observar o comportamento entre classes.

**Estratégia final.** O melhor modelo validado foi um ensemble determinístico V3. Cinco classificadores geram um voto cada: (1) BERTimbau com contexto de 512 tokens; (2) BERTimbau 256 com soft labels para duplicatas conflitantes; (3) TF-IDF de palavras e caracteres com LinearSVC; (4) baseline TF-IDF com Regressão Logística; e (5) BERTimbau ordinal corrigido. O sexto modelo, NorBERTo-base, não participa como sexto voto: suas probabilidades são consultadas apenas em uma situação específica de empate 2-2-1 envolvendo a classe c234.

**Representações textuais.** O ensemble combina representações esparsas clássicas (TF-IDF de palavras e caracteres) com embeddings contextuais de Transformers pré-treinados para português. Essa diversidade foi mantida deliberadamente para combinar representações e padrões de erro distintos entre os componentes, inclusive em modelos que, isoladamente, apresentaram desempenho inferior aos melhores Transformers.

| **Componente**           | **Representação**                                        | **Classificador / papel**                                                             |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| BERTimbau 512            | Subwords / embeddings contextuais, até 512 tokens        | Transformer multiclass; 1º na hierarquia de desempate V2                              |
| BERTimbau soft-label 256 | Subwords / embeddings contextuais, até 256 tokens        | Transformer multiclass treinado com distribuições alvo em duplicatas conflitantes     |
| TF-IDF word+char         | Word n-grams (1,2) + char_wb n-grams (3,5)               | LinearSVC, C=0,5                                                                      |
| Baseline                 | TF-IDF de palavras (configuração padrão do scikit-learn) | LogisticRegression(class_weight="balanced")                                           |
| BERTimbau ordinal        | Subwords / embeddings contextuais, até 256 tokens        | Escore ordinal escalar + dois thresholds ordenados; previsão corrigida por thresholds |
| NorBERTo-base            | ModernBERT / embeddings contextuais, até 1024 tokens     | Somente desempate V3 em empates 2-2-1 com c234                                        |

# 3. Pré-processamento

Os textos foram mantidos o mais próximos possível do conteúdo original. Não foram aplicados stemming, lematização, remoção de stopwords nem normalizações linguísticas agressivas. A preparação dependeu da família de modelo:

- Validação básica: resp_text foi convertido para string e valores ausentes, quando tratados nos scripts, foram substituídos por string vazia. O conjunto efetivamente utilizado não apresentou textos faltantes.
- Modelos Transformer BERTimbau: tokenização pelo tokenizer de neuralmind/bert-base-portuguese-cased, com truncation=True. Como o modelo é cased, não foi aplicado lowercase manual.
- BERTimbau principal: padding fixo até max_length=512. As variantes 256/384 foram avaliadas em experimentos controlados de comprimento de contexto.
- NorBERTo: tokenização com max_length=1024, padding=False no tokenizer e padding dinâmico por lote via DataCollatorWithPadding(padding="longest").
- TF-IDF word: lowercase=True, sublinear_tf=True, n-grams de 1 a 2; TF-IDF char: analyzer="char_wb", n-grams de 3 a 5. O baseline, por exigência de fidelidade, manteve TfidfVectorizer() com parâmetros padrão.
- Duplicatas: a string exata de resp_text foi usada como chave de grupo. Esse agrupamento foi essencial para impedir que o mesmo texto aparecesse simultaneamente no treino e na validação.
- Soft labels: para textos repetidos com rótulos conflitantes, o alvo de cada ocorrência no treino foi a distribuição empírica das classes observadas para aquele texto no subconjunto de treinamento do fold. Na etapa final, treinada em 100% do train.xlsx, essa distribuição foi calculada sobre todo o treino.

# 4. Parâmetros avaliados na busca em grade e variações controladas

A busca principal de hiperparâmetros do BERTimbau foi realizada com max_length=256 e os mesmos cinco folds congelados. Foram avaliadas 3 taxas de aprendizado × 3 números de épocas × 5 folds, totalizando 45 execuções, todas concluídas. Após selecionar learning_rate e epochs, o comprimento de contexto foi avaliado separadamente para não multiplicar desnecessariamente o custo computacional.

| **Família / etapa**        | **Parâmetro**          | **Valores efetivamente avaliados** | **Observação**                                   |
| -------------------------- | ---------------------- | ---------------------------------- | ------------------------------------------------ |
| BERTimbau – grid principal | learning_rate          | {1e-5, 2e-5, 3e-5}                 | max_length fixo em 256                           |
| BERTimbau – grid principal | epochs                 | {1, 2, 3}                          | 3 × 3 × 5 = 45 runs                              |
| BERTimbau – contexto       | max_length             | {256, 384, 512}                    | Sweep controlado após o grid principal           |
| LinearSVC                  | C                      | {0,5; 1,0; 2,0}                    | Demais parâmetros do TF-IDF mantidos fixos       |
| Label smoothing            | label_smoothing_factor | {0,1}                              | Ablação única; não foi promovida ao modelo final |

**Observação metodológica.** Experimentos como soft labels, formulação ordinal, head+tail, SetFit, Albertina/DeBERTa, XLM-R e NorBERTo foram hipóteses/ablação ou candidatos complementares, e não devem ser interpretados como uma única busca em grade conjunta. Essa separação evita reportar como "grid" combinações que não foram de fato pesquisadas dessa forma.

# 5. Valores ótimos e configuração do modelo final

| **Componente**        | **Parâmetro**                     | **Valor usado no modelo final**                                 |
| --------------------- | --------------------------------- | --------------------------------------------------------------- |
| BERTimbau 512         | Modelo                            | neuralmind/bert-base-portuguese-cased                           |
| BERTimbau 512         | max_length                        | 512                                                             |
| BERTimbau 512         | learning_rate / epochs            | 3e-5 / 2                                                        |
| BERTimbau 512         | batch físico / grad. accumulation | 4 / 4 (batch efetivo 16)                                        |
| BERTimbau 512         | weight_decay / precisão           | 0,01 / FP16                                                     |
| Soft-label 256        | max_length / LR / epochs          | 256 / 3e-5 / 2                                                  |
| Soft-label 256        | batch físico / grad. accumulation | 8 / 2 (batch efetivo 16)                                        |
| LinearSVC             | C                                 | 0,5                                                             |
| LinearSVC             | word TF-IDF                       | ngram_range=(1,2), min_df=2, max_df=0,995, max_features=120000  |
| LinearSVC             | char TF-IDF                       | char_wb (3,5), min_df=2, max_features=180000                    |
| Baseline LR           | Modelo                            | TfidfVectorizer() + LogisticRegression(class_weight="balanced") |
| Ordinal 256           | max_length / LR / epochs          | 256 / 3e-5 / 2                                                  |
| Ordinal 256           | Cabeça                            | escore escalar + dois thresholds ordenados; BCEWithLogits       |
| NorBERTo              | Modelo                            | Itau-Unibanco/NorBERTo-base                                     |
| NorBERTo              | max_length / LR / epochs          | 1024 / 2e-5 / 2                                                 |
| NorBERTo              | batch físico / grad. accumulation | 2 / 8 (batch efetivo 16)                                        |
| NorBERTo              | weight_decay / precisão           | 0,01 / FP16; max_grad_norm=1,0                                  |
| Todos os Transformers | seed                              | 42                                                              |

**Regra do ensemble V2.** Conta-se o voto dos cinco classificadores. Havendo vencedor único, ele é escolhido. Em empate 2-2-1, apenas as duas classes líderes permanecem elegíveis; o desempate segue a hierarquia fixa BERT512 > soft-label 256 > LinearSVC > baseline LR > ordinal. A classe com apenas um voto nunca pode vencer o desempate.

**Extensão V3.** O resultado do V2 é preservado integralmente, exceto quando há empate 2-2-1 e c234 é uma das duas líderes. Nessa situação, as probabilidades do NorBERTo são comparadas somente entre as duas classes líderes, e a maior define a previsão final. Empates c1 versus c5 continuam usando o V2.

# 6. Procedimento experimental e reprodutibilidade

## 6.1 Dados e divisão

O conjunto de treino contém 20.092 exemplos. A distribuição é c1=6.347 (31,59%), c234=6.853 (34,11%) e c5=6.892 (34,30%). Para desenvolvimento e seleção, foram utilizados cinco folds fixos com StratifiedGroupKFold: a classe clarity foi estratificada e resp_text exato foi usado como grupo. Essa escolha evita vazamento causado por textos duplicados.

Os folds foram congelados e armazenados em data/splits/folds.csv. Quatro folds possuem 4.019 exemplos de validação e um possui 4.016; os respectivos treinos possuem 16.073 ou 16.076 exemplos. Foi verificado que nenhum texto exato aparece simultaneamente no treino e na validação de um fold.

## 6.2 Diagnóstico de duplicatas

Foram identificados 18.432 textos distintos entre as 20.092 linhas. Há 463 grupos de textos repetidos e 291 desses grupos apresentam rótulos conflitantes. Esse diagnóstico motivou tanto o agrupamento dos folds quanto o experimento de soft labels.

## 6.3 Protocolo de seleção

1. Criar e congelar os cinco folds, sem utilizar qualquer informação do conjunto final de teste.
2. Executar o baseline oficial TF-IDF + Regressão Logística.
3. Realizar o grid principal do BERTimbau em max_length=256 e selecionar LR=3e-5, epochs=2.
4. Avaliar o efeito do comprimento de contexto em 256, 384 e 512 tokens; selecionar 512 como melhor variante principal.
5. Testar hipóteses complementares: head+tail, label smoothing, duplicate-aware soft labels, formulação ordinal, LinearSVC word+char e outras famílias de Transformer/SetFit.
6. Gerar predições OOF alinhadas pelos row_id e construir ensembles sem usar o test set para qualquer escolha.
7. Congelar o Ensemble V3 após a validação OOF. Somente depois disso, treinar cada componente em 100% das 20.092 amostras e inferir as 900 linhas do test1.xlsx.

## 6.4 Controles contra vazamento e overfitting ao teste

- Folds nunca foram recriados após o início da comparação dos modelos.
- Textos duplicados foram agrupados no mesmo fold.
- A regra do ensemble, seus thresholds e sua hierarquia foram definidos exclusivamente a partir de OOF no treino.
- O test1.xlsx não foi utilizado para escolher modelo, hiperparâmetro, threshold, peso, componente ou regra de ensemble.
- Na fase final, as distribuições de previsões do teste foram tratadas apenas como diagnóstico operacional, nunca como critério de decisão.

# 7. Resultado – somente o modelo final

Conforme o modelo do professor, a tabela abaixo apresenta apenas o modelo final escolhido. A Accuracy é a métrica principal solicitada; Macro-F1 e F1 por classe são exibidos apenas como informação complementar de diagnóstico.

| **Modelo final** | **Accuracy OOF**       | **Macro-F1 OOF**       | **F1 c1** | **F1 c234** | **F1 c5** |
| ---------------- | ---------------------- | ---------------------- | --------- | ----------- | --------- |
| Ensemble V3      | 0,475661955 (47,5662%) | 0,473932944 (47,3933%) | ≈0,497544 | ≈0,388872   | ≈0,535383 |

| **Fold** | **Accuracy do Ensemble V3** |
| -------- | --------------------------- |
| 0        | 0,472754                    |
| 1        | 0,483702                    |
| 2        | 0,462304                    |
| 3        | 0,483815                    |
| 4        | 0,475740                    |

Na validação OOF, a regra V3 atuou em 873 empates-alvo, alterou 367 previsões, corrigiu 147 erros e quebrou 110 acertos, resultando em saldo líquido de +37 acertos. O V3 melhorou o V2 nos cinco folds.

**Inferência no conjunto final de teste.** Como os 900 exemplos do test1.xlsx não possuem rótulo verdadeiro, não há Accuracy de teste disponível antes da avaliação pelo professor. O modelo congelado produziu 268 previsões c1, 302 c234 e 330 c5. Essa distribuição não foi usada para modificar o sistema.

# 8. Repositório de código

<https://github.com/valiryo/Processamento-de-Linguagem-Natural>

# 9. Instruções para reprodução

## 9.1 Ambiente registrado

| **Item**            | **Versão / configuração registrada**                                  |
| ------------------- | --------------------------------------------------------------------- |
| Sistema de execução | Windows + PowerShell + ambiente virtual .venv                         |
| Python              | 3.11.9                                                                |
| scikit-learn        | 1.9.1                                                                 |
| PyTorch             | 2.14.0+cu130                                                          |
| Transformers        | 5.17.0                                                                |
| GPU utilizada       | NVIDIA RTX 3050 Laptop, 4 GB                                          |
| CUDA                | Disponível; FP16 utilizado nos Transformers                           |
| Outras dependências | pandas, numpy, scipy, openpyxl e dependências do Hugging Face/Trainer |

## 9.2 Estrutura esperada

data/raw/train.xlsx  
data/raw/test1.xlsx  
data/splits/folds.csv  
requirements.txt  
src/models/baseline.py  
src/experiments/\*.py  
results/

## 9.3 Passo a passo

1\. Na raiz do projeto, criar a venv se necessário com python -m venv .venv e ativá-la no PowerShell: .\\.venv\\Scripts\\Activate.ps1.

2\. Instalar ou atualizar as dependências do ambiente: python -m pip install -r requirements.txt.

3\. Garantir train.xlsx em data/raw/train.xlsx, test1.xlsx em data/raw/test1.xlsx e folds.csv em data/splits/folds.csv.

4\. Executar o baseline final: python src/experiments/baseline_final.py.

5\. Executar o LinearSVC final: python src/experiments/tfidf_word_char_linearsvc_final.py.

6\. Executar o BERTimbau 512 final: python src/experiments/bertimbau_512_final.py.

7\. Executar o BERTimbau soft-label final: python src/experiments/bertimbau_duplicate_soft_labels_final.py.

8\. Executar o BERTimbau ordinal final: python src/experiments/bertimbau_ordinal_256_final.py.

9\. Executar o NorBERTo final: python src/experiments/norberto_base_1024_final.py.

10\. Executar o ensemble congelado: python src/experiments/ensemble_v3_final.py.

11\. Usar a coluna y_pred do arquivo results/ensemble_majority_5_v3_norberto_final.csv para preencher, na mesma ordem das 900 linhas, a coluna clarity da planilha de teste entregue pelo professor. Não reordenar linhas.

## 9.4 Arquivos intermediários esperados

| **Componente**          | **Arquivo final**                                     |
| ----------------------- | ----------------------------------------------------- |
| Baseline                | results/baseline_final.csv                            |
| LinearSVC               | results/tfidf_word_char_linearsvc_final.csv           |
| BERTimbau 512           | results/bertimbau_512_final.csv                       |
| Soft-label 256          | results/bertimbau_duplicate_soft_labels_256_final.csv |
| Ordinal corrigido       | results/bertimbau_ordinal_256_final.csv               |
| NorBERTo                | results/norberto_base_1024_final.csv                  |
| Ensemble V2 (auditoria) | results/ensemble_majority_5_v2_final.csv              |
| Ensemble V3 final       | results/ensemble_majority_5_v3_norberto_final.csv     |

# Apêndice A – Análise do conjunto de dados

Esta seção registra diagnósticos utilizados para orientar as escolhas experimentais, sem alterar a definição da tarefa.

| **Estatística**                               | **Valor**                        |
| --------------------------------------------- | -------------------------------- |
| Instâncias de treino                          | 20.092                           |
| Textos distintos                              | 18.432                           |
| Ocorrências adicionais causadas por repetição | 1.660                            |
| Grupos de textos repetidos                    | 463                              |
| Grupos repetidos com rótulos conflitantes     | 291                              |
| Comprimento médio                             | ≈1.000 caracteres / 144 palavras |
| Mediana                                       | ≈709 caracteres / 101 palavras   |

## A.1 Cobertura de tokenização do BERTimbau

| **max_length** | **Textos que cabem** | **Textos truncados** | **Tokens removidos (registrado)** |
| -------------- | -------------------- | -------------------- | --------------------------------- |
| 128            | 7.409 (36,88%)       | 63,12%               | 2.962.261                         |
| 256            | 12.888 (64,14%)      | 35,86%               | 1.717.164                         |
| 384            | 16.287 (81,06%)      | 18,94%               | —                                 |
| 512            | 17.918 (89,18%)      | 10,82%               | 659.386                           |

| **Classe** | **Trunc. 128** | **Trunc. 256** | **Trunc. 384** | **Trunc. 512** |
| ---------- | -------------- | -------------- | -------------- | -------------- |
| c1         | 65,51%         | 39,01%         | 20,73%         | 12,26%         |
| c234       | 66,76%         | 40,04%         | 22,05%         | 12,48%         |
| c5         | 57,31%         | 28,79%         | 14,19%         | 7,85%          |

A análise mostrou que 256 tokens truncavam mais de um terço do corpus, enquanto 512 reduziam o truncamento para cerca de 10,8%. Esse diagnóstico motivou a avaliação controlada do comprimento de contexto; a seleção final foi feita pelas métricas OOF, não apenas pela cobertura.

# Apêndice B – Histórico experimental

Os resultados abaixo documentam as principais linhas exploradas. Eles não substituem a tabela oficial da seção 7, que contém somente o modelo final. Resultados com "±" correspondem à média e ao desvio-padrão entre os cinco folds; quando é apresentado um único valor para modelos com OOF, reporta-se a métrica calculada sobre as 20.092 previsões OOF concatenadas. Experimentos parciais indicam explicitamente o fold.

| **Experimento**                     | **Accuracy**         | **Macro-F1**         | **Situação / observação**                                                    |
| ----------------------------------- | -------------------- | -------------------- | ---------------------------------------------------------------------------- |
| Baseline TF-IDF + LR                | ≈0,4518              | ≈0,4496              | Baseline oficial; class_weight="balanced"                                    |
| BERTimbau 256                       | 0,4629 ± 0,0057      | 0,4582 ± 0,0072      | Vencedor do grid em LR=3e-5, epochs=2 antes do sweep de contexto             |
| BERTimbau 384                       | 0,4616 ± 0,0095      | 0,4571 ± 0,0126      | Não superou 256/512                                                          |
| BERTimbau 512                       | 0,464962 ± 0,006768  | 0,460948 ± 0,007814  | Melhor BERTimbau principal                                                   |
| Soft labels 256                     | 0,464116 ± 0,008919  | 0,460074 ± 0,010493  | Incluído no ensemble como variante de treinamento distinta                   |
| Head+Tail 256                       | 0,459437 ± 0,008725  | 0,455909 ± 0,009450  | 127 tokens iniciais + 127 finais; não promovido                              |
| Label smoothing 0,1                 | ≈0,4621 ± 0,0082     | ≈0,4574 ± 0,0115     | Não melhorou o conjunto final                                                |
| Ordinal corrigido                   | ≈0,4497              | ≈0,4505              | Desempenho standalone inferior; incluído para diversidade de decisão ordinal |
| TF-IDF word+char + LinearSVC        | 0,444107             | 0,443493             | C=0,5; incluído para diversidade de representação                            |
| NorBERTo 1024                       | ≈0,463568            | ≈0,463088            | Usado como desempate seletivo V3, não como sexto voto                        |
| SetFit / MPNet original             | ≈0,432561 ± 0,002606 | ≈0,429703 ± 0,002339 | CV completa; encerrado                                                       |
| SetFit corrigido – fold 0           | 0,426723             | 0,422749             | Apenas fold 0; execução muito mais lenta; não promovido                      |
| XLM-R 384 – fold 0                  | 0,448370             | 0,430404             | Resultado parcial registrado; não promovido                                  |
| Albertina/DeBERTa – fold 0, época 1 | ≈0,4563              | ≈0,4493              | Piloto; época 2 piorou; folds 1–4 não executados                             |
| Ensemble V2                         | 0,473820             | 0,472128             | Majority-5 com desempate restrito às classes líderes                         |
| Ensemble V3                         | 0,475661955          | 0,473932944          | Modelo final congelado                                                       |

## B.1 Head+Tail

O experimento head_tail_256_v1 manteve max_length=256, mas para textos com mais de 254 tokens selecionou 127 tokens iniciais e 127 finais, preservando \[CLS\] e \[SEP\]. Foram modificados 7.204 de 20.092 textos (35,85%), valor coerente com a taxa de truncamento em 256. O resultado não superou o BERTimbau padrão/512.

## B.2 Soft labels para duplicatas conflitantes

Em cada fold, a distribuição alvo de um texto repetido foi calculada exclusivamente com as ocorrências pertencentes ao treino daquele fold. Assim, por exemplo, se um texto aparecesse no treino com proporções 25% c1, 50% c234 e 25% c5, todas as suas ocorrências de treinamento recebiam o alvo \[0,25; 0,50; 0,25\]. A validação continuava utilizando o rótulo individual real. A loss foi a cross-entropy com distribuição alvo, calculada como -Σ target·log_softmax(logits), em float32.

## B.3 Formulação ordinal

A formulação ordinal codificou c1→\[0,0\], c234→\[1,0\] e c5→\[1,1\]. O modelo aprende um escore escalar s e dois thresholds ordenados t1&lt;t2, produzindo os logits s−t1 e s−t2; após a função sigmoide, eles representam respectivamente P(y&gt;c1) e P(y>c234), com loss BCEWithLogits. O script histórico inicialmente gravava y_pred por argmax das probabilidades de classe; esse procedimento foi identificado como incorreto para a decisão ordinal e o ensemble passou a reconstruir a classe pelo número de probabilidades cumulativas maiores que 0,5. Essa é a versão "ordinal corrigida" usada no sistema final.

## B.4 NorBERTo e correção do gradient accumulation

O NorBERTo exigiu um workaround específico com Transformers 5.x/ModernBERT: o Trainer podia inferir incorretamente que o modelo tratava a normalização da loss por num_items_in_batch. Foi utilizado FixedGASTrainer com model_accepts_loss_kwargs=False. Também se redefiniu a seed imediatamente antes da inicialização do modelo, garantindo inicialização reprodutível da cabeça classificadora. O scheduler foi linear, sem warmup, FP16 ligado, BF16 desligado, gradient checkpointing desligado e max_grad_norm=1,0.

# Apêndice C – Geração da previsão final

Depois de congelada a solução, cada componente foi treinado novamente usando 100% das 20.092 amostras de train.xlsx e aplicado às 900 linhas de test1.xlsx. Não houve reotimização de hiperparâmetros. Cada etapa gerou um CSV intermediário, permitindo auditoria e alinhamento por row_id.

| **Componente**    | **c1** | **c234** | **c5** | **Total** |
| ----------------- | ------ | -------- | ------ | --------- |
| Baseline LR       | 297    | 294      | 309    | 900       |
| LinearSVC         | 304    | 294      | 302    | 900       |
| BERTimbau 512     | 284    | 277      | 339    | 900       |
| Soft-label 256    | 294    | 269      | 337    | 900       |
| Ordinal corrigido | 209    | 394      | 297    | 900       |
| NorBERTo          | 275    | 306      | 319    | 900       |
| Ensemble V2       | 273    | 298      | 329    | 900       |
| Ensemble V3       | 268    | 302      | 330    | 900       |

No conjunto final, o V2 apresentou 56 empates 2-2-1. Desses, 41 envolviam c234 e eram elegíveis para a regra V3; 26 previsões foram efetivamente alteradas após comparar as probabilidades do NorBERTo apenas entre as duas classes líderes. A validação do arquivo final confirmou que nenhuma alteração ocorreu fora dos empates-alvo.

No ordinal final, os thresholds aprendidos foram aproximadamente t1=-0,500121 e t2=0,820322. A regra ordinal corrigida foi aplicada diretamente; no teste, ela diferiu do argmax antigo em 394 das 900 linhas, reforçando a importância da correção metodológica.
