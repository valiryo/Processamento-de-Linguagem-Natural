# EP1 — Processamento de Linguagem Natural
## Estado documentado dos experimentos

**Data de consolidação:** 23/09/2026  
**Tarefa:** classificação ternária da clareza de respostas de e-SIC  
**Classes:** `c1`, `c234`, `c5`  
**Métrica principal:** Accuracy  
**Métricas diagnósticas:** Macro-F1, desvio entre folds, F1 por classe, matriz de confusão e OOF.

---

## 1. Protocolo experimental

O conjunto de treino possui **20.092 instâncias**.

Distribuição aproximada:

- `c1`: 6.347 — 31,59%
- `c234`: 6.853 — 34,11%
- `c5`: 6.892 — 34,30%

Existem **18.432 textos únicos**, **463 grupos de textos repetidos** e **291 grupos em que o mesmo texto aparece com rótulos diferentes**.

Por esse motivo, todos os experimentos utilizam os mesmos **5 folds congelados**, gerados originalmente com `StratifiedGroupKFold`, usando:

- estratificação: `clarity`
- agrupamento: `resp_text`
- `random_state=42`

Os folds ficam em:

```text
data/splits/folds.csv
```

Eles nunca devem ser recriados durante a comparação de modelos.

Tamanhos de validação:

| Fold | n_val |
|---:|---:|
| 0 | 4.019 |
| 1 | 4.019 |
| 2 | 4.019 |
| 3 | 4.016 |
| 4 | 4.019 |

Ainda não há uso do conjunto de teste final. Toda seleção de modelo, técnica e hiperparâmetro permanece restrita ao `train.xlsx` por cross-validation.

---

## 2. Ambiente

- Python: 3.11.9
- scikit-learn: 1.9.1
- PyTorch: 2.14.0+cu130
- Transformers: 5.17.0
- GPU: NVIDIA RTX 3050 Laptop, 4 GB
- CUDA: disponível

Modelo Transformer principal:

```text
neuralmind/bert-base-portuguese-cased
```

---

## 3. Quadro consolidado de resultados

| Método | Accuracy | Macro-F1 | Observação |
|---|---:|---:|---|
| Baseline oficial — TF-IDF + Logistic Regression | 0,4518 ± 0,0066 | 0,4496 ± 0,0069 | Referência oficial |
| BERTimbau 256 padrão | 0,4629 ± 0,0057 | 0,4582 ± 0,0072 | Melhor config. 256 |
| BERTimbau 384 | 0,4616 ± 0,0095 | 0,4571 ± 0,0126 | Pior e mais instável que 256 |
| BERTimbau 512 padrão | **0,4650 ± 0,0068** | **0,4609 ± 0,0078** | Melhor modelo individual convencional |
| BERTimbau 256 + duplicate-aware soft labels | 0,4641 ± 0,0089 | 0,4601 ± 0,0105 | Pequeno ganho sobre BERT256 |
| BERTimbau 256 ordinal corrigido | 0,4497 ± 0,0076 | 0,4505 ± 0,0089 | Melhora `c234`, piora Accuracy global |
| BERTimbau 256 head+tail | 0,4594 ± 0,0087 | 0,4559 ± 0,0095 | Não melhorou |
| BERTimbau 256 + label smoothing 0,1 | 0,4621 ± 0,0082 | 0,4574 ± 0,0115 | Sem sinal positivo |
| TF-IDF word+char + LinearSVC, C=0,5 | 0,4441 ± 0,0067 | 0,4434 ± 0,0070 | Fraco sozinho, útil pela diversidade |
| Ensemble majority-5 v1 | **0,4737 ± 0,0082** | **0,4715 ± 0,0101** | Melhor resultado confirmado até agora |
| Ensemble majority-5 v2* | **0,4738 ± 0,0078** | **0,4718 ± 0,0097** | Correção do desempate; recalculado a partir dos OOFs |

\* A versão v2 foi recalculada a partir dos mesmos OOFs já produzidos. O script foi corrigido para que, em empate `2-2-1`, o desempate ocorra apenas entre as classes empatadas na liderança. Falta apenas materializar localmente os CSVs da v2 rodando o script correspondente.

---

## 4. Baseline oficial

Modelo:

```text
TfidfVectorizer()
+
LogisticRegression(class_weight="balanced")
```

Resultados:

| Fold | Accuracy | Macro-F1 |
|---:|---:|---:|
| 0 | 0,447375 | 0,444703 |
| 1 | 0,458572 | 0,457162 |
| 2 | 0,443643 | 0,441190 |
| 3 | 0,450946 | 0,449395 |
| 4 | 0,458323 | 0,455594 |

Consolidado:

- Accuracy: **0,4518 ± 0,0066**
- Macro-F1: **0,4496 ± 0,0069**

OOF por classe:

- `c1`: F1 ≈ 0,4822
- `c234`: F1 ≈ 0,3688
- `c5`: F1 ≈ 0,4979

Desde o baseline já fica evidente que `c234` é a classe mais difícil.

---

## 5. Grid Search do BERTimbau 256

Grid realizado:

- learning rate: `1e-5`, `2e-5`, `3e-5`
- epochs: `1`, `2`, `3`
- 5 folds
- total: 45 treinamentos

Melhor configuração:

```text
max_length = 256
learning_rate = 3e-5
epochs = 2
train_batch_size = 8
eval_batch_size = 8
gradient_accumulation_steps = 2
effective_batch_size = 16
weight_decay = 0.01
FP16 = True
gradient_checkpointing = False
```

Resultado:

- Accuracy: **0,462922 ± 0,005664**
- Macro-F1: **0,458239 ± 0,007166**

Observação importante: para `2e-5` e `3e-5`, o desempenho tende a atingir o pico na segunda época; na terceira época há aumento de `eval_loss` e queda de desempenho, indicando início de overfitting.

---

## 6. Comprimento de contexto

Cobertura do BERTimbau:

| max_length | Textos completos | Textos truncados |
|---:|---:|---:|
| 128 | 36,88% | 63,12% |
| 256 | 64,14% | 35,86% |
| 384 | 81,06% | 18,94% |
| 512 | 89,18% | 10,82% |

Resultados:

### 384

- Accuracy: **0,461628 ± 0,009544**
- Macro-F1: **0,457098 ± 0,012563**

Não houve ganho em relação ao 256 e a variabilidade aumentou.

### 512

- Accuracy: **0,464962 ± 0,006768**
- Macro-F1: **0,460948 ± 0,007814**

O 512 é o melhor BERTimbau convencional atual, mas custa aproximadamente 2,36 vezes o tempo do 256.

Por isso, o 256 permaneceu como ambiente principal para testar novas hipóteses de forma barata.

---

## 7. Diagnóstico OOF do BERTimbau 512

OOF completo com 20.092 previsões:

- Accuracy: **0,465210**
- Macro-F1: **0,462150**

F1 por classe:

- `c1`: **0,484143**
- `c234`: **0,372892**
- `c5`: **0,529416**

Matriz de confusão:

```text
[[3076, 1792, 1479],
 [2050, 2388, 2415],
 [1234, 1775, 3883]]
```

O maior gargalo continua sendo `c234`.

Entre os erros do BERT512:

- aproximadamente 74,75% são erros entre classes adjacentes;
- aproximadamente 25,25% são erros extremos `c1 ↔ c5`.

Também foi observada baixa confiança geral:

- probabilidade máxima média nos acertos: ~0,547
- probabilidade máxima média nos erros: ~0,506

---

## 8. Duplicatas conflitantes

No OOF do BERT512:

### Textos únicos

- 17.969 ocorrências
- Accuracy ≈ 0,46608
- Macro-F1 ≈ 0,46216

### Duplicatas consistentes

- 441 ocorrências
- Accuracy ≈ 0,62132
- Macro-F1 ≈ 0,56673

### Duplicatas conflitantes

- 1.682 ocorrências
- Accuracy ≈ 0,41498
- Macro-F1 ≈ 0,37039

Há 291 grupos conflitantes.

Um oracle que sempre escolhesse o label majoritário de cada grupo conflitante teria Accuracy de apenas ~0,53329 dentro desses grupos. Estimou-se que aproximadamente 785 erros do dataset inteiro são inevitáveis se o classificador for determinístico e usar apenas `resp_text`.

As contradições são metodologicamente relevantes, mas não explicam o desempenho global relativamente baixo dos modelos.

---

## 9. Duplicate-aware soft labels

Hipótese:

quando um texto repetido possui rótulos contraditórios dentro do treino de um fold, substituir targets hard incompatíveis pela distribuição empírica das classes daquele grupo.

Exemplo:

```text
1 × c1
3 × c234
1 × c5
→ target = [0.2, 0.6, 0.2]
```

A distribuição é calculada exclusivamente no treino daquele fold.

Resultado:

- Accuracy CV: **0,464116 ± 0,008919**
- Macro-F1 CV: **0,460074 ± 0,010493**
- Accuracy OOF: **0,464115**
- Macro-F1 OOF: **0,460563**

F1 OOF:

- `c1`: 0,490034
- `c234`: 0,365269
- `c5`: 0,526388

O ganho global sobre BERT256 foi pequeno:

- Accuracy: ~+0,12 p.p.
- Macro-F1: ~+0,18 p.p.

Entretanto, o método possui valor metodológico importante porque trata diretamente os rótulos contraditórios sem utilizar qualquer informação da validação.

---

## 10. Classificação ordinal

Motivação:

as classes possuem ordem natural:

```text
c1 < c234 < c5
```

Representação usada:

```text
c1   -> [0, 0]
c234 -> [1, 0]
c5   -> [1, 1]
```

Foram utilizados dois thresholds:

```text
P(y > c1)
P(y > c234)
```

Houve um erro na avaliação original: a primeira versão aplicava uma regra incompatível com a formulação ordinal.

A decisão correta é:

```text
classe = número de thresholds com probabilidade > 0,5
```

Resultados corrigidos:

- Accuracy: **0,449683 ± 0,007613**
- Macro-F1: **0,450492 ± 0,008919**
- Accuracy OOF: **0,449681**
- Macro-F1 OOF: **0,451245**

F1 OOF:

- `c1`: 0,439033
- `c234`: **0,416212**
- `c5`: 0,498490

Apesar da pior Accuracy, foi o modelo individual que mais favoreceu `c234`. Por isso ele se mostrou útil como componente de ensemble.

---

## 11. Head+tail em max_length=256

Política testada:

para textos com mais de 254 tokens de conteúdo:

```text
127 primeiros tokens
+
127 últimos tokens
```

Textos menores permanecem intactos.

Foram modificados 7.204 textos, cerca de 35,85% do dataset.

Resultado:

- Accuracy: **0,459437 ± 0,008725**
- Macro-F1: **0,455909 ± 0,009450**

OOF:

- Accuracy: 0,459437
- Macro-F1: 0,456295

Conclusão: não melhorou.

O final do texto aparenta conter informação útil para `c1/c234` em algumas respostas longas, mas a substituição da região intermediária prejudicou especialmente `c5`.

---

## 12. Label smoothing global

Configuração:

```text
BERTimbau
max_length = 256
LR = 3e-5
epochs = 2
batch físico = 8
gradient accumulation = 2
weight_decay = 0.01
FP16 = True
label_smoothing_factor = 0.1
```

Resultados por fold:

| Fold | Accuracy | Macro-F1 |
|---:|---:|---:|
| 0 | 0,459816 | 0,448483 |
| 1 | 0,473999 | 0,472654 |
| 2 | 0,453347 | 0,446732 |
| 3 | 0,466384 | 0,466404 |
| 4 | 0,457079 | 0,452878 |

Consolidado:

- Accuracy: **0,462125 ± 0,008170**
- Macro-F1: **0,457430 ± 0,011484**

OOF:

- Accuracy: **0,462124**
- Macro-F1: **0,458065**

F1 OOF:

- `c1`: 0,488882
- `c234`: 0,360784
- `c5`: 0,524530

Conclusão:

`label_smoothing_factor=0.1` não mostrou sinal positivo em relação ao BERT256 padrão. Portanto não há justificativa atual para abrir um grid de fatores `0.05/0.10/0.15/0.20`.

---

## 13. TF-IDF word+char + LinearSVC

Representação:

- TF-IDF word `(1,2)`
- TF-IDF `char_wb` `(3,5)`
- concatenação das matrizes esparsas
- 120.000 features word
- 180.000 features char
- 300.000 features totais
- `LinearSVC(class_weight="balanced")`

C testados:

```text
0.5
1.0
2.0
```

Resultados:

| C | Accuracy | Macro-F1 |
|---:|---:|---:|
| **0,5** | **0,444107 ± 0,006665** | **0,443400 ± 0,007039** |
| 1,0 | 0,437140 ± 0,005577 | 0,436877 ± 0,006042 |
| 2,0 | 0,434950 ± 0,004545 | 0,434732 ± 0,004927 |

OOF do melhor `C=0.5`:

- Accuracy: **0,444107**
- Macro-F1: **0,443493**

F1:

- `c1`: 0,467777
- `c234`: **0,378468**
- `c5`: 0,484233

Embora seja pior como modelo isolado, o SVC se mostrou bastante diferente do BERT512:

- discordância BERT512 × SVC: ~41,73%
- casos em que BERT512 erra e SVC acerta: ~14,53% do dataset
- casos em que BERT512 acerta e SVC erra: ~16,64%

Por isso o modelo possui valor como fonte de diversidade.

---

## 14. Ensemble majority-5

Componentes:

1. BERTimbau 512
2. BERTimbau 256 + duplicate-aware soft labels
3. TF-IDF word+char + LinearSVC
4. baseline TF-IDF + Logistic Regression
5. BERTimbau ordinal corrigido

A vantagem do ensemble vem do fato de modelos individualmente mais fracos capturarem padrões diferentes dos encontrados pelo melhor BERT.

### Versão v1 — resultado executado localmente

OOF:

- Accuracy: **0,473721**
- Macro-F1: **0,471846**

F1:

- `c1`: **0,497110**
- `c234`: **0,384827**
- `c5`: **0,533600**

Resultados por fold:

| Fold | Accuracy | Macro-F1 |
|---:|---:|---:|
| 0 | 0,472257 | 0,466101 |
| 1 | 0,481961 | 0,480853 |
| 2 | 0,461060 | 0,457146 |
| 3 | 0,480080 | 0,480624 |
| 4 | 0,473252 | 0,472812 |

Média dos folds:

- Accuracy: **0,473722 ± 0,008231**
- Macro-F1: **0,471507 ± 0,010095**

Comparação com BERT512 OOF:

```text
BERT512:
Accuracy  = 0,465210
Macro-F1  = 0,462150

Ensemble v1:
Accuracy  = 0,473721
Macro-F1  = 0,471846
```

Ganho aproximado:

- Accuracy: **+0,851 p.p.**
- Macro-F1: **+0,970 p.p.**

O ganho ocorre também em `c234`, sem destruir o desempenho das classes extremas.

### Correção v2 do desempate

A primeira implementação possuía uma inconsistência no caso de votação `2-2-1`: usava a previsão do BERT512 como desempate mesmo quando a classe prevista por ele poderia ser a classe com apenas um voto.

A regra corrigida é:

1. identificar apenas as classes empatadas na maior contagem;
2. escolher entre elas usando uma prioridade fixa de modelos:
   `BERT512 → soft256 → SVC → baseline → ordinal`.

Recalculando a v2 diretamente sobre os OOFs existentes:

- Accuracy OOF: **0,473820**
- Macro-F1 OOF: **0,472128**

F1:

- `c1`: 0,495994
- `c234`: **0,386179**
- `c5`: 0,534211

Média entre folds:

- Accuracy: **0,473821 ± 0,007818**
- Macro-F1: **0,471779 ± 0,009710**

A diferença é pequena, mas a v2 deve ser considerada a regra metodologicamente correta.

---

## 15. Principais conclusões até agora

1. O BERTimbau supera o baseline oficial, mas a vantagem individual é modesta.

2. `c234` é o principal gargalo em praticamente todos os modelos.

3. Aumentar o contexto de 256 para 384 não ajudou. O 512 trouxe apenas um pequeno ganho, com custo computacional significativamente maior.

4. As duplicatas contraditórias devem ser tratadas metodologicamente, mas não explicam a baixa performance global.

5. Duplicate-aware soft labels produziram ganho pequeno, porém consistente com a natureza do problema e útil para o relatório.

6. A formulação ordinal não melhorou Accuracy, mas foi a melhor abordagem individual para `c234`, mostrando complementaridade.

7. Head+tail não melhorou o desempenho global.

8. Label smoothing global em 0,1 não trouxe ganho e não merece, neste momento, um grid adicional.

9. TF-IDF word+char + LinearSVC foi pior isoladamente, mas bastante diferente do BERT e útil para ensemble.

10. O maior avanço até agora veio de **diversidade entre modelos**, não de microajustes no mesmo BERT.

11. O ensemble elevou a Accuracy de ~46,52% do BERT512 OOF para aproximadamente **47,38%**, um ganho de cerca de **0,86 ponto percentual**.

---

## 16. Estado atual e direção experimental

### Melhor modelo individual

```text
BERTimbau 512
Accuracy OOF = 0,465210
Macro-F1 OOF = 0,462150
```

### Melhor solução atual

```text
Ensemble majority-5 v2
Accuracy OOF ≈ 0,473820
Macro-F1 OOF ≈ 0,472128
```

### Estratégia daqui em diante

O foco não deve mais ser pequenas alterações locais de loss ou truncamento no BERTimbau.

A prioridade passa a ser encontrar **novas fontes de diversidade com qualidade razoável**, principalmente:

- um segundo Transformer arquiteturalmente diferente;
- SetFit, assim que houver resultados comparáveis;
- eventualmente outros modelos distribuídos que tragam erros complementares.

Qualquer nova técnica deve continuar usando os mesmos folds congelados e produzir OOF completo para que sua complementaridade possa ser medida diretamente.

---

## 17. Arquivos principais existentes

```text
data/splits/folds.csv

results/baseline_oof.csv
results/bertimbau_512_oof.csv
results/bertimbau_duplicate_soft_labels_256_oof.csv
results/bertimbau_ordinal_256_oof.csv
results/bertimbau_head_tail_256_oof.csv
results/bertimbau_label_smoothing_256_oof.csv
results/tfidf_word_char_linearsvc_oof.csv

results/ensemble_majority_5_cv.csv
results/ensemble_majority_5_oof.csv
results/ensemble_majority_5_summary.csv
```

Script corrigido para a regra v2:

```text
src/experiments/ensemble_majority_5_oof_v2.py
```

---

## 18. Regra para o conjunto de teste final

Quando `test.xlsx` for disponibilizado:

1. não utilizar o teste para escolher modelo ou hiperparâmetros;
2. congelar a solução exclusivamente a partir dos resultados de CV;
3. treinar os componentes finais usando 100% do `train.xlsx`;
4. produzir as previsões do teste;
5. no caso de ensemble, reproduzir exatamente a regra escolhida durante a fase de CV;
6. não realizar ajustes posteriores com base no resultado do teste.

---

**Status atual:** o melhor caminho encontrado até aqui é a combinação de modelos complementares. O próximo experimento deve buscar uma nova arquitetura/representação capaz de adicionar diversidade real ao ensemble, em vez de apenas pequenas variações do BERTimbau já explorado.
