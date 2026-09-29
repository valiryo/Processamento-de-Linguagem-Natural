import pypandoc

content = r"""# EP1 de Processamento de Linguagem Natural
## Relatório técnico, resultados experimentais e guia para continuidade

**Estado do projeto:** 29/09/2026  
**Tarefa:** classificação ternária da clareza de respostas de e-SIC  
**Objetivo principal:** maximizar **Accuracy**, preservando rigor metodológico e evitando vazamento de informação.

---

# 1. Objetivo deste documento

Este documento consolida o trabalho experimental realizado até o momento no EP1 de Processamento de Linguagem Natural.

Ele possui três objetivos:

1. registrar os experimentos já executados e seus resultados;
2. documentar as decisões metodológicas tomadas;
3. permitir que outros integrantes do grupo executem novos experimentos de forma compatível com os resultados existentes.

Este documento deve ser tratado como o **estado oficial atual do projeto**.

Experimentos já concluídos não devem ser refeitos sem motivo concreto, os folds não devem ser recriados e nenhum resultado futuro do conjunto de teste deve ser utilizado para selecionar modelo, hiperparâmetro, ensemble ou regra de decisão.

---

# 2. Problema

A tarefa consiste em classificar respostas de pedidos de acesso à informação (`resp_text`) em três classes de clareza:

- `c1`
- `c234`
- `c5`

A variável alvo é `clarity`.

A métrica principal do trabalho é **Accuracy**.

As métricas secundárias utilizadas para diagnóstico são:

- Macro-F1;
- F1 por classe;
- matriz de confusão;
- média e desvio entre folds;
- previsões OOF;
- disagreement entre modelos;
- complementaridade;
- quantidade de casos em que modelo A acerta e modelo B erra;
- oracle, exclusivamente como diagnóstico.

O objetivo continua sendo **maximizar Accuracy mantendo metodologia experimental correta**.

---

# 3. Dataset

O conjunto de treinamento possui **20.092 instâncias**.

| Classe | Exemplos | Proporção |
|---|---:|---:|
| c1 | 6.347 | 31,59% |
| c234 | 6.853 | 34,11% |
| c5 | 6.892 | 34,30% |

Existem:

- 18.432 textos únicos;
- 463 grupos de textos repetidos;
- 291 grupos repetidos com labels diferentes;
- 1.660 ocorrências adicionais produzidas pelas duplicatas.

Portanto, o dataset contém **duplicatas conflitantes reais**. Essa característica foi determinante para o protocolo de validação.

---

# 4. Regra mais importante: folds congelados

O arquivo oficial é:

```text
data/splits/folds.csv
```

Foram construídos cinco folds usando `StratifiedGroupKFold`.

- Estratificação: `clarity`
- Grupo: `resp_text`

A finalidade do agrupamento é impedir que exatamente a mesma resposta apareça simultaneamente no treinamento e na validação, inclusive quando o mesmo texto possui labels conflitantes.

| Fold | Validação |
|---:|---:|
| 0 | 4.019 |
| 1 | 4.019 |
| 2 | 4.019 |
| 3 | 4.016 |
| 4 | 4.019 |

Já foi verificado que nenhum `resp_text` aparece simultaneamente em treino e validação.

## REGRA ABSOLUTA

**NUNCA recriar os folds.**

Todo novo experimento deve utilizar exatamente `data/splits/folds.csv`. Caso contrário, seus resultados não são diretamente comparáveis aos resultados deste relatório.

---

# 5. Ambiente utilizado

```text
Windows
Python 3.11.9
scikit-learn 1.9.1
PyTorch 2.14.0+cu130
Transformers 5.17.0
CUDA = True
```

GPU:

```text
NVIDIA GeForce RTX 3050 Laptop
4 GB VRAM
```

Raiz original:

```text
C:\Users\valer\Projects\PLN\Processamento-de-Linguagem-Natural
```

Seed principal:

```text
42
```

---

# 6. Estrutura esperada do projeto

```text
Processamento-de-Linguagem-Natural/
│
├── data/
│   ├── raw/
│   │   └── train.xlsx
│   └── splits/
│       └── folds.csv
│
├── results/
├── src/
│   ├── evaluation/
│   └── experiments/
└── ...
```

Novos experimentos devem possuir **scripts separados**. Não modificar scripts de experimentos antigos de maneira que impeça a reprodução do resultado histórico.

---

# 7. Como executar os experimentos

## 7.1 Ativar o ambiente

No Windows, a partir da raiz:

```bash
.venv\Scripts\activate
```

Verificar:

```bash
python --version
```

Esperado:

```text
Python 3.11.9
```

Verificar CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

No ambiente original, CUDA deve estar disponível e a GPU é uma NVIDIA GeForce RTX 3050 Laptop GPU.

---

# 8. Regra para novos scripts

Cada experimento deve ter seu próprio script, por exemplo:

```text
src/experiments/meu_novo_modelo.py
```

O script deve carregar:

```text
data/raw/train.xlsx
data/splits/folds.csv
```

Nunca gerar novos folds.

Normalização mínima recomendada:

```python
df = pd.read_excel(DATA_PATH)
folds = pd.read_csv(FOLDS_PATH)

assert len(df) == len(folds)

df["resp_text"] = df["resp_text"].fillna("").astype(str)
df["clarity"] = df["clarity"].astype(str)
```

A conversão explícita de `resp_text` para string é importante porque valores provenientes do Excel podem ser interpretados como tipos numéricos. Sem isso, `datasets/pyarrow` pode gerar `ArrowTypeError`.

---

# 9. Convenção de labels

A ordem utilizada nos experimentos é:

```text
0 -> c1
1 -> c234
2 -> c5
```

Antes de combinar OOFs, confirmar sempre que a codificação utilizada pelo novo modelo corresponde a essa ordem.

---

# 10. OOF: requisito obrigatório

Todo experimento que possa participar de comparação ou ensemble deve produzir previsões **Out-of-Fold**.

Formato recomendado:

```text
row_id
fold
y_true_id
y_pred_id
y_true
y_pred
prob_c1
prob_c234
prob_c5
correct
resp_text
```

O campo mais importante para combinação entre experimentos é `row_id`.

Nunca alinhar OOFs simplesmente pela posição atual das linhas. Fazer merge explicitamente por `row_id`.

---

# 11. Baseline oficial

Modelo:

```text
TF-IDF + Logistic Regression
class_weight="balanced"
```

Resultado aproximado:

```text
Accuracy   ≈ 0,4518
Macro-F1   ≈ 0,4496
```

O experimento original apresentou `ConvergenceWarning` com `lbfgs` e 100 iterações, mas permanece como baseline oficial.

Arquivos:

```text
results/baseline_oof.csv
results/baseline_folds.csv
```

---

# 12. BERTimbau — tokenização

Modelo:

```text
neuralmind/bert-base-portuguese-cased
```

| max_length | Truncados |
|---:|---:|
| 128 | 63,12% |
| 256 | 35,86% |
| 384 | 18,94% |
| 512 | 10,82% |

Também foi observado que `c1` e `c234` sofrem mais truncamento que `c5`.

---

# 13. BERTimbau 256

```text
Accuracy:
0,462922 ± 0,005664

Macro-F1:
0,458239 ± 0,007166
```

Foi a primeira melhoria neural consistente sobre o baseline.

---

# 14. BERTimbau 384

```text
Accuracy:
0,461628 ± 0,009544

Macro-F1:
0,457098 ± 0,012563
```

Conclusão: aumentar o contexto de 256 para 384 não melhorou o resultado.

---

# 15. BERTimbau 512

CV:

```text
Accuracy:
0,464962 ± 0,006768

Macro-F1:
0,460948 ± 0,007814
```

OOF aproximado:

```text
Accuracy:
0,465210

Macro-F1:
0,462150
```

F1:

```text
c1    ≈ 0,484143
c234  ≈ 0,372892
c5    ≈ 0,529416
```

É o melhor BERTimbau standalone principal.

Arquivo:

```text
results/bertimbau_512_oof.csv
```

---

# 16. Duplicate-aware soft labels

BERTimbau 256 utilizando a distribuição de labels dos grupos duplicados conflitantes.

```text
Accuracy:
0,464116 ± 0,008919

Macro-F1:
0,460074 ± 0,010493
```

A melhoria standalone é pequena, mas a abordagem é metodologicamente relevante e o modelo foi mantido como componente do ensemble.

Arquivos:

```text
results/bertimbau_duplicate_soft_labels_256_cv.csv
results/bertimbau_duplicate_soft_labels_256_history.csv
results/bertimbau_duplicate_soft_labels_256_oof.csv
```

---

# 17. BERTimbau ordinal

```text
Accuracy:
≈ 0,4497

Macro-F1:
≈ 0,4505
```

O standalone é fraco, mas apresentou complementaridade, especialmente em `c234`.

**Atenção:** existiu uma versão antiga com reconstrução incorreta de `y_pred`. Utilizar somente a reconstrução ordinal corrigida.

Arquivos:

```text
results/bertimbau_ordinal_256_cv.csv
results/bertimbau_ordinal_256_oof.csv
```

---

# 18. Head + Tail

Estratégia:

```text
127 tokens iniciais + 127 tokens finais
```

para textos acima do limite.

Textos modificados:

```text
7.204 / 20.092
≈ 35,85%
```

Resultado:

```text
Accuracy:
0,459437 ± 0,008725

Macro-F1:
0,455909 ± 0,009450
```

**Linha encerrada.**

---

# 19. Label smoothing

```text
label_smoothing_factor = 0.1

Accuracy ≈ 0,4621
Macro-F1 ≈ 0,4574
```

Não houve ganho relevante.

**Não abrir grid de label smoothing. Linha encerrada.**

---

# 20. TF-IDF word + char + LinearSVC

Representação:

```text
word TF-IDF: ngram_range=(1,2)
char_wb: ngram_range=(3,5)
LinearSVC C=0.5
```

Resultado:

```text
Accuracy:
0,444107

Macro-F1:
0,443493

c1    = 0,467777
c234  = 0,378468
c5    = 0,484233
```

Standalone é fraco, porém BERT512 e LinearSVC discordam em aproximadamente **41,7%** dos exemplos. Isso mostrou que um modelo não precisa ser o melhor standalone para ser útil.

Arquivos:

```text
results/tfidf_word_char_linearsvc_cv.csv
results/tfidf_word_char_linearsvc_oof.csv
results/tfidf_word_char_linearsvc_summary.csv
```

---

# 21. Ensemble Majority-5

Componentes:

```text
1. BERTimbau 512
2. BERTimbau 256 + duplicate-aware soft labels
3. TF-IDF word+char + LinearSVC
4. baseline TF-IDF + Logistic Regression
5. BERTimbau ordinal corrigido
```

---

# 22. Ensemble V2

Contar os cinco votos.

Se houver vencedor único, utilizar o vencedor.

Em empate `2-2-1`, considerar somente as duas classes empatadas na liderança.

Hierarquia:

```text
BERT512
>
soft labels
>
LinearSVC
>
baseline
>
ordinal
```

Resultado:

```text
Accuracy:
0,4738204260

Macro-F1:
0,4721275773

F1 c1:
0,495994

F1 c234:
0,386179

F1 c5:
0,534211
```

CV Accuracy:

```text
≈ 0,473821 ± 0,007818
```

---

# 23. NorBERTo

Modelo:

```text
Itau-Unibanco/NorBERTo-base
```

Foi escolhido por arquitetura, tokenizer, pré-treino e contexto diferentes do BERTimbau.

Truncamento aproximado:

| max_length | Truncados |
|---:|---:|
| 256 | 37,00% |
| 512 | 11,55% |
| 1024 | 1,99% |
| 2048 | 0,085% |

Escolha:

```text
max_length = 1024
```

---

# 24. Gradient Accumulation no NorBERTo

Foi identificado um problema entre `Trainer` e `ModernBertForSequenceClassification`.

Correção:

```python
class FixedGASTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
```

Também:

```text
max_grad_norm = 1.0
```

Resetar a seed imediatamente antes de inicializar o modelo.

---

# 25. Configuração final NorBERTo

```text
max_length = 1024
learning_rate = 2e-5
epochs = 2
physical_batch = 2
gradient_accumulation = 8
effective_batch = 16
weight_decay = 0.01
FP16 = ON
gradient_checkpointing = OFF
seed = 42
```

---

# 26. Resultado NorBERTo

OOF:

```text
Accuracy:
0,463568

Macro-F1:
0,463088

c1    ≈ 0,479132
c234  ≈ 0,396618
c5    ≈ 0,513515
```

CV:

```text
Accuracy:
≈ 0,463567 ± 0,006688

Macro-F1:
≈ 0,462513 ± 0,005413
```

| Fold | Accuracy | Macro-F1 |
|---:|---:|---:|
| 0 | 0,475491 | 0,470850 |
| 1 | 0,461060 | 0,461768 |
| 2 | 0,455088 | 0,453814 |
| 3 | 0,461902 | 0,463332 |
| 4 | 0,464295 | 0,462801 |

---

# 27. NorBERTo × Ensemble V2

```text
Disagreement:
5.231 / 20.092
≈ 26,04%

NorBERTo correto / ensemble errado:
1.863

Ensemble correto / NorBERTo errado:
2.069

Oracle diagnóstico:
≈ 56,65%
```

Para `true c234`:

```text
NorBERTo correto / ensemble errado:
903

Ensemble correto / NorBERTo errado:
766
```

Isso mostrou utilidade especial do NorBERTo em `c234`.

---

# 28. Ensemble V3 — melhor resultado oficial

Nos empates `2-2-1`:

```text
c1 vs c234: NorBERTo ajudava
c234 vs c5: NorBERTo ajudava
c1 vs c5: NorBERTo piorava
```

## Regra V3

Preservar integralmente o V2.

Única mudança: quando houver empate `2-2-1` e `c234` for uma das duas classes líderes, utilizar as probabilidades do NorBERTo **somente entre as duas classes líderes**.

Se `c234` não participa, manter o V2.

```text
Empates alvo: 873
Previsões alteradas: 367
Erros corrigidos: 147
Acertos quebrados: 110
Saldo: +37 acertos
```

OOF:

```text
Accuracy:
0,475661955
= 47,5662%

Macro-F1:
0,473932944
= 47,3933%

F1 c1:
0,497544

F1 c234:
0,388872

F1 c5:
0,535383
```

| Fold | Accuracy |
|---:|---:|
| 0 | 0,472754 |
| 1 | 0,483702 |
| 2 | 0,462304 |
| 3 | 0,483815 |
| 4 | 0,475740 |

O V3 melhora o V2 nos cinco folds.

**MELHOR RESULTADO OFICIAL ATUAL: 47,5662% Accuracy OOF.**

---

# 29. Albertina

Modelo:

```text
PORTULAN/albertina-100m-portuguese-ptbr-encoder
```

Arquitetura DeBERTa.

FP16 falhou com overflow. A solução foi carregar explicitamente com `dtype=torch.float32`.

Configuração:

```text
max_length = 512
LR = 2e-5
epochs = 2
physical_batch = 1
gradient_accumulation = 16
effective_batch = 16
FP32 obrigatório
```

Apenas fold 0.

Época 1:

```text
Accuracy ≈ 0,4563
Macro-F1 ≈ 0,4493
eval_loss ≈ 1,0379
```

Época 2:

```text
Accuracy ≈ 0,4499
Macro-F1 ≈ 0,4373
eval_loss ≈ 1,0413
```

F1 final aproximado:

```text
c1    = 0,4659
c234  = 0,3148
c5    = 0,5311
```

Tempo: aproximadamente **6h48**.

Contra V3 no fold 0:

```text
disagreement ≈ 27,20%
Albertina correta / V3 errado: 380
V3 correto / Albertina errada: 472
oracle ≈ 56,73%
```

**Albertina encerrada. Não rodar folds 1–4.**

---

# 30. SetFit

Modelo:

```text
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Melhor configuração original:

```text
max_length = 256
LR = 1e-5
epochs = 1
batch = 16
num_iterations = 1
```

CV original:

```text
Accuracy ≈ 0,432561 ± 0,002606
Macro-F1 ≈ 0,429703 ± 0,002339
```

O script original não salvava OOF.

Tentativas com oversampling e `num_iterations=5` foram abandonadas por custo computacional excessivo.

SetFit corrigido, fold 0:

```text
Accuracy:
0,4267230654

Macro-F1:
0,4227486074

c1    ≈ 0,448713
c234  ≈ 0,326026
c5    ≈ 0,493506

Tempo:
≈ 226,99 min
≈ 3h47
```

Contra V3:

```text
Disagreement:
1.623 / 4.019
≈ 40,38%

Ambos corretos: 1.195
SetFit correto / V3 errado: 520
V3 correto / SetFit errado: 705
Ambos errados: 1.599

Oracle:
≈ 60,21%
```

Regras simples de uso do SetFit não produziram ganho confiável.

**SetFit encerrado como candidato ao ensemble oficial.**

---

# 31. XLM-RoBERTa

Modelo:

```text
FacebookAI/xlm-roberta-base
```

## Tokenização

| max_length | Cabem inteiros | Truncados | Tokens descartados |
|---:|---:|---:|---:|
| 256 | 64,61% | 35,39% | 32,96% |
| 384 | 81,26% | 18,74% | 19,48% |
| 512 | 89,23% | 10,77% | 12,18% |

Em 384:

```text
c1 truncado: 20,47%
c234 truncado: 21,77%
c5 truncado: 14,15%
```

Escolhido `max_length=384`.

Arquivos:

```text
results/xlmr_tokenization_summary.csv
results/xlmr_tokenization_by_class.csv
```

---

# 32. XLM-R — benchmark

Configuração:

```text
model = FacebookAI/xlm-roberta-base
max_length = 384
learning_rate = 2e-5
weight_decay = 0.01
physical_batch = 4
gradient_accumulation = 4
effective_batch = 16
FP16 = ON
gradient_checkpointing = OFF
max_steps = 100
seed = 42
```

Resultado:

```text
Tempo total: 271,78 s
Tempo / step: 2,718 s
Peak VRAM allocated: 5,001 GB
Peak VRAM reserved: 5,859 GB
```

---

# 33. XLM-R — fold 0

Configuração registrada:

```text
model = FacebookAI/xlm-roberta-base
fold = 0
max_length = 384
learning_rate = 2e-5
epochs = 2
physical_batch = 4
gradient_accumulation = 4
effective_batch = 16
```

Resultado:

```text
Accuracy:
0,448370

Macro-F1:
0,430404

runtime:
≈ 109,56 minutos
```

OOF: 4.019 exemplos.

Arquivos:

```text
xlmr_384_fold0_summary.csv
xlmr_384_fold0_oof.csv
```

---

# 34. XLM-R × Ensemble V3

Os 4.019 exemplos foram alinhados por `row_id`, com `y_true` coincidente.

| Modelo | Accuracy |
|---|---:|
| XLM-R 384 | 44,837% |
| Ensemble V3 | 47,275% |

```text
Disagreement:
1.265 / 4.019
≈ 31,48%

Ambos corretos: 1.371
XLM-R correto / V3 errado: 431
V3 correto / XLM-R errado: 529
Ambos errados: 1.688

Oracle:
≈ 57,9995%
```

| Classe real | XLM-R corrige V3 | V3 corrige XLM-R | Saldo XLM-R |
|---|---:|---:|---:|
| c1 | 117 | 155 | -38 |
| c234 | 127 | 268 | **-141** |
| c5 | 187 | 106 | **+81** |

O XLM-R apresenta alguma especialização em `c5`, mas é especialmente desfavorável em `c234`.

**XLM-R encerrado após fold 0. Não executar folds 1–4 sem nova evidência forte.**

---

# 35. Estado oficial atual

Melhor solução:

## ENSEMBLE V3

```text
OOF Accuracy:
47,5662%

OOF Macro-F1:
47,3933%
```

Componentes:

```text
BERTimbau 512
BERTimbau duplicate-aware soft labels
TF-IDF word+char + LinearSVC
baseline TF-IDF + Logistic Regression
BERTimbau ordinal corrigido
```

mais NorBERTo exclusivamente como desempate probabilístico nos empates `2-2-1` envolvendo `c234`.

Linhas encerradas:

```text
Head + Tail
Label smoothing
Albertina
SetFit como candidato oficial
XLM-R
```

---

# 36. Principal gargalo: c234

A classe `c234` é consistentemente a mais difícil.

Exemplos:

```text
BERT512 F1 c234 ≈ 0,3729
NorBERTo F1 c234 ≈ 0,3966
SetFit F1 c234 ≈ 0,3260
Albertina F1 c234 ≈ 0,3148
```

O NorBERTo foi particularmente útil porque seus erros diferentes ajudaram justamente nessa classe.

---

# 37. Principal aprendizado

O maior ganho do projeto não veio de aumentar contexto ou microajustar hiperparâmetros de um Transformer.

Veio da **complementaridade**.

Modelos relativamente fracos standalone, como LinearSVC e o ordinal, permaneceram úteis porque seus erros são diferentes.

O NorBERTo é o exemplo principal: não dominou o BERT512 standalone, mas sua complementaridade em `c234` permitiu elevar o Ensemble V2 de aproximadamente 47,3820% para o V3 com **47,5662%**.

---

# 38. Cuidado com overfitting do ensemble

Não:

```text
fazer grids grandes de pesos;
testar dezenas de thresholds;
testar dezenas de regras condicionais;
selecionar regras porque melhoram 2 ou 3 exemplos;
usar oracle como resultado;
micro-otimizar indefinidamente o mesmo OOF.
```

O V3 permanece congelado como benchmark enquanto não surgir evidência forte de um novo modelo.

---

# 39. Como avaliar um novo modelo

Fazer merge por `row_id`:

```python
merged = novo.merge(
    v3,
    on="row_id",
    suffixes=("_novo", "_v3"),
    validate="one_to_one",
)

assert (
    merged["y_true_id_novo"]
    == merged["y_true_id_v3"]
).all()
```

Calcular:

```text
Accuracy
Macro-F1
F1 c1
F1 c234
F1 c5
disagreement
novo correto / V3 errado
V3 correto / novo errado
análise por classe
oracle
```

Oracle é apenas diagnóstico e nunca deve ser reportado como performance realizável.

---

# 40. Funil para experimentos caros

Não executar automaticamente cinco folds de um Transformer novo.

```text
1. análise de tokenização
        ↓
2. benchmark curto
        ↓
3. fold 0
        ↓
4. análise de complementaridade
        ↓
5. decisão go/no-go
        ↓
6. somente então folds 1–4
```

Continuar apenas com sinal forte de:

- Accuracy competitiva; ou
- forte complementaridade com V3; ou
- saldo favorável modelo correto/V3 errado; ou
- melhoria especialmente interessante em `c234`.

---

# 41. Stacking — cuidado metodológico

Não:

```text
pegar OOF dos modelos
→ treinar meta-modelo nesses OOFs
→ avaliar o meta-modelo nesses mesmos OOFs
```

Isso é inválido.

Uma avaliação rigorosa pode exigir nested CV e/ou retreinamento dos modelos base. Não promover stacking como resultado científico sem resolver explicitamente o problema de leakage.

---

# 42. Arquivos importantes

## Folds

```text
data/splits/folds.csv
```

## Baseline

```text
results/baseline_oof.csv
results/baseline_folds.csv
```

## BERT512

```text
results/bertimbau_512_oof.csv
```

## Soft labels

```text
results/bertimbau_duplicate_soft_labels_256_cv.csv
results/bertimbau_duplicate_soft_labels_256_history.csv
results/bertimbau_duplicate_soft_labels_256_oof.csv
```

## Ordinal

```text
results/bertimbau_ordinal_256_cv.csv
results/bertimbau_ordinal_256_oof.csv
```

## LinearSVC

```text
results/tfidf_word_char_linearsvc_cv.csv
results/tfidf_word_char_linearsvc_oof.csv
results/tfidf_word_char_linearsvc_summary.csv
```

## Ensemble V2

```text
results/ensemble_majority_5_v2_cv.csv
results/ensemble_majority_5_v2_oof.csv
results/ensemble_majority_5_v2_summary.csv
```

## Ensemble V3

Usar os arquivos finais de `ensemble_majority_5_v3_norberto`.

## NorBERTo

Usar os arquivos finais de CV, history, OOF e summary.

## Albertina

```text
results/albertina_512_cv.csv
results/albertina_512_history.csv
results/albertina_512_oof.csv
results/albertina_512_summary.csv
```

## SetFit

```text
results/setfit_corrected_cv.csv
results/setfit_corrected_experiments.csv
results/setfit_corrected_oof.csv
```

## XLM-R

```text
results/xlmr_tokenization_summary.csv
results/xlmr_tokenization_by_class.csv
xlmr_384_fold0_summary.csv
xlmr_384_fold0_oof.csv
```

---

# 43. Regras obrigatórias para novos experimentos

1. Usar exclusivamente `data/splits/folds.csv`.
2. Não recriar folds.
3. Não usar `test.xlsx` para desenvolvimento.
4. Todo candidato sério deve salvar OOF.
5. Comparações devem ser alinhadas por `row_id`.
6. Comparações científicas devem usar os mesmos folds.
7. Resultado de um fold deve ser explicitamente identificado como piloto/fold 0.
8. Oracle é apenas diagnóstico.
9. Utilizar seed 42.
10. Em Transformers, resetar a seed imediatamente antes da inicialização do modelo.
11. Registrar modelo, `max_length`, LR, epochs, batch, gradient accumulation, precisão, seed, fold e runtime.
12. Não esconder resultados negativos.

---

# 44. Formato recomendado para novos resultados

```text
results/<experimento>_cv.csv
results/<experimento>_history.csv
results/<experimento>_oof.csv
results/<experimento>_summary.csv
```

`summary.csv`:

```text
model
fold
max_length
learning_rate
epochs
physical_batch
gradient_accumulation
effective_batch
accuracy
macro_f1
runtime_minutes
```

`oof.csv`:

```text
row_id
fold
y_true_id
y_pred_id
y_true
y_pred
prob_c1
prob_c234
prob_c5
correct
resp_text
```

---

# 45. O que ainda pode ser testado

Já foram explorados:

```text
BERTimbau
NorBERTo
Albertina
XLM-R
SetFit/MPNet
TF-IDF + LogisticRegression
TF-IDF word+char + LinearSVC
modelo ordinal
soft labels
head+tail
label smoothing
ensembles
```

Uma linha ainda razoável, de baixo custo, é:

```text
embeddings congelados
+
classificador linear
```

Desenho:

```text
SentenceTransformer / encoder
        ↓
calcular embeddings uma única vez
        ↓
salvar embeddings
        ↓
5 folds congelados
        ↓
LogisticRegression ou LinearSVC
        ↓
OOF
        ↓
comparar complementaridade com V3
```

O objetivo não é repetir SetFit. O encoder permanece congelado e não deve ser aberto um grid grande.

---

# 46. Prioridade para novos testes

Priorizar **representações realmente diferentes**.

Evitar microvariações como:

```text
BERTimbau com LR ligeiramente diferente
BERTimbau com 300 tokens
label smoothing 0.05
dezenas de pesos de ensemble
```

Procurar soluções que:

- cometam erros diferentes;
- melhorem `c234`;
- capturem propriedades que os Transformers atuais não capturam;
- tragam complementaridade a baixo custo.

O benchmark a superar é:

## 47,5662% Accuracy OOF

---

# 47. Quando o test set chegar

Se nenhuma nova solução superar convincentemente o V3, congelar definitivamente o V3.

Treinar em 100% de `train.xlsx`:

```text
BERTimbau512
BERTimbau soft-label
LinearSVC
baseline LR
ordinal corrigido
NorBERTo
```

Gerar previsões para o test.

Aplicar exatamente Majority-5 V2 e depois:

```text
se houver empate 2-2-1
e c234 estiver entre as duas classes líderes
→ usar probabilidades do NorBERTo somente entre as duas líderes
```

Não recalibrar nada utilizando o test set.

---

# 48. Checklist antes de um novo experimento

## Antes

```text
[ ] Estou usando train.xlsx?
[ ] Estou usando data/splits/folds.csv?
[ ] Não recriei folds?
[ ] Estou usando seed 42?
[ ] Sei qual é o mapeamento c1/c234/c5?
[ ] O script é separado?
[ ] Sei qual hipótese estou testando?
```

## Durante

```text
[ ] Loss está finita?
[ ] grad_norm está finito?
[ ] GPU está estável?
[ ] Não ocorreu OOM?
[ ] Runtime é aceitável?
```

## Depois

```text
[ ] Salvei configuração?
[ ] Salvei métricas?
[ ] Salvei runtime?
[ ] Salvei OOF?
[ ] OOF possui row_id?
[ ] Probabilidades estão presentes quando disponíveis?
[ ] Comparei com V3?
[ ] Analisei c234?
[ ] Calculei disagreement?
[ ] Calculei novo correto / V3 errado?
[ ] Calculei V3 correto / novo errado?
[ ] Usei oracle apenas como diagnóstico?
```

---

# 49. Resumo dos principais resultados

| Experimento | Accuracy aproximada | Situação |
|---|---:|---|
| Baseline TF-IDF + LR | 45,18% | Referência |
| BERTimbau 256 | 46,29% | Concluído |
| BERTimbau 384 | 46,16% | Concluído |
| BERTimbau 512 | **46,50%** | Melhor BERT principal |
| Soft labels 256 | 46,41% | Componente ensemble |
| Ordinal | 44,97% | Fraco standalone, útil no ensemble |
| Head+Tail | 45,94% | Encerrado |
| Label smoothing | 46,21% | Encerrado |
| LinearSVC | 44,41% | Fraco standalone, útil no ensemble |
| NorBERTo | 46,36% | Complementar, especialmente c234 |
| Ensemble V2 | **47,382%** | Superado |
| **Ensemble V3** | **47,5662%** | **MELHOR OFICIAL** |
| Albertina | ~45% no fold 0 | Encerrado |
| SetFit corrigido | 42,67% no fold 0 | Encerrado |
| XLM-R 384 | 44,84% no fold 0 | Encerrado |

**Atenção:** Albertina, SetFit corrigido e XLM-R possuem resultados apenas do fold 0, e não CV completa.

---

# 50. Conclusão

O projeto começou com um baseline TF-IDF + Logistic Regression em aproximadamente **45,18% de Accuracy**.

O fine-tuning de BERTimbau elevou o desempenho para aproximadamente **46,5%**, mas aumentos de contexto e pequenas alterações de treinamento produziram ganhos limitados.

A principal descoberta experimental foi que **diversidade de erros pode ser mais valiosa que performance standalone**.

LinearSVC e o modelo ordinal são exemplos de classificadores relativamente fracos individualmente que ainda contribuem para o ensemble.

O NorBERTo confirmou essa hipótese: sua Accuracy standalone não foi muito superior aos melhores BERTimbau, mas sua capacidade de corrigir erros diferentes, especialmente em `c234`, permitiu criar uma regra simples e interpretável de desempate.

O melhor resultado atual é:

```text
Ensemble V3

OOF Accuracy:
47,5662%

OOF Macro-F1:
47,3933%
```

Esse é o benchmark que qualquer nova solução deve enfrentar.

O número de experimentos realizados sobre os mesmos OOFs já é significativo. A prioridade daqui para frente deve ser testar poucas hipóes fortes, preferencialmente envolvendo representações diferentes e de baixo custo computacional, em vez de continuar micro-otimizando Transformers e regras de ensemble.

O rigor metodológico tem prioridade sobre pequenos ganhos aparentes.

A solução final somente deverá ser congelada após a fase de desenvolvimento utilizando exclusivamente `train.xlsx` e os cinco folds oficiais.

O futuro conjunto de teste deverá servir **somente para avaliação final da solução previamente escolhida**.
"""

output = "/mnt/data/relatorio_experimentos_pln.md"
pypandoc.convert_text(content, "md", format="md", outputfile=output, extra_args=["--standalone"])
print(output)
