# Resumo dos Exercícios Práticos (EP1 e EP2)

**ACH2118 - Introdução ao Processamento de Língua Natural (EACH - USP)**  
_Professor: Ivandré Paraboni_

---

## 📋 Contexto Geral da Tarefa (EP1)

A tarefa consiste em classificar de forma automática a **clareza da resposta** fornecida ao usuário em sistemas públicos de resposta a requisições do cidadão (e-SIC).

- **Classes da tarefa (Classificação Ternária):** `{c1, c234, c5}`
  - `c1`: Nota 1 dada pelo usuário (baixa clareza).
  - `c234`: Notas intermediárias 2, 3 e 4 agregadas em uma única classe.
  - `c5`: Nota 5 dada pelo usuário (alta clareza).
- **Dados:**
  - **Treino:** Córpus rotulado (`train.xlsx`) contendo o texto da resposta (`resp_text`) e o rótulo de clareza (`clarity`).
  - **Teste:** Conjunto de textos sem rótulos que deve ser classificado pelo modelo desenvolvido.

---

## 🚀 Evolução: EP1 vs. EP2

### Exercício Prático 1 (EP1)

- **Objetivo:** Construir do zero a infraestrutura de desenvolvimento e um pipeline robusto de experimentação com aprendizado de máquina supervisionado para PLN.
- **Tarefa de Teste:** Rotular o primeiro conjunto de testes (**Teste 1**) contendo 900 instâncias.
- **Baseline de Referência:** Regressão Logística tradicional utilizando contagens TF-IDF.

### Exercício Prático 2 (EP2)

- **Objetivo:** Evoluir obrigatoriamente a solução desenvolvida no EP1 para obter resultados significativamente superiores.
- **Novos Conjuntos de Teste:**
  - **Teste 2 (897 instâncias):** Um novo conjunto de testes sem rótulos que constitui a **avaliação oficial do EP2**. Suas predições devem ser entregues.
  - **Teste 1 (900 instâncias):** Agora fornecido com os 900 rótulos reais. Deve ser usado apenas para realizar um teste estatístico de **McNemar** comparando formalmente o desempenho do EP1 com as melhorias do EP2.

---

## 📦 Entregáveis e Prazos

| Item                   | EP1                                                                                                                                      | EP2                                                                                                                                                                            |
| :--------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Limite**        | Domingo, 04/10                                                                                                                           | Domingo, 08/11                                                                                                                                                                 |
| **Formato de Envio**   | Arquivo `grupoXX.zip` enviado pelo responsável                                                                                           | Arquivo `grupoXX.zip` enviado pelo responsável                                                                                                                                 |
| **Componentes do ZIP** | 1. Relatório do projeto no modelo indicado;<br>2. Apresentação em formato **PDF** (máx. 10 min);<br>3. Planilha do **Teste 1** rotulado. | 1. Relatório do projeto com o novo modelo e análise estatística;<br>2. Apresentação em formato **PDF** (máx. 10 min);<br>3. Planilha do **Teste 2** rotulado (897 instâncias). |
| **Apresentações**      | Em aula nos dias 06/10 e 13/10                                                                                                           | Em aula nos dias 10/11 e 17/11                                                                                                                                                 |

_Nota: A ordem das apresentações em aula ocorre do grupo com menor acurácia de teste para o de maior acurácia (do último ao primeiro colocado)._

---

## 📊 Critérios de Avaliação e Composição de Notas

| Critério                        | Peso no EP1 | Peso no EP2 | Detalhes / Observações                                                                                                                                                    |
| :------------------------------ | :---------- | :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Acurácia de Teste**           | 70%         | 70%         | Métrica principal de desempenho oficial do modelo.                                                                                                                        |
| **Inovação / Elaboração**       | 20%         | 10%         | Métodos clássicos como TF-IDF/BoW com Regressão Logística, SVM ou Naive Bayes **não pontuam como inovação**. No EP2, a repetição do método do EP1 também perde esta nota. |
| **Análise de Características**  | -           | 5%          | Avaliação qualitativa e quantitativa dos dados e variáveis.                                                                                                               |
| **Melhoria estatística vs EP1** | -           | 5%          | Exige resultado estatisticamente superior ao EP1 através do teste de McNemar ($p < 0,05$).                                                                                |
| **Relatório**                   | 5%          | 5%          | Clareza e adequação ao modelo fornecido.                                                                                                                                  |
| **Apresentação em aula**        | 5%          | 5%          | Desempenho na apresentação oral em sala de aula.                                                                                                                          |

- **Normalização:** A nota final de ambos os EPs é normalizada pela maior nota da turma.

---

## ⚠️ Penalidades e Valorizações

### Penalidades (Aplicável a ambos os EPs)

- **Baixo Desempenho (Nota 3,0 automática):** Soluções cuja acurácia média seja inferior à média do baseline de classe majoritária não são avaliadas e perdem o direito ao critério "inovação".
- **Erros de Entrega (Nota 3,0 automática no EP2):** Entregar acidentalmente rótulos do _Teste 1_ no lugar do _Teste 2_, ou realizar a comparação dos modelos do EP1 vs EP2 usando o conjunto _Teste 2_.
- **Desconto cumulativo de 20% por ocorrência de:**
  - Desempenho final inferior ao baseline oficial (Regressão Logística + TF-IDF);
  - Atraso na entrega do relatório ou código;
  - Atraso na entrega do PDF da apresentação;
  - Formato de arquivo inválido (ex: mais ou menos linhas/colunas, tipos diferentes de `.xlsx`, rótulos fora de posição).

### Valorização de Esforço (Bônus para Perda de Integrantes)

Caso algum integrante do grupo desista ou desapareça durante o projeto, o grupo pode reportar no relatório para receber uma bonificação compensatória:

- **No EP1:** Bônus de **10%** na nota final para cada integrante perdido.
- **No EP2:** Bônus de **5%** na nota final para cada integrante perdido.
