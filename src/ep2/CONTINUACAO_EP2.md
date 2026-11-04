# EP2 — estado verificado em 03/11/2026

> **ATUALIZAÇÃO FINAL — GRUPO 13:** o usuário autorizou o congelamento e a avaliação oficial e confirmou os quatro integrantes do EP1. O gate de incerteza foi congelado; as 900 e 897 predições foram geradas. Teste1: EP1 434/900 (48,2222%), EP2 438/900 (48,6667%), McNemar exato bicaudal p=0,7343420330, sem superioridade significativa. O método não foi reajustado após essa avaliação. A entrega atual é `results/ep2/entrega_grupo13/grupo13.zip`, com código complementar em `codigo_grupo13.zip` e instruções em `README_ENTREGA.md` na mesma pasta. O trecho abaixo preserva o registro anterior ao congelamento e NÃO representa mais o estado atual. Não voltar ao desenvolvimento usando o teste1 já aberto.

**Decisão do usuário:** preparar o gate de incerteza como candidato e manter o teste1 fechado. Não congelar o método nem executar McNemar até nova decisão. O candidato NÃO substitui a planilha de backup.

## Auditoria e decisões

- Treino: 20.092 linhas; 18.432 textos distintos; 463 grupos repetidos; 291 grupos com classes conflitantes. Uma célula numérica em `resp_text` (row_id 5026) é convertida em string, exatamente como nos scripts históricos.
- Folds: os cinco originais, íntegros; grupos de texto não atravessam folds. Nenhum fold foi recriado.
- V3: a regra foi reproduzida pelo código histórico e coincide em todas as linhas. Acurácia 9.557/20.092 = **47,5661955%**; macro-F1 **47,3932944%**.
- Ambiente: Python 3.11.9, PyTorch 2.14.0+cu130, Transformers 5.17.0; CUDA disponível, NVIDIA GeForce RTX 3050 Laptop GPU. Não houve novo treinamento neural nesta rodada.
- Existem os seis componentes de inferência do V3, manifest, CSVs separados e XLSX de backup. Nesta rodada eles foram preservados; não foram promovidos a saídas do candidato.
- Checkpoints presentes incluem cinco arquivos de pesos na pasta NorBERTo e arquivos dos experimentos EP2. A presença de checkpoint não comprova, por si só, a configuração de treinamento.
- O modelo oficial de relatório foi encontrado em `C:\Users\valer\OneDrive\Desktop\ep2-modelo-relatorio-com-analise.pdf` e seus dez itens orientam o rascunho.
- Os resultados completos de TAPT e moBERTo foram recalculados. Especialista e fronteiras adjacentes possuem somente piloto no fold 0. O consenso TAPT+moBERTo é exploratório, após observação do fold 0.
- Os resultados brutos dos experimentos 06 (discurso isolado) e 07 (limpeza) citados no guia não foram encontrados, inclusive nos ZIPs inspecionados. O script discursivo está presente; o da limpeza não foi localizado. Não tratar os números do guia como reproduzidos nesta rodada.
- Faltam número do grupo e confirmação dos integrantes efetivos. O relatório identifica a lista como transcrita do EP1, sem pressupor desistências.

Fontes verificáveis: `results/ep2/gating_discursivo/auditoria.json`, `results/ep2/analise_candidato/experimentos_reproduzidos.csv`, manifests dos novos diretórios.

## Hipóteses limitadas e custo

| Hipótese | Motivação | Custo / expectativa | Resultado |
|---|---|---|---|
| H1: gate de incerteza + sinais discursivos | Escolher especialistas conforme confiança, estrutura e marcadores; investigar erros de c234 | CPU, sem novos encoders; estimativa inicial de poucos minutos | 47,4020%; rejeitada; piora em relação ao V3 |
| H2: retirar sinais discursivos, mantendo gate de incerteza | Ablação de H1; investigar se confiança já contém o sinal útil | Reutiliza os mesmos OOFs; avaliação adicional de segundos | 47,7105%; destacado como **candidato exploratório**, a pedido do usuário |

O registro inicial previa promover apenas a variante discursiva se superasse V3 e as duas ablações, com ganho em ao menos três folds. Esse critério falhou e não foi reescrito. A escolha posterior da ablação de incerteza está documentada separadamente em `candidato_incerteza/candidato.json`; não é seleção pré-especificada nem confirmação independente.

Nenhuma segunda busca de hiperparâmetros ou ciclo de GPU foi realizado. O tempo em `resumo.json` corresponde à execução retomada dos folds 1–4, não à extração inicial de atributos/fold 0; não o citar como duração total.

## Resultado verificável

| Método | Acurácia | Macro-F1 | Recall c234 | Acertos |
|---|---:|---:|---:|---:|
| V3 | 47,5662% | 47,3933% | 37,2246% | 9.557 |
| Mistura constante | 47,6856% | 47,4845% | 36,9327% | 9.581 |
| Gate de incerteza | **47,7105%** | **47,5200%** | 37,0641% | 9.586 |
| Gate discursivo | 47,4020% | 47,2061% | 36,5387% | 9.524 |

O candidato soma 29 acertos contra V3 (+0,1443 ponto percentual), mas apenas 5 contra pesos constantes. Corrige 332 erros e perde 303 acertos. Melhora em 4/5 folds; recall c234 piora. **Não alegar superioridade significativa ou ganho confirmado do componente condicional.**

## Protocolo do gate

Para cada fold externo f, as previsões-base são provenientes de treinamento fora de f. Dividir f em duas partes por grupos; calibrar gate/scaler em A e avaliar em B, depois inverter. Nenhum gate é ajustado nos OOFs de outros folds: os respectivos modelos-base poderiam ter visto os rótulos do fold avaliado. Os índices exatos de calibração/avaliação estão nos JSONs por fold.

O candidato final é reajustado em **todos os OOFs de treino**, sem ler testes. Isso usa mais dados de calibração do que cada avaliação (~2.000 linhas), como é explicitado no relatório. O comportamento final ainda não foi validado em teste independente.

Sete entradas: probabilidades de BERT512, soft256, ordinal e NorBERTo; votos de SVC, baseline e V3 convertidos para distribuição auxiliar 0,6/0,2/0,2. Onze sinais: quatro entropias, quatro margens, três frações dos cinco votos. Softmax gera pesos; a mistura recebe interceptos de classe. L2=0,02; L-BFGS-B; até 300 iterações; StandardScaler apenas na calibração. Não há ajuste de limiar no conjunto avaliado.

## Comandos executados — PowerShell na raiz

```powershell
.\.venv\Scripts\python.exe src/ep2/gating_discursivo.py --check-data
.\.venv\Scripts\python.exe src/ep2/gating_discursivo.py --fold 0
.\.venv\Scripts\python.exe src/ep2/gating_discursivo.py
.\.venv\Scripts\python.exe src/ep2/analisar_candidato.py
.\.venv\Scripts\python.exe src/ep2/candidato_incerteza.py preparar
.\.venv\Scripts\python.exe src/ep2/test_gating_discursivo.py -v
.\.venv\Scripts\python.exe src/ep2/documentar_candidato.py
```

O gate retoma folds já concluídos se a assinatura das fontes/configuração coincidir. Para reprodução independente, use uma nova pasta, sem sobrescrever resultados:

```powershell
.\.venv\Scripts\python.exe src/ep2/gating_discursivo.py --out results/ep2/gating_reproducao
```

O ambiente existente foi usado para aprendizado. `reportlab`, `python-docx` e `pypdf` foram instalados para documentos; versões estão no relatório. Não reinstalar os Transformers ou modelos para reproduzir o gate.

## Artefatos preparados

- `results/ep2/gating_discursivo/`: protocolo pré-especificado, auditoria, dados de calibração por fold, OOF, tabelas, exemplos e ablação.
- `results/ep2/candidato_incerteza/modelo.json`: pesos do candidato, serializados sem pickle; `candidato.json` registra estado **NÃO CONGELADO**, hashes e limitação da escolha exploratória.
- `results/ep2/analise_candidato/`: interpretação global do baseline, contribuições locais, atributos por classe e histórico recalculado. As 4.019 predições do baseline no fold 0 coincidem com o histórico. Contribuições completas recompõem a margem; a lista dos 12 maiores termos é apenas um resumo.
- `results/ep2/documentacao_2026-11-03/`: relatório MD/DOCX/PDF, apresentação PDF de 9 slides e roteiro de 8min40s. **Rascunhos: falta avaliação oficial, número do grupo e revisão pelos integrantes.**

## Etapas futuras — NÃO executadas

Somente depois de decidir encerrar a seleção do método, executar:

```powershell
.\.venv\Scripts\python.exe src/ep2/candidato_incerteza.py congelar --confirmar-escolha-final
.\.venv\Scripts\python.exe src/ep2/candidato_incerteza.py inferir
```

A inferência exige congelamento e caches históricos íntegros. Reutiliza os componentes já calculados; não retreina modelos nem baixa encoders. Gera arquivos novos em `results/ep2/candidato_incerteza/final/`, sem sobrescrever `results/ep2/final/test2_rotulado_EP2.xlsx`. Falha se as saídas já existirem. Faz a mesma transformação de atributos e verifica quantidades, rótulos e hashes de texto.

Depois, e somente depois, a avaliação oficial pode ser executada:

```powershell
.\.venv\Scripts\python.exe src/ep2/entrega_ep2.py mcnemar --metodo-congelado --predicoes-ep2 results/ep2/candidato_incerteza/final/predicoes_test1.csv --saida results/ep2/candidato_incerteza/final/mcnemar_oficial.json
```

Esse comando **abre o gabarito**. Ele usa as 900 predições históricas do EP1 e verifica o alinhamento com o teste1 original. Não usar o subcomando `entrega_ep2.py validar` durante desenvolvimento: ele também abre o teste1 rotulado. Para McNemar exato, a estatística convencional é min(b,c); o script histórico retorna as contagens b/c e o p-valor, permitindo relatar a estatística a partir delas.

Os caminhos de inferência e McNemar são preparados, mas ainda não foram executados com os testes reais nesta rodada. O bloqueio antes do congelamento foi testado. Não interpretar os comandos como comprovação de execução.

### Verificações concluídas

Cinco testes passaram: gradiente analítico contra diferenças finitas; isolamento de grupos e cobertura de todas as linhas; bloqueio de inferência antes do congelamento; integridade/serialização do candidato; e exportação completa usando **900 + 897 textos artificiais em diretório temporário**, sem abrir testes oficiais. Esse último teste validou CSVs e XLSX sintéticos e não produziu uma submissão real.

Os arquivos Python novos compilaram; o relatório PDF foi conferido estruturalmente (5 páginas), o DOCX passou na verificação de integridade e a apresentação tem 9 páginas. Não foi feito ensaio oral nem revisão humana dos rascunhos.

Ao terminar: atualizar os rascunhos com resultados reais; confirmar número do grupo; disponibilizar código; gerar `grupoXX.zip` com os três itens exigidos. Não incluir a planilha de teste1, ambientes virtuais ou checkpoints. Não há ZIP final novo nesta rodada, pois a escolha foi mantida aberta pelo usuário.
