# Reprodução final auditada em 08/11/2026

Para entregar e reproduzir o gate congelado, seguir [README_ENTREGA.md](../../README_ENTREGA.md), [REPRODUCAO_EP2.md](../../REPRODUCAO_EP2.md) e [AUDITORIA_FINAL_EP2.md](../../AUDITORIA_FINAL_EP2.md). As instruções abaixo registram o desenvolvimento anterior; não substituem a rota final auditada.

# EP2 — Experimento 01: especialista BERTimbau para `c234`

> **Estado final — grupo 13:** o gate de incerteza foi congelado e avaliado no teste1 após autorização do usuário: 438/900 acertos, contra 434/900 do EP1; McNemar p=0,734342, sem superioridade significativa. A entrega está em `results/ep2/entrega_grupo13/`. Consulte a atualização no início de [CONTINUACAO_EP2.md](CONTINUACAO_EP2.md). O texto abaixo preserva a preparação histórica do experimento 01 e contém afirmações desatualizadas de que ele não havia sido executado.

**ACH2118 • 28/10/2026 • experimento exploratório, sem resultados neurais ainda**

## Motivação

O Ensemble V3 do EP1 obteve **0,475661955 de accuracy OOF** (9.557/20.092). Seu recall de `c234` foi **0,3722**. No fold 0, o V3 acertou **1.900/4.019** (**0,4727544**). Dentre as **2.896** predições V3 que não eram `c234` nesse fold, **904** tinham `c234` como classe verdadeira; outras **1.433** eram extremos corretamente classificados e **559** eram extremos classificados com o extremo errado. Portanto, um especialista precisa identificar um subconjunto seletivo de `c234` para compensar os acertos extremos que pode destruir.

A hipótese é que um encoder fine-tuned especificamente para `c234` versus `c1 ∪ c5` consiga identificar exemplos intermediários não reconhecidos pela classificação ternária. **A hipótese não foi testada ainda**; neste ambiente não há GPU CUDA nem biblioteca Transformers instalada. Não confundir melhoria de recall de `c234` com melhoria de accuracy global.

## Arquivo fornecido

`src/ep2/especialista_c234.py`: script de validação de dados, treino binário, inferência e comparação com as predições originais do V3. **Nunca lê o teste1 ou teste2.**

## Preparação (Windows PowerShell)

Na raiz do repositório original, mantenha a organização que você já utiliza:

```text
Processamento-de-Linguagem-Natural/
├── data/
│   ├── raw/
│   │   ├── train.xlsx
│   │   └── test1.xlsx  # não utilizado
│   └── splits/
│       └── folds.csv
├── results/
│   └── ensemble_majority_5_v3_norberto_oof.csv
└── src/
    └── ep2/
        └── especialista_c234.py
```

**O caminho `data/raw/train.xlsx` já é o padrão do script atualizado.** Não é necessário mover arquivos nem passar `--train` nos comandos.

**Não recrie os folds.** O script testa alinhamento da ordem das 20.092 linhas, concordância de rótulos/folds com o OOF, ausência de textos idênticos em folds diferentes e accuracy original.

Use o ambiente Python que já funcionou no EP1, com `torch` **habilitado para CUDA**, `transformers`, `numpy`, `pandas`, `scikit-learn` e `openpyxl`. Se faltar Transformers ou utilitários:

```powershell
python -m pip install "transformers>=4.40,<6" numpy pandas scikit-learn openpyxl
```

**PyTorch CUDA:** se ainda não estiver instalado, use as instruções do [site oficial do PyTorch](https://pytorch.org/get-started/locally/) para a combinação de SO/driver/CUDA da sua máquina. Não execute treinamento em CPU por engano.

### 1) Verificar integridade sem treinamento

```powershell
python src/ep2/especialista_c234.py --self-test
python src/ep2/especialista_c234.py --check-data
```

Esperado:

```text
treino=16073 | validação fold 0=4019
V3 OOF completo: 0.47566196
classe positiva c234 no treino do fold: 5482/16073
Validação de integridade: OK. Nenhum treinamento executado.
```

### 2) Treinar primeira variante: sem pesos de classe

```powershell
python src/ep2/especialista_c234.py --fold 0 --class-weight none
```

Parâmetros fixados para o piloto:

| Item | Configuração |
|---|---|
| Encoder | `neuralmind/bert-base-portuguese-cased` |
| Tarefa | binária: `extremos (0)` versus `c234 (1)` |
| Holdout | fold 0 do `folds.csv` (4.019 exemplos) |
| Treino | demais quatro folds (16.073 exemplos) |
| Comprimento | 256 tokens |
| Learning rate | `3e-5` |
| Épocas | 2 (predições da época 2) |
| Batch físico | 8 |
| Acumulação | 2 (batch efetivo ~16) |
| Weight decay | 0,01 |
| Warmup | 10% dos passos, scheduler linear |
| Precisão | FP16/AMP em GPU NVIDIA |
| Random seed | 42 |
| Loss inicial | cross-entropy sem pesos |
| Critério primário | **accuracy ternária do V3 + regra fixa**, comparada no mesmo fold |

Observação: há diferenças intencionais em relação ao experimento ternário do EP1 (cabeça binária e warmup explícito de 10%). Não declarar os procedimentos como idênticos; o uso do mesmo encoder e folds torna a comparação de validação relevante.

### 3) Se ocorrer CUDA out of memory

Reduza o batch físico mantendo aproximadamente o batch efetivo:

```powershell
python src/ep2/especialista_c234.py --fold 0 --class-weight none --batch-size 4 --grad-accum 4
```

Se for alterar a configuração, preserve e registre o valor utilizado nos resultados. A primeira configuração é semelhante à que funcionou no BERTimbau 256 do EP1, mas a memória real pode variar.

### 4) Experimento de ablação com balanceamento (depois de analisar o primeiro)

```powershell
python src/ep2/especialista_c234.py --fold 0 --class-weight balanced
```

Isso aumenta o custo dos erros positivos da classe `c234`. Compare com a variante sem pesos, mas **não escolha uma regra ou limiar baseado nas mesmas instâncias do fold 0 e depois apresente o fold 0 como teste independente dessa regra**. Para avaliar decisões de combinação ajustáveis, usar validação interna adicional/cross-fitting dentro do treino; depois avaliar em partições externas não usadas nas escolhas.

## Como avaliamos sem vazamento

O especialista é treinado apenas nos quatro folds de treino. O V3 é carregado do arquivo **OOF**, ou seja, sua predição sobre os exemplos do fold 0 não veio de treino nesses mesmos exemplos. A regra de combinação não é aprendida no fold 0:

```python
pred_combinada = c234 if (P(c234) >= 0.5 and pred_v3 != c234) else pred_v3
```

O limiar **0,5** foi fixado **antes de executar o piloto** e não é calibrado no holdout. Não estamos dizendo que é ótimo. A regra só pode recuperar erros em que V3 marcou `c1`/`c5` mas o correto era `c234`. Ela não corrige um falso positivo `c234` do V3 nem confusões entre `c1` e `c5`.

Para fazer previsões finais, é necessário avaliar todas as cinco partições e depois, se selecionado, treinar o modelo final em 100% do conjunto de treino. Nenhum modelo do EP2 foi selecionado ainda.

## Saídas

Com o comando acima, serão gerados em `results/ep2/especialista_c234/fold0_none_seed42/`:

- `config.json`: parâmetros, checksum SHA-256 dos 3 arquivos de entrada e ambiente.
- `metricas.json`: binary precision/recall/F1/AUC/AP de `c234`, accuracy ternária e macro-F1 antes e depois, matriz de confusão, **erros corrigidos vs acertos quebrados** e histórico de loss.
- `predicoes_fold.csv`: índices originais, verdade, previsão V3, probabilidade `c234`, resultado binário, nova previsão e indicadores de acerto.
- `por_extensao.csv`: comparação de accuracy global por faixa de comprimento e recall `c234`.
- `checkpoint/`: pesos do especialista e tokenizer para reuso (~400 MB); omita com `--no-save-model` se necessário.

**Como trazer o resultado para esta conversa:** envie `metricas.json`, `predicoes_fold.csv` e, se possível, `por_extensao.csv`. Não é necessário enviar o checkpoint pesado inicialmente.

## Verificações já executadas neste ambiente

- `python -m py_compile src/ep2/especialista_c234.py`: **OK**.
- `--self-test`: **OK**.
- `--check-data` com os arquivos reais fornecidos: **OK**, treino 16.073/validação 4.019; OOF V3 0,47566196.
- **Treinamento neural/inferência do especialista: NÃO executados** aqui porque CUDA e Transformers não estão disponíveis neste ambiente.

## Integração e ética experimental

Não consultar `test1` rotulado ou `test2` para selecionar arquitetura, pesos, thresholds ou combinações. `test1` só para McNemar **após** o modelo EP2 ser congelado; `test2` somente para predição oficial de 897 rótulos. O V3 permanece como referência intacta. Registrar qualquer escolha motivada pelo fold 0, inclusive descartes.
