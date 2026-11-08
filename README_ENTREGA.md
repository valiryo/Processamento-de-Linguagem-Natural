# EP2 — Grupo 13 — instruções de reprodução

Entrega de 08/11/2026. Método: gate de incerteza congelado sobre quatro distribuições neurais e três votos codificados. O V3 do EP1 é uma das entradas, não a saída final do EP2.

## 1. Arquivos e acesso

- `grupo13.zip`: exatamente relatório PDF, apresentação PDF e `test2_rotulado_grupo13.xlsx`. Este é o ZIP acadêmico principal.
- `codigo_grupo13.zip`: fontes com os bytes originais, modelo do gate, congelamento, dependências e instruções.
- `artefatos_reproducao_grupo13.zip`: folds, OOFs históricos e caches intermediários necessários. Disponibilizar junto ao código, fora do ZIP principal.
- Dados oficiais, obtidos nos arquivos da disciplina: `train.xlsx` (20.092), `test1.xlsx` **sem rótulos** (900), `test2.xlsx` (897). Copiar em `data/raw/`, sem abrir e salvar novamente no Excel.
- Para McNemar, separadamente: `test1-rotulado-para-teste-estatistico.xlsx` do professor e predições históricas do EP1 disponíveis no repositório. Não são entradas para ajuste ou inferência.

Repositório: https://github.com/valiryo/Processamento-de-Linguagem-Natural . A auditoria confirmou remotamente o commit `b7ad5ba63fa6d402a8939aada9fca9f278f3fb41`. A revisão documental e os auxiliares desta auditoria precisam ser publicados pelo grupo. A existência dos ZIPs locais não equivale à disponibilidade remota.

## 2. Modalidade 1 — reproduzir o método congelado, em CPU

Esta modalidade foi projetada para não exigir CUDA nem novos treinamentos neurais. Executa o gate original sobre os caches originais, verifica as fontes e compara as saídas com os hashes registrados. Não copia a planilha pronta para fingir inferência.

1. Coloque os dois pacotes complementares na pasta atual do PowerShell. Extraia em uma pasta **nova**, chamada `ep2_reproducao`:

```powershell
$ErrorActionPreference = 'Stop'
if (Test-Path .\ep2_reproducao) { throw 'Escolha uma pasta nova para preservar execuções anteriores.' }
Expand-Archive .\codigo_grupo13.zip -DestinationPath .\ep2_reproducao
Expand-Archive .\artefatos_reproducao_grupo13.zip -DestinationPath .\ep2_reproducao
Set-Location .\ep2_reproducao
New-Item -ItemType Directory -Force .\data\raw | Out-Null
```

2. Copie os três arquivos oficiais para `ep2_reproducao/data/raw/`. Instale Python **3.11.9 x64**. Crie o ambiente e instale o lock mínimo; a ativação do ambiente não é necessária:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ep2-cpu-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Falha na instalação.' }
.\.venv\Scripts\python.exe -m pip check
```

3. Confira os dados, todos os arquivos, o congelamento e a ordenação. Em seguida execute o gate e confira as 897 respostas:

```powershell
.\.venv\Scripts\python.exe src/ep2/reproduzir_ep2.py conferir
if ($LASTEXITCODE -ne 0) { throw 'Integridade reprovada; não prossiga.' }
.\.venv\Scripts\python.exe src/ep2/reproduzir_ep2.py inferir
if ($LASTEXITCODE -ne 0) { throw 'Inferência reprovada; não substituir a entrega.' }
.\.venv\Scripts\python.exe src/ep2/reproduzir_ep2.py validar-saida
```

Saída: `results/ep2/reproducao/final/test2_rotulado_GATE.xlsx`, 897 linhas, colunas `resp_text` e `clarity`, rótulos c1/c234/c5, distribuição 296/273/328. O comando confere os hashes contra a entrega congelada. Repetir `inferir` apenas valida uma saída já existente, sem sobrescrever. Uma divergência é erro, nunca motivo para escolher novos parâmetros.

4. Recalcule as três ablações em cinco folds, usando os OOFs originais. O ajuste pequeno do gate ocorre na CPU e mantém as metades de calibração/avaliação separadas:

```powershell
.\.venv\Scripts\python.exe src/ep2/reproduzir_ep2.py oof
```

Saída: `results/ep2/reproducao/oof/`. Acurácias globais esperadas: V3 47,5662%; constante 47,6856%; incerteza 47,7105%; discurso 47,4020%. Para repetir todos os ajustes, use uma pasta nova com `--saida results/ep2/reproducao_2`; o comando recusa retomar silenciosamente uma avaliação anterior.

5. Somente para verificar a comparação estatística já concluída, copie o gabarito do professor em `data/raw/test1-rotulado-para-teste-estatistico.xlsx`. Obtenha a referência real do EP1 no commit histórico e execute:

```powershell
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/valiryo/Processamento-de-Linguagem-Natural/9461ced46632d90ebf185a5bbd35b6f166247248/results/ensemble_majority_5_v3_norberto_final.csv' -OutFile .\results\ensemble_majority_5_v3_norberto_final.csv
.\.venv\Scripts\python.exe src/ep2/reproduzir_ep2.py mcnemar
```

Esperado: EP1 434/900, EP2 438/900; b=37, c=41, estatística exata min(b,c)=37, p=0,7343420330. **Não há superioridade estatisticamente significativa.** Os CSVs individuais do teste1 são materiais de avaliação local; não entram no ZIP acadêmico.

## 3. Modalidade 2 — reconstrução integral dos componentes

O procedimento detalhado e ordenado está em [REPRODUCAO_EP2.md](REPRODUCAO_EP2.md), seção 4. Começa pela criação de uma cópia limpa, preserva o `folds.csv` original, treina os componentes, gera OOFs, reproduz as ablações e ajusta/congela o mesmo tipo de gate. Os comandos `preparar` e `congelar` só aparecem nessa cópia, sem artefatos previamente congelados.

Essa modalidade exige GPU NVIDIA, downloads dos modelos e muitas horas. **Não foi reexecutada integralmente nesta auditoria.** Seeds iguais não garantem probabilidades ou hashes idênticos entre ambientes. Para reproduzir exatamente os números entregues, use a modalidade 1 com os artefatos originais.

## 4. Limites e documentos

- A acurácia do teste2 é desconhecida; não há gabarito disponível para comparação de modelos.
- Instalação mínima, inferência e OOF: consultar os resultados efetivamente testados em [AUDITORIA_FINAL_EP2.md](AUDITORIA_FINAL_EP2.md).
- A interpretação e os experimentos anteriores usam arquivos adicionais listados no guia e no manifesto. Eles não são necessários para inferir com o gate congelado.
- `candidato.json` conserva seu estado histórico de preparação. `congelamento.json` é o registro posterior vigente; nenhum desses documentos científicos foi reescrito.
- Não executar o antigo gerador de documentos para reproduzir predições. Os PDFs corretos já estão em `grupo13.zip`.
