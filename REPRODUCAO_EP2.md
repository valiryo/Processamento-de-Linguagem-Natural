# Reprodução técnica do EP2 — Grupo 13

## 1. Escopo e ambiente

Há duas formas de reprodução: execução numérica do **gate congelado sobre as probabilidades originais** e reconstrução de todos os componentes a partir dos textos. A primeira permite verificar exatamente as 897 predições entregues, sem retreinar redes. A segunda reproduz o procedimento científico, mas não promete igualdade bit a bit após novos treinamentos.

Ambiente de desenvolvimento observado: Windows x64, PowerShell, CPython 3.11.9, NVIDIA GeForce RTX 3050 Laptop GPU (4.096 MiB), driver 591.59, PyTorch 2.14.0+cu130, CUDA de runtime 13.0, Transformers 5.17.0. O runtime informado é o do PyTorch; não foi assumida instalação de um toolkit CUDA separado.

- `requirements-ep2-cpu-lock.txt`: NumPy, pandas, SciPy, scikit-learn, openpyxl, threadpoolctl e dependências transitivas fixadas. Basta para inferir e recalcular as ablações do gate.
- `requirements-ep2-docs-lock.txt`: acrescenta ReportLab, python-docx, pypdf e transitivas. Não é necessário para classificar.
- `requirements-ep2-neural-lock.txt`: registra o ambiente neural observado, incluindo PyTorch, Transformers, datasets, accelerate, sentencepiece e transitivas. Reinstalação neural completa não foi realizada nesta auditoria; não confundir registro do ambiente com teste de instalação concluído.
- `requirements.txt`: lista histórica ampla de exploração, sem fixação; não é o instalador recomendado do EP2. Jupyter, ipykernel, matplotlib e SetFit pertencem a explorações antigas, não à execução final.
- `statsmodels` não está instalado e não é importado pelos scripts do método final. O McNemar exato usa a distribuição binomial implementada com `math.comb`; a conferência independente usa `scipy.stats.binomtest`. Não há dependência oculta de statsmodels ou eli5.

CPU é suficiente para a modalidade congelada. Não foi medido um limite mínimo de RAM; não há número mínimo certificado. A reconstrução neural usa FP16 e os scripts exigem CUDA, portanto CPU não é uma alternativa direta para esses treinamentos sem alterações. A máquina de 4 GB executou os experimentos registrados; NorBERTo ficou próximo do limite (pico alocado de aproximadamente 3,84–3,87 GiB). Fechar outras aplicações que ocupem a GPU. Não reduzir batch, contexto ou outros hiperparâmetros para contornar limitações sem registrar uma reprodução diferente.

## 2. Inventário e procedência

O inventário exato está em `src/ep2/artefatos_reproducao.json`: caminho relativo, SHA-256 de cada fonte/artefato e hashes dos dados externos. `reproduzir_ep2.py conferir` verifica todos os itens, além da cadeia original de `candidato.json`, `congelamento.json` e metadados dos caches. Os manifests científicos originais foram preservados.

| Dependência | Finalidade | Disponibilidade e reconstrução |
| --- | --- | --- |
| `data/raw/train.xlsx` | 20.092 textos/rótulos de treino | Arquivo oficial externo; ignorado pelo Git. SHA conferido contra o candidato. Não sintetizar nem regravar. |
| `data/raw/test1.xlsx` | 900 entradas sem rótulos, parte do cache conjunto | Arquivo oficial do EP1, externo. Não substituir pelo gabarito. |
| `data/raw/test2.xlsx` | 897 entradas e estrutura da planilha final | Arquivo oficial externo; texto e ordem preservados. |
| `data/splits/folds.csv` | Partição histórica por grupos de texto | Versionado e incluído nos artefatos. SHA 910b1e8b445c99f90925dafe4fdb61cfb683a3bdf59bcac831090181af91c74f. Copiar este arquivo; não executar outra divisão. |
| `results/bertimbau_512_oof.csv` | Probabilidades OOF BERT512 | Versionado e incluído nos artefatos; gerar com `src/experiments/bertimbau_oof.py`. O script `bertimbau_length_512_cv.py` não é o gerador desse CSV. |
| `results/bertimbau_duplicate_soft_labels_256_oof.csv` | Probabilidades OOF soft-label | Versionado e incluído; gerar com `bertimbau_duplicate_soft_labels.py`. |
| `results/bertimbau_ordinal_256_oof.csv` | Probabilidades ordinais OOF | Versionado e incluído; gerar com `bertimbau_ordinal_256.py`. O voto ordinal corrigido é calculado pelo ensemble, sem reescrever este OOF. |
| `results/norberto_base_1024_oof.csv` | Probabilidades OOF NorBERTo | Versionado e incluído; gerar com `norberto_base_1024_cv.py --folds 0 1 2 3 4`. Sem o argumento, o padrão executa somente o fold 0. |
| `results/baseline_oof.csv` | Voto LR + TF-IDF e referência da interpretação | Versionado e incluído; o auxiliar `reconstruir_classicos.py baseline` reproduz o procedimento do notebook 03. |
| `results/tfidf_word_char_linearsvc_oof.csv` | Voto SVC palavra/caractere | Versionado e incluído; auxiliar `reconstruir_classicos.py svc`, com C=0,5 fixo. |
| `results/ensemble_majority_5_v3_norberto_oof.csv` | V3, cinco votos, referência e entrada do gate | Versionado e incluído; reconstruir ensemble v2 e depois v3 com os scripts abaixo. |
| `results/ep2/gating_discursivo/*` | Protocolo, ablações, OOF final, calibrações e atributos | Incluídos nos artefatos; ignorados pelo Git. O wrapper recalcula em pasta alternativa e compara as predições. |
| `results/ep2/candidato_incerteza/{modelo,candidato,congelamento}.json` | Gate final e cadeia de integridade | Incluídos no código complementar; ignorados pelo Git. Na modalidade 1, carregar e conferir, sem treinar nem recongelar. |
| `results/ep2/inferencia_v3/entrada_combinada.xlsx` | Entradas conjuntas de teste1+teste2, sem gabarito | Incluída nos artefatos. Reconstrução por `inferir_v3_ep2.py preparar`; uma nova serialização pode mudar seu hash. |
| `results/ep2/inferencia_v3/manifest.json` | Associa dados, scripts, entrada e ordem | Incluído nos artefatos. Criado por `preparar`, nunca reaproveitar um manifest antigo com caches recém-treinados. |
| `results/ep2/inferencia_v3/componentes/*.csv` e `*.meta.json` | Probabilidades/votos dos seis componentes e V3 em 1.797 linhas | Incluídos nos artefatos. `rodar` executa os componentes históricos sequencialmente e registra hashes. Caches contêm saídas intermediárias para teste1, sem gabarito; não são planilhas de submissão. |
| `results/ep2/analise_candidato/*` | Interpretação global/local e histórico tabulado | Incluídos nos artefatos; `analisar_candidato.py` lê também os OOFs de TAPT/moBERTo e atributos discursivos. |
| `results/ep2/{tapt_bertimbau,moberto_512}/fold*_seed42/predicoes_fold.csv` | Recalcular as duas linhas históricas do relatório | Incluídos nos artefatos; novos treinamentos dessas alternativas não são necessários para o método final. |
| `test1-rotulado-para-teste-estatistico.xlsx` | Gabarito só para McNemar | Externo, fornecido pelo professor; ausente dos dois pacotes complementares. |
| `results/ensemble_majority_5_v3_norberto_final.csv` | Predições reais do EP1 para os 900 textos | Já versionado no commit do EP1; obtenção por URL fixa no README. Não confundir com o V3 recalculado no cache do EP2. Ausente dos novos ZIPs. |

Os pacotes não incluem pesos de Transformers. Não há dependência de um checkpoint pessoal para a modalidade 1. O pacote de artefatos contém todos os itens de `artefatos_zip` do manifesto; os três dados brutos são obtidos pelo professor. A publicação dos complementos ainda precisa ser feita pelo grupo.

A referência externa do EP1 possui dois hashes aceitos: a cópia Windows (CRLF) e o arquivo público baixado do GitHub (LF). A auditoria baixou o arquivo e confirmou igualdade após normalização exclusiva dos finais de linha, além de igualdade dos DataFrames. Essa exceção é explícita no manifesto; arquivos científicos e dados oficiais continuam sujeitos aos hashes originais.

Há caminhos absolutos históricos em alguns registros de auditoria, como metadados de resultados anteriores. Eles são informativos. Os módulos usados na reprodução resolvem a raiz a partir de `__file__`; o pacote foi preparado para extração em outra pasta. Os hashes são sensíveis a LF/CRLF: usar as fontes exatas do ZIP, sem normalização automática por editor ou Git. Não enfraquecer a validação para aceitar bytes diferentes.

## 3. Modalidade 1 — execução rápida a partir de artefatos

Seguir os passos numerados do README. A inferência chama `candidato_incerteza.infer()` original; apenas seu diretório de saída é redirecionado em memória. Os pesos, a padronização, os atributos, os hiperparâmetros e a regra permanecem iguais. Os originais nunca são sobrescritos.

O cálculo é: quatro probabilidades neurais + distribuições (0,6/0,2/0,2) para SVC, baseline e V3; 11 atributos de entropia, margem e fração de votos; pesos por softmax; mistura de probabilidades; interceptos de classe e argmax em `c1,c234,c5`. O gate usa todas essas sete entradas; não é apenas a votação V3.

O comando `oof` reexecuta as três variantes nos mesmos cinco folds, dividindo cada fold em duas metades com grupos separados. Ajusta scaler/gate numa metade e avalia na outra, invertendo depois. Nenhum gabarito de teste é lido. Compara os row_ids, folds, metades, predições e métricas com os resultados registrados.

Para testes rápidos adicionais, na raiz da pasta extraída:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s src/ep2 -p test_gating_discursivo.py -v
.\.venv\Scripts\python.exe -m compileall -q src
.\.venv\Scripts\python.exe src/ep2/reproduzir_ep2.py --help
```

Interpretação opcional, a partir dos artefatos completos (escreve novos resumos na cópia de reprodução):

```powershell
.\.venv\Scripts\python.exe src/ep2/analisar_candidato.py
```

Esse script ajusta somente o baseline interpretável em CPU, sem refazer seleção do gate. Caso altere arquivos de análise que constam no manifesto, a verificação estrita posterior poderá apontar diferenças de serialização; mantenha a extração original para validar os bytes da entrega.

## 4. Modalidade 2 — reconstrução integral em cópia separada

**Não executar hoje apenas para entregar.** Estes comandos treinam quatro componentes neurais em cinco folds e novamente em todo o treino. O custo neural completo não foi reexecutado na auditoria.

1. Partindo da modalidade 1 com dados conferidos, prepare uma nova raiz. Este auxiliar copia fontes, locks, dados sem gabarito e o arquivo exato de folds; não copia OOFs, caches nem o candidato congelado:

```powershell
.\.venv\Scripts\python.exe src/ep2/preparar_reconstrucao.py --destino results/ep2/reconstrucao_integral
if ($LASTEXITCODE -ne 0) { throw 'Falha na preparação da cópia.' }
Set-Location .\results\ep2\reconstrucao_integral
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ep2-neural-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Instalação neural falhou; não iniciar treinamento.' }
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe src/ep2/reproduzir_ep2.py conferir-dados
```

2. Prepare os modelos pré-treinados em um cache dedicado, com revisões explícitas. As revisões abaixo foram observadas no cache local em 08/11; os scripts antigos não registraram a revisão de cada treinamento, portanto isso não é prova retrospectiva da identidade de todos os downloads originais. O auxiliar evita usar silenciosamente um `main` futuro:

```powershell
.\.venv\Scripts\python.exe src/ep2/preparar_modelos_hf.py
if ($LASTEXITCODE -ne 0) { throw 'Download de modelos incompleto.' }
$env:HF_HUB_CACHE = Join-Path (Get-Location) 'results\ep2\hf_cache'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe src/ep2/preparar_modelos_hf.py --conferir
```

Revisões: BERTimbau `94d69c95f98f7d5b2a8700c420230ae10def0baa`; NorBERTo `db73446f89c96044863ea05a39f680524b84bccb`. O download desses modelos em cache novo não foi executado nesta auditoria. O acesso externo a Hugging Face e à distribuição de PyTorch é necessário para uma instalação neural nova.

3. Gere os OOFs com os folds fornecidos. Execute os comandos sequencialmente e pare se qualquer um retornar erro:

```powershell
.\.venv\Scripts\python.exe src/ep2/reconstruir_classicos.py baseline
if ($LASTEXITCODE -ne 0) { throw 'Falha no baseline.' }
.\.venv\Scripts\python.exe src/ep2/reconstruir_classicos.py svc
if ($LASTEXITCODE -ne 0) { throw 'Falha no SVC.' }
.\.venv\Scripts\python.exe src/experiments/bertimbau_oof.py
if ($LASTEXITCODE -ne 0) { throw 'Falha no OOF BERT512.' }
.\.venv\Scripts\python.exe src/experiments/bertimbau_duplicate_soft_labels.py
if ($LASTEXITCODE -ne 0) { throw 'Falha no OOF soft-label.' }
.\.venv\Scripts\python.exe src/experiments/bertimbau_ordinal_256.py
if ($LASTEXITCODE -ne 0) { throw 'Falha no OOF ordinal.' }
.\.venv\Scripts\python.exe src/experiments/norberto_base_1024_cv.py --folds 0 1 2 3 4
if ($LASTEXITCODE -ne 0) { throw 'Falha no OOF NorBERTo.' }
.\.venv\Scripts\python.exe src/experiments/ensemble_majority_5_oof_v2.py
if ($LASTEXITCODE -ne 0) { throw 'Falha no ensemble v2.' }
.\.venv\Scripts\python.exe src/experiments/ensemble_majority_5_v3_norberto.py
if ($LASTEXITCODE -ne 0) { throw 'Falha no ensemble v3.' }
```

4. Execute o protocolo de avaliação e ajuste o mesmo gate de incerteza. A escolha dessa variante está fixada pela entrega; não escolher outra com base nos testes. As novas métricas podem diferir do histórico por diferenças numéricas nos treinamentos-base:

```powershell
.\.venv\Scripts\python.exe src/ep2/gating_discursivo.py
if ($LASTEXITCODE -ne 0) { throw 'Falha na avaliação do gate.' }
.\.venv\Scripts\python.exe src/ep2/candidato_incerteza.py preparar
if ($LASTEXITCODE -ne 0) { throw 'Falha no ajuste final.' }
.\.venv\Scripts\python.exe src/ep2/candidato_incerteza.py congelar --confirmar-escolha-final
if ($LASTEXITCODE -ne 0) { throw 'Falha no congelamento da reprodução.' }
```

Esse congelamento é o registro da **nova reprodução**, não uma substituição do congelamento da entrega. Não copiar esses novos arquivos sobre os originais. O hash novo só é válido para as novas fontes; hashes antigos não devem ser reescritos para esconder divergências.

5. Treine os componentes finais em todo o treino e gere suas probabilidades para os 900+897 textos. Em seguida aplique o gate; `separar` não é necessário nesta rota, pois produziria somente o V3:

```powershell
.\.venv\Scripts\python.exe src/ep2/inferir_v3_ep2.py preparar
if ($LASTEXITCODE -ne 0) { throw 'Falha na entrada conjunta.' }
.\.venv\Scripts\python.exe src/ep2/inferir_v3_ep2.py rodar
if ($LASTEXITCODE -ne 0) { throw 'Falha em um componente final.' }
.\.venv\Scripts\python.exe src/ep2/candidato_incerteza.py inferir
if ($LASTEXITCODE -ne 0) { throw 'Falha na inferência do gate.' }
```

`rodar` executa baseline, SVC, BERT512, soft256, ordinal256, NorBERTo e V3, um processo por vez. Só reutiliza um cache quando seus hashes são válidos. A saída final é `results/ep2/candidato_incerteza/final/test2_rotulado_GATE.xlsx`. O exportador confere 897 linhas, duas colunas, rótulos válidos, textos e ordem. O manifesto dessa execução registra hashes. Esta planilha é resultado da reconstrução, não deve substituir a entrega congelada; não é possível prometer os mesmos 897 rótulos sem efetivamente realizar o treinamento e compará-los.

6. Para auditar a reconstrução, usar os hashes e métricas gerados nessa cópia. O wrapper da modalidade 1 é propositalmente estrito com os hashes históricos e não deve ser usado para aceitar silenciosamente novos OOFs. Caso queira McNemar nessa nova execução, registrar separadamente e não usar seus resultados para ajustar o método. A auditoria atual só refez McNemar para a entrega congelada.

## 5. Custo, determinismo e limites

Registros históricos incluídos nos artefatos: soft-label 256 aproximadamente 13,7–13,9 minutos por fold; ordinal 256 aproximadamente 13,1–16,5 minutos por fold; NorBERTo aproximadamente 97–181 minutos por fold, cerca de 11,1 horas nos cinco folds. Esses números são tempos observados nesses CSVs, não orçamento garantido para outro computador. Não há uma estimativa total auditada para todos os treinamentos e downloads.

Seeds históricas: 42; batch efetivo 16 nos quatro componentes neurais; mesmas configurações do relatório. Há operações de GPU cuja reexecução pode variar e os scripts não garantem determinismo bit a bit entre versões/hardware. O guardião de hashes valida a proveniência dos artefatos originais; ele não certifica que um novo treinamento reproduzirá os mesmos bytes.

`analisar_candidato.py` precisa dos OOFs de alternativas históricas e dos atributos, incluídos no pacote. `finalizar_grupo13.py` é um gerador interno que também exige a planilha efetivamente entregue no EP1 em `entrega/test1_resultado_final.xlsx`, `results/ep2/validacao_2026-11-03.json`, os backups V3, a saída original de inferência, análises e métricas. Esses arquivos internos adicionais não integram a rota mínima; não executar esse gerador na pasta limpa para inferir. Os PDFs prontos permanecem corretos. `auditar_penalidades.py` também depende desse ambiente de manutenção. Não prometer que todos os scripts exploratórios antigos funcionam apenas com o lock mínimo.

## 6. Entrega e publicação

Manter `grupo13.zip` com os três itens. Disponibilizar `codigo_grupo13.zip` e `artefatos_reproducao_grupo13.zip` separadamente, com o link acessível ao professor. Os complementos não devem ficar apenas no computador do grupo. Os dados oficiais são obtidos na disciplina; não foram incluídos no Git nem nesses complementos.

O repositório remoto já contém a versão principal do EP2. Faltam as correções da auditoria de 08/11, os três guias, locks e novos auxiliares. Nenhum push foi feito pela auditoria. O `.gitignore` continua excluindo ZIPs, checkpoints, dados brutos e resultados EP2; os artefatos podem ser disponibilizados como anexo externo ou release pelo usuário. Não adicionar pesos grandes ao histórico Git.
