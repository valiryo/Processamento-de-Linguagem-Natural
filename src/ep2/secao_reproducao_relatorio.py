"""Instruções autocontidas de execução para a seção 10 do relatório final."""


def build_reproduction_section():
    blocks = []
    def p(text): blocks.append(('paragraph', text))
    def h(text): blocks.append(('subheading', text))
    def code(text): blocks.append(('code', text.strip('\n')))
    def table(headers, rows): blocks.append(('table', (headers, rows)))

    p('Repositório: https://github.com/valiryo/Processamento-de-Linguagem-Natural. Esta seção contém o procedimento de instalação, execução e verificação. Os arquivos README_ENTREGA.md, REPRODUCAO_EP2.md e AUDITORIA_FINAL_EP2.md, incluídos no pacote de código, detalham a auditoria. A reprodução congelada abaixo foi executada em uma pasta isolada e em um ambiente virtual novo; a reconstrução neural integral está documentada, mas não foi reexecutada nessa auditoria.')
    h('10.1. Pacotes, dados e ambiente')
    p('São necessários codigo_grupo13.zip e artefatos_reproducao_grupo13.zip, disponibilizados pelo grupo separadamente de grupo13.zip. O primeiro contém fontes, dependências, modelo do gate e congelamento; o segundo contém os folds originais, OOFs e caches intermediários. Ambos devem ser extraídos na mesma raiz. O ZIP acadêmico grupo13.zip contém somente relatório PDF, apresentação PDF e planilha teste2; ele não é um pacote executável.')
    table(['Arquivo oficial em data/raw/', 'Uso'], [
        ['train.xlsx', '20.092 textos/rótulos; validação da procedência e reconstrução do treino.'],
        ['test1.xlsx', '900 textos sem rótulos; entrada da inferência conjunta original.'],
        ['test2.xlsx', '897 textos; entrada e estrutura da planilha final.'],
        ['test1-rotulado-para-teste-estatistico.xlsx', 'Somente para McNemar, depois de gerar as predições.']])
    p('Obter os dados oficiais nos materiais da disciplina. Não substituir test1.xlsx pelo gabarito nem abrir e salvar novamente as planilhas: isso pode mudar os hashes. O arquivo data/splits/folds.csv está no pacote de artefatos e deve ser preservado; não criar novas partições. Os dados brutos e os checkpoints neurais não acompanham os complementos. As fontes do ZIP preservam os finais de linha necessários aos hashes históricos.')
    p('Ambiente especificado: Windows x64, PowerShell e Python 3.11.9 x64 com o comando py disponível. A modalidade congelada usa CPU e não precisa de PyTorch ou GPU. Instalar somente requirements-ep2-cpu-lock.txt para essa modalidade. O lock fixa também dependências transitivas; requirements.txt é uma lista histórica ampla e não é o instalador recomendado do EP2.')
    table(['Dependências / finalidade', 'Versões registradas'], [
        ['CPU: NumPy, pandas, SciPy', '2.4.6; 3.0.6; 1.17.1'],
        ['CPU: scikit-learn, openpyxl, threadpoolctl', '1.9.1; 3.1.5; 3.7.0'],
        ['Neural: PyTorch, Transformers', '2.14.0+cu130; 5.17.0'],
        ['Neural: datasets, accelerate, sentencepiece', '5.0.1; 1.15.0; 0.2.2'],
        ['Documentos: ReportLab, python-docx, pypdf', '5.1.0; 1.2.0; 6.21.4']])
    p('requirements-ep2-docs-lock.txt acrescenta as bibliotecas de documentos; elas não são necessárias à inferência. requirements-ep2-neural-lock.txt registra o ambiente para reconstrução neural. statsmodels e eli5 não são dependências: McNemar usa a distribuição binomial exata e a interpretação lexical é calculada diretamente com o baseline. Não foi estabelecido um mínimo certificado de RAM.')

    h('10.2. Modalidade 1 — reprodução congelada, testada em CPU')
    p('Passo 1 — Em uma pasta contendo os dois ZIPs complementares, abrir PowerShell e executar o bloco abaixo. A pasta ep2_reproducao deve ser nova. Na mesma sessão, a função ExecutarPythonEP2 executa o Python do ambiente criado e interrompe o comando se houver erro; ela não altera o modelo.')
    code(r'''
$ErrorActionPreference = 'Stop'
if (Test-Path .\ep2_reproducao) { throw 'Use uma pasta nova.' }
Expand-Archive .\codigo_grupo13.zip -DestinationPath .\ep2_reproducao
Expand-Archive .\artefatos_reproducao_grupo13.zip -DestinationPath .\ep2_reproducao
Set-Location .\ep2_reproducao
New-Item -ItemType Directory -Force .\data\raw | Out-Null
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar o ambiente.' }
function ExecutarPythonEP2 {
    & .\.venv\Scripts\python.exe @args
    if ($LASTEXITCODE -ne 0) { throw 'Execucao interrompida; confira o erro.' }
}
ExecutarPythonEP2 -m pip install -r requirements-ep2-cpu-lock.txt
ExecutarPythonEP2 -m pip check
''')
    p('Passo 2 — Copiar manualmente train.xlsx, test1.xlsx sem rótulos e test2.xlsx para a pasta data/raw criada. Não fornecer ainda o gabarito ao processo de inferência. Executar, a partir da raiz ep2_reproducao:')
    code('''
ExecutarPythonEP2 src/ep2/reproduzir_ep2.py conferir
ExecutarPythonEP2 src/ep2/reproduzir_ep2.py inferir
ExecutarPythonEP2 src/ep2/reproduzir_ep2.py validar-saida
''')
    p('O primeiro comando verifica arquivos, hashes, grupos, folds, ordenação, classes, probabilidades, modelo e congelamento. O segundo executa candidato_incerteza.infer() original sobre os caches: apenas o diretório de saída muda. Não executa preparar ou congelar, não treina Transformers e não refaz seleção. Se as saídas já existirem, compara sua integridade e não as sobrescreve. Hash divergente é erro a investigar, não autorização para modificar o manifesto científico.')
    p('A planilha gerada fica em results/ep2/reproducao/final/test2_rotulado_GATE.xlsx. Nessa pasta também ficam os CSVs locais e o manifesto de saída. Resultado esperado: 897 linhas, colunas resp_text e clarity, 296 c1, 273 c234 e 328 c5, na ordem original. O programa exige igualdade de SHA-256 dos CSVs e do XLSX com as saídas congeladas. O teste1 gerado é apenas material local para avaliação estatística; não deve ser enviado como planilha de submissão.')
    p('Passo 3 — Reproduzir as métricas de desenvolvimento e os testes rápidos. O comando oof reajusta os três gates nas mesmas metades de calibração/avaliação, a partir dos OOFs-base fornecidos, e compara todas as predições com o histórico:')
    code('''
ExecutarPythonEP2 src/ep2/reproduzir_ep2.py oof
ExecutarPythonEP2 -m unittest discover -s src/ep2 -p test_gating_discursivo.py -v
''')
    p('Resultados OOF em results/ep2/reproducao/oof/: V3 47,5662%; constante 47,6856%; incerteza 47,7105%; discurso 47,4020%. São acurácias globais sobre 20.092 linhas. Para repetir integralmente o cálculo, usar uma pasta nova, por exemplo acrescentando --saida results/ep2/reproducao_2 ao comando oof. Uma saída OOF já existente é recusada para evitar uma retomada confundida com recálculo.')
    p('Passo 4 — Somente após gerar as predições, copiar o gabarito oficial para data/raw/test1-rotulado-para-teste-estatistico.xlsx. Obter as predições reais do EP1 no commit histórico pelo bloco abaixo. A variável de URL é montada em partes apenas para manter o comando legível:')
    code(r'''
$baseEP1 = 'https://raw.githubusercontent.com/valiryo/'
$repoEP1 = 'Processamento-de-Linguagem-Natural/'
$commitEP1 = '9461ced46632d90ebf185a5bbd35b6f166247248/'
$arquivoEP1 = 'results/ensemble_majority_5_v3_norberto_final.csv'
$urlEP1 = $baseEP1 + $repoEP1 + $commitEP1 + $arquivoEP1
Invoke-WebRequest -Uri $urlEP1 -OutFile .\results\ep2\ep1_referencia_publica.csv
ExecutarPythonEP2 src/ep2/reproduzir_ep2.py mcnemar `
    --ep1 results/ep2/ep1_referencia_publica.csv
''')
    p('Saída: results/ep2/reproducao/mcnemar_reproduzido.json. Esperam-se 434/900 acertos no EP1 e 438/900 no EP2, b=37, c=41, estatística exata min(b,c)=37 e p=0,7343420330. O programa compara o resultado com o registro original. O CSV público do EP1 usa LF; a cópia histórica Windows usa CRLF. Os dois hashes exatos foram verificados como os mesmos dados e estão explicitamente aceitos no manifesto. Não há superioridade significativa; nenhuma escolha é reaberta.')
    p('Passo 5 — Opcionalmente, recalcular a interpretação global/local do baseline na cópia extraída. O pacote de artefatos fornece os OOFs adicionais de TAPT/moBERTo, os atributos discursivos e as referências exigidas. Esse comando reajusta apenas o classificador auxiliar em CPU e escreve em results/ep2/analise_candidato:')
    code('ExecutarPythonEP2 src/ep2/analisar_candidato.py')
    p('Manter uma extração original se for comparar novamente todos os hashes: recalcular arquivos de análise pode alterar sua serialização. O gerador interno finalizar_grupo13.py não faz parte da execução mínima; ele exige também backups e materiais históricos da entrega. Os PDFs já acompanham grupo13.zip.')

    h('10.3. Modalidade 2 — reconstrução integral desde os textos')
    p('Esta modalidade reproduz o procedimento de treinamento, não uma garantia de resultados idênticos após novos treinamentos. Exige downloads e muitas horas de GPU; não foi executada integralmente na auditoria final. Partir da raiz extraída da modalidade 1, com os três dados oficiais conferidos, e conservar os pacotes originais. A função ExecutarPythonEP2 da seção anterior deve estar definida na mesma sessão.')
    p('Passo 1 — Criar uma cópia nova contendo apenas fontes, locks, dados sem gabarito e o folds.csv original. O auxiliar não copia OOFs, caches ou candidato congelado. Assim, os comandos de ajuste/congelamento abaixo não colidem com o modelo já entregue:')
    code(r'''
ExecutarPythonEP2 src/ep2/preparar_reconstrucao.py `
    --destino results/ep2/reconstrucao_integral
Set-Location .\results\ep2\reconstrucao_integral
py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Falha ao criar o ambiente neural.' }
ExecutarPythonEP2 -m pip install -r requirements-ep2-neural-lock.txt
ExecutarPythonEP2 -m pip check
ExecutarPythonEP2 -c "import torch; assert torch.cuda.is_available()"
ExecutarPythonEP2 src/ep2/reproduzir_ep2.py conferir-dados
''')
    p('Ambiente neural observado: RTX 3050 Laptop com 4 GB, driver 591.59 e runtime CUDA 13.0 do PyTorch. Os scripts usam FP16 e exigem CUDA; CPU não substitui essa rota sem modificações. As configurações da seção 5 são preservadas. A instalação neural completa em ambiente novo não foi validada nesta auditoria; interromper em caso de erro de instalação ou falta de CUDA.')
    p('Passo 2 — Preparar os modelos pré-treinados em cache separado. O auxiliar fixa as revisões observadas no cache local: BERTimbau 94d69c95f98f7d5b2a8700c420230ae10def0baa e NorBERTo db73446f89c96044863ea05a39f680524b84bccb. Os scripts históricos não registraram cada revisão baixada; esses registros atuais não provam retrospectivamente a identidade de todos os downloads antigos. O download em cache novo não foi repetido na auditoria.')
    code(r'''
ExecutarPythonEP2 src/ep2/preparar_modelos_hf.py
$env:HF_HUB_CACHE = Join-Path (Get-Location) 'results\ep2\hf_cache'
$env:HF_HUB_OFFLINE = '1'
ExecutarPythonEP2 src/ep2/preparar_modelos_hf.py --conferir
''')
    p('Passo 3 — Gerar os OOFs e a referência V3 em sequência, sempre preservando os cinco folds originais. O auxiliar clássico usa C=0,5 já escolhido; não executa nova busca. bertimbau_oof.py gera o OOF BERT512; norberto_base_1024_cv.py recebe explicitamente todos os folds porque seu padrão é somente o fold 0:')
    code('''
ExecutarPythonEP2 src/ep2/reconstruir_classicos.py baseline
ExecutarPythonEP2 src/ep2/reconstruir_classicos.py svc
ExecutarPythonEP2 src/experiments/bertimbau_oof.py
ExecutarPythonEP2 src/experiments/bertimbau_duplicate_soft_labels.py
ExecutarPythonEP2 src/experiments/bertimbau_ordinal_256.py
ExecutarPythonEP2 src/experiments/norberto_base_1024_cv.py --folds 0 1 2 3 4
ExecutarPythonEP2 src/experiments/ensemble_majority_5_oof_v2.py
ExecutarPythonEP2 src/experiments/ensemble_majority_5_v3_norberto.py
''')
    p('As saídas principais ficam em results/: baseline_oof.csv, tfidf_word_char_linearsvc_oof.csv, bertimbau_512_oof.csv, bertimbau_duplicate_soft_labels_256_oof.csv, bertimbau_ordinal_256_oof.csv, norberto_base_1024_oof.csv e ensemble_majority_5_v3_norberto_oof.csv. Cada OOF mantém os identificadores e folds do treino. Os scripts neurais também gravam históricos e saídas por fold; não combinar arquivos de execuções diferentes.')
    p('Passo 4 — Executar as ablações e ajustar o mesmo gate de incerteza. A variante é a escolha fixa da entrega, e não uma nova seleção por teste. Preparar e congelar são executados somente nesta cópia sem candidato anterior:')
    code(r'''
ExecutarPythonEP2 src/ep2/gating_discursivo.py
ExecutarPythonEP2 src/ep2/candidato_incerteza.py preparar
ExecutarPythonEP2 src/ep2/candidato_incerteza.py congelar --confirmar-escolha-final
''')
    p('A avaliação fica em results/ep2/gating_discursivo/ e o modelo, candidato e congelamento em results/ep2/candidato_incerteza/. O novo congelamento pertence à reprodução e não substitui o original. Não reescrever hashes antigos para aceitar novos OOFs.')
    p('Passo 5 — Treinar os componentes finais em todo o treino e gerar as probabilidades para os 900+897 textos. O comando rodar executa os seis componentes e a regra V3 em processos sequenciais. A etapa inferir do candidato aplica o gate do EP2; separar não é necessário, pois produziria apenas saídas do V3:')
    code('''
ExecutarPythonEP2 src/ep2/inferir_v3_ep2.py preparar
ExecutarPythonEP2 src/ep2/inferir_v3_ep2.py rodar
ExecutarPythonEP2 src/ep2/candidato_incerteza.py inferir
''')
    p('Saída da reconstrução: results/ep2/candidato_incerteza/final/test2_rotulado_GATE.xlsx. O exportador verifica 897 linhas, duas colunas, rótulos, textos e ordem, e registra hashes no manifest.json dessa execução. Novos treinamentos podem produzir outras probabilidades ou rótulos; não substituir a entrega congelada por essa planilha. O wrapper da modalidade 1 compara com os hashes históricos e não deve ser usado para aceitar silenciosamente artefatos reconstruídos diferentes.')

    h('10.4. Validações, custos observados e limites')
    p('A simulação em ambiente novo aprovou instalação CPU/documentos, integridade, inferência congelada, cinco testes unitários e recálculo das três ablações. Os CSVs e o XLSX coincidiram em SHA-256 com os originais; as 20.092 predições OOF e métricas também coincidiram. McNemar foi refeito com a referência pública do EP1. O baseline foi reconstruído nos cinco folds, com as mesmas predições. Arquivos alterados artificialmente na cópia de teste foram rejeitados. A auditoria não reexecutou todos os treinamentos neurais nem o treinamento completo do SVC.')
    p('Tempos medidos na simulação: aproximadamente 8,7 s para inferência com validações, 56,1 s para as três ablações OOF e 41,8 s para reconstruir o baseline. Registros neurais históricos: soft-label 13,7–13,9 min/fold, ordinal 13,1–16,5 min/fold e NorBERTo 97–181 min/fold (cerca de 11,1 h nos cinco folds). Esses valores são observações, não garantias de duração em outra máquina; não há estimativa total auditada de downloads e treinamentos.')
    p('Seeds e hiperparâmetros iguais não asseguram igualdade bit a bit entre versões ou GPUs. A modalidade 1 reproduz os números da entrega porque preserva os artefatos originais. Os hashes e caminhos estão em src/ep2/artefatos_reproducao.json, candidato.json, congelamento.json e manifest_entrega.json. Os complementos devem estar acessíveis ao professor; sua existência no computador do grupo não equivale a publicação. O teste1 continua reservado à avaliação estatística e o teste2 somente à inferência, com acurácia desconhecida.')
    return blocks
