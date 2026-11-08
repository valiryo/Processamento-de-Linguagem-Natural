# Auditoria final do EP2 — Grupo 13 — 08/11/2026

**Atualização documental posterior à auditoria, em 08/11:** a seção 10 do relatório foi ampliada com instruções autocontidas de instalação, inferência, validação, McNemar e reconstrução integral. O relatório passou de seis para nove páginas e o ZIP principal foi atualizado. Apresentação, planilha, modelo e resultados permaneceram iguais. As menções abaixo à preservação integral do ZIP e ao relatório de seis páginas descrevem o estado no encerramento da auditoria inicial. Os registros dessa revisão estão em `results/ep2/revisao_secao10/`; os comandos foram conferidos no PDF e analisados pelo parser PowerShell.

## Parecer

**A reprodução do gate congelado foi executada e aprovada em uma pasta extraída dos pacotes, com ambiente virtual novo e dependências instaladas pelos locks.** As saídas finais de teste1/teste2 e a planilha teste2 coincidiram em SHA-256 com as originais. As três variantes do gate foram recalculadas, produzindo as mesmas predições OOF e métricas. McNemar foi reproduzido, inclusive usando a referência EP1 baixada do GitHub.

**A reconstrução neural integral desde os textos não foi executada nesta auditoria.** O baseline clássico foi reconstruído nos cinco folds e coincidiu nas 20.092 predições. Os treinamentos neurais e o SVC completo permanecem fora da validação de execução desta auditoria; seus scripts, dependências, imports e comandos foram inspecionados. Não há certificação de igualdade bit a bit em novos treinamentos.

O ZIP acadêmico principal está correto e foi preservado byte a byte. Os complementos foram preparados localmente. **A disponibilidade dos complementos e desta revisão para o professor depende de publicação pelo grupo.** Nenhum push ou envio foi realizado.

## Problemas encontrados e correções

| Gravidade | Problema real | Correção / situação |
| --- | --- | --- |
| Alta | O código complementar não continha os OOFs-base, folds e caches necessários à validação estrita/inferência | Criado `artefatos_reproducao_grupo13.zip` com os arquivos reais; inventário e hashes em `src/ep2/artefatos_reproducao.json`. |
| Alta | README misturava preparação/congelamento com uso de um candidato já congelado | Duas modalidades separadas, comandos numerados e saídas alternativas. O wrapper usa o módulo científico original sem refazer seleção. |
| Alta | `inferir_v3_ep2.py` isoladamente entrega V3, não o gate do EP2 | A rota recomendada executa explicitamente `candidato_incerteza.infer()` e valida a saída congelada. A rota integral também termina no gate. |
| Alta | `requirements.txt` amplo, sem versões e incompleto para o EP2 | Locks separados de CPU, documentos e ambiente neural. Instalação mínima/documentos e `pip check` aprovados em venv novo. Ambiente original preservado. |
| Média | Possível uso do script de CV de comprimento 512 como gerador de OOF | Guia indica `bertimbau_oof.py`; NorBERTo recebe explicitamente os cinco folds. |
| Média | OOF do baseline era gerado em notebook; SVC original executa busca em três valores de C | Novo auxiliar reproduz o baseline e fixa SVC em C=0,5 já escolhido, sem nova busca. Baseline executado e conferido. |
| Média | Hash do CSV EP1 público diverge da cópia Windows | Download real confirmou LF contra CRLF, e igualdade dos dados. Manifesto aceita exatamente os dois hashes verificados, sem mudar manifests científicos. |
| Média | Executar inferência no workspace original falhava por saídas já existentes | Wrapper grava em `results/ep2/reproducao`, protege diretórios científicos e valida saídas existentes sem sobrescrever. |
| Média | Gerador antigo poderia restaurar README e pacote incompletos | Gerador passa a usar o README auditado da raiz e o empacotador novo. O conteúdo dos PDFs não foi regenerado. |
| Média | Modelos pré-treinados eram identificados por nomes sem revisão explícita | Revisões observadas no cache registradas; auxiliar opcional prepara cache separado com revisões fixadas. Download novo não executado. Isso não prova retrospectivamente todas as revisões usadas em treinamentos antigos. |
| Alta, pendente | Pacotes locais não equivalem a material acessível ao professor | Publicar código da auditoria e os dois complementos. A auditoria não possui autorização para publicar. |

## Estado do GitHub e dos arquivos

Consulta remota somente de leitura em 08/11/2026 confirmou `main = b7ad5ba63fa6d402a8939aada9fca9f278f3fb41`, igual ao HEAD local. A versão principal do EP2 está publicada. O gerador de relatório já tinha alteração local da revisão anterior; ela foi preservada e recebeu apenas a integração documental desta auditoria.

Os OOFs históricos do EP1 e `data/splits/folds.csv` já são versionados. Os dados brutos, resultados EP2, ZIPs e checkpoints estão ignorados. Os novos locks, guias, manifesto portátil e auxiliares devem entrar no próximo commit. Os arquivos grandes de artefatos permanecem fora do Git, conforme a orientação de manter código no repositório.

Arquivos adicionados: `README_ENTREGA.md`, `REPRODUCAO_EP2.md`, `AUDITORIA_FINAL_EP2.md`, os três `requirements-ep2-*-lock.txt`, e em `src/ep2/`: `reproduzir_ep2.py`, `reconstruir_classicos.py`, `preparar_reconstrucao.py`, `preparar_modelos_hf.py`, `empacotar_reproducao.py`, `artefatos_reproducao.json`.

Arquivos modificados: `requirements.txt` (orientação para os locks), `src/ep2/README.md` (link para a rota final) e `src/ep2/finalizar_grupo13.py` (preservação da documentação auditada e integração do empacotamento, além das mudanças preexistentes da revisão do relatório). Atualizados os pacotes complementares, cópias dos guias e manifesto de entrega em `results/ep2/entrega_grupo13/`.

Sem alterações: arquitetura, hiperparâmetros, folds, rótulos, modelo congelado, fontes científicas protegidas, predições oficiais, PDF/DOCX do relatório, apresentação e planilha acadêmica. A proveniência anterior dos documentos permanece registrada separadamente no manifesto de entrega; hashes dos manifests de seleção/congelamento não foram substituídos.

## Testes executados

| Teste | Resultado | Evidência / alcance |
| --- | --- | --- |
| Instalação CPU + documentos em CPython 3.11.9, venv novo | **Testado e aprovado** | Locks instalados, `pip check` sem conflitos; 22 distribuições no ambiente mínimo com documentos. Torch ausente desse venv, demonstrando que o gate não precisa de GPU. |
| Sintaxe de todos os scripts | **Testado e aprovado** | 73 arquivos Python analisados. |
| Imports dos módulos finais e reconstrução | **Testado e aprovado** | 23 módulos no ambiente histórico; imports CPU/documentos também testados no venv novo. Nenhum treino ou download disparado por esses imports. |
| Interfaces CLI necessárias | **Testado e aprovado** | `--help` do wrapper, candidato, gate, inferência histórica, entrega e novos auxiliares. Scripts sem argparse foram importados, nunca iniciados com um `--help` que pudesse treinar acidentalmente. |
| Integridade e dados em pasta extraída | **Testado e aprovado** | 141 arquivos/dependências conferidos; alinhamento de textos, classes, row_ids, folds, grupos e probabilidades; cadeia congelamento/caches validada. |
| Inferência congelada | **Testado e aprovado** | Aproximadamente 8,7 s na execução medida, incluindo validações. CSVs e XLSX iguais em SHA-256 à entrega. Nenhum gabarito de teste1 fornecido à pasta antes dessa etapa. |
| Ablações OOF | **Testado e aprovado** | Aproximadamente 56,1 s na execução medida; três variantes × cinco folds; 20.092 previsões iguais, com mesmas metades e métricas. |
| McNemar | **Testado e aprovado** | Referência local e arquivo público do commit EP1 testados separadamente; estatística/p-valor iguais. Gabarito copiado somente após inferência e OOF. |
| Baseline a partir dos textos | **Testado e aprovado** | Cópia limpa criada pelo auxiliar; cinco folds executados em aproximadamente 41,8 s; DataFrame OOF igual ao histórico. |
| Cinco testes unitários existentes | **Testado e aprovado** | Gradiente numérico, grupos/cobertura, serialização, bloqueio pré-congelamento e exportação sintética; executados também no ambiente novo. |
| Modelo/cache alterados | **Testado e aprovado** | Alterações artificiais somente nas cópias isoladas foram rejeitadas. Originais preservados; cópias restauradas após cada teste. |
| Destino protegido e repetição | **Testado e aprovado** | Tentativa de escrever no candidato original recusada; segunda inferência apenas validou os arquivos existentes. |
| ZIPs e documentos | **Testado e aprovado** | ZIP principal mantém três itens; seis páginas de relatório, nove slides, dez seções e números consistentes. ZIP principal com hash inalterado. |
| Instalação neural completa em venv novo | **Não testado por custo de instalação/download** | Versões e imports observados no ambiente original. Lock neural não é certificado como uma instalação nova concluída. |
| Treinamentos neurais completos e download em cache novo | **Não testado por custo computacional/download** | Comandos documentados, sem iniciá-los. Arquivos de referência e caches permitem a modalidade congelada. |
| SVC reconstruído nos cinco folds | **Não executado nesta auditoria** | Auxiliar/CLI e configuração fixa inspecionados; artefato histórico validado. |
| Publicação/entrega no prazo | **Dependente do usuário** | Não houve push, upload ou envio. |
| Apresentação oral ≤10 minutos | **Dependente de ensaio pelo grupo** | Roteiro soma 510 segundos; duração de fala real não foi medida. |

Os comandos de teste que retornaram 1 ao adulterar cópias são testes de rejeição aprovados, não falhas abertas. Uma primeira tentativa de instalação sem rede foi bloqueada pelo sandbox; a execução com acesso à rede permitido concluiu a instalação. Não há dependência ausente bloqueando a modalidade 1 testada.

As evidências estão em `results/ep2/auditoria_2026-11-08/` no workspace e os JSONs/logs selecionados acompanham o código em `auditoria_2026-11-08/`. Eles distinguem ambiente, inventário, testes, durações e códigos de retorno; não incluem gabarito nem CSV final individual de teste1.

## Resultados preservados

| Métrica | EP1 / V3 | EP2 / incerteza |
| --- | --- | --- |
| Acurácia OOF global | 47,5662% | 47,7105% |
| Acertos OOF | 9.557/20.092 | 9.586/20.092 |
| Acertos teste1 | 434/900 | 438/900 |
| Acurácia teste1 | 48,2222% | 48,6667% |

McNemar: ambos corretos 397; apenas EP1 37; apenas EP2 41; ambos errados 425. Estatística exata min(b,c)=37; p bicaudal=0,7343420330298468. Não há superioridade significativa a 5%. O ganho sobre mistura constante continua sendo apenas cinco acertos OOF; o resultado discursivo inferior não foi ocultado. A escolha da ablação de incerteza continua declarada como exploratória.

Teste2: 897 linhas, duas colunas `resp_text`/`clarity`, distribuição 296 c1, 273 c234, 328 c5, textos e ordem idênticos ao original. Acurácia desconhecida. A auditoria comparou a reprodução com as predições congeladas para verificar identidade, não com rótulos verdadeiros nem para selecionar um modelo.

## Checklist final

| Item | Parecer |
| --- | --- |
| Professor consegue instalar? | **Sim, rota CPU/documentos testada**, com Python 3.11.9 x64 e acesso aos pacotes; rota neural completa ainda não testada em instalação nova. |
| Consegue validar dados? | **Sim**, após obter os arquivos oficiais exatos e os dois complementos. |
| Consegue reconstruir o método? | **Gate e ablações: sim, executados. Pipeline neural desde textos: comandos fornecidos, execução completa não validada nesta auditoria.** |
| Consegue executar a inferência? | **Sim**, sobre os caches originais; testado em venv sem Torch. |
| Consegue gerar as 897 predições? | **Sim**, XLSX e CSV reproduzidos com os mesmos hashes. |
| Consegue reproduzir métricas de validação? | **Sim**, três variantes OOF reexecutadas sobre OOFs-base fornecidos. |
| Consegue verificar McNemar? | **Sim**, com gabarito oficial e referência pública EP1, ambos verificados por hash. |
| Os três arquivos acadêmicos estão corretos? | **Sim**, estrutura e consistência conferidas; mantidos sem mudanças. |
| Todos os artefatos necessários estão disponíveis? | **Localmente, sim para a rota congelada; remotamente, os complementos e a revisão da auditoria ainda precisam ser disponibilizados pelo usuário.** Dados oficiais são externos; checkpoints neurais não são necessários nessa rota. |

## Ações manuais antes da entrega

1. Publicar o commit com os guias, locks, manifesto e auxiliares desta auditoria. Não adicionar ZIPs, dados ou pesos grandes ao histórico Git.
2. Disponibilizar os dois complementos `codigo_grupo13.zip` e `artefatos_reproducao_grupo13.zip` por um link acessível ao professor, separado do ZIP acadêmico. O relatório aponta o repositório-base; usar o README publicado para localizar os complementos.
3. Enviar o `grupo13.zip` principal, pelo mesmo responsável da entrega EP1, até o prazo de hoje, 08/11/2026.
4. Confirmar que o link externo funciona para outra pessoa e ensaiar a apresentação de nove slides para ficar dentro de dez minutos.

Não falta um treinamento para enviar a planilha já congelada. A pendência concreta é a disponibilização do código revisado e dos artefatos, além do envio acadêmico pelo grupo. Não declarar reprodução neural integral validada.
