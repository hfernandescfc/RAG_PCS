OCR – Guidelines de Rotulagem (ok vs ruim)

Objetivo: rotular páginas como “ok” ou “ruim” pensando em usabilidade do conteúdo extraído por OCR no RAG.

Definições
- ok: texto legível o suficiente para busca/QA. Pode conter pequenos erros, números, siglas e trechos técnicos.
- ruim: texto majoritariamente danificado ou pouco útil, incluindo casos tabulares quebrados no modo fluxo.

Sinais fortes de “ruim”
- Mojibake/caracteres inválidos recorrentes (�, blocos trocados, encoding quebrado).
- Texto sem espaços ou com palavras excessivamente longas “coladas”.
- Página tabular densa processada como fluxo (metodo=ocr) com colunas embaralhadas.
- Predomínio de ruído: muitos dígitos/ponteiros/pontuação aleatória sem contexto linguístico.
- Vazio ou quase vazio quando deveria haver conteúdo (não confundir com páginas em branco esperadas).

Sinais fracos (avaliar contexto)
- Muitos números: pode ser ok em relatórios financeiros ou legislações numeradas.
- Pouca pontuação: títulos, cabeçalhos e listas podem ter baixa pontuação e ainda serem ok.
- Texto curto: títulos/cabeçalhos podem ser ok se legíveis.

Casos ambíguos (priorize utilidade no RAG)
- Tabela limpa (metodo=ocr_tabela) mas com linhas soltas: se células principais estiverem legíveis, marque ok.
- Imagens com legendas: se legenda legível e coerente com o documento, ok; se só ruído, ruim.

Boas práticas
- Leia o preview inteiro; se necessário, abra o documento para confirmar (quando disponível).
- Seja consistente: se um padrão específico for considerado ruim em um arquivo, mantenha critério para páginas similares.
- Anote dúvidas ou padrões recorrentes; isso ajuda a refinar a heurística/modelo.

