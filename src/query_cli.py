import os
import sys
import requests
from typing import List
from langchain_community.vectorstores import Chroma
from rag_pipeline.config import DB_PATH, get_embeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from rag_pipeline.prompts import PROMPT
from rag_pipeline.retrieval import HybridRetriever as SharedHybridRetriever
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
import numpy as np

print("ï¿½Y"ï¿½ Sistema RAG com Busca Hï¿½ï¿½brida Avanï¿½ï¿½ada\n")

# Verificar banco
db_path = DB_PATH
if not os.path.exists(db_path):
    print("ï¿½?O Banco nÇœo encontrado!")
    sys.exit(1)

# Carregar embeddings
print("ï¿½Y'ï¿½ Carregando sistema...")
embeddings = get_embeddings()

db = Chroma(persist_directory=db_path, embedding_function=embeddings)
print("ï¿½o. Banco carregado")

# Verificar Ollama
print("ï¿½Yï¿½- Verificando Ollama...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=2)
    if response.status_code != 200:
        raise Exception()
    print("ï¿½o. Ollama OK\n")
except:
    print("ï¿½?O Ollama nÇœo estÇ­ rodando!")
    sys.exit(1)

llm = Ollama(model="llama3.2:3b", temperature=0, num_ctx=4096, num_predict=1024)

# ============================================================================
# BUSCA Hï¿½?BRIDA: Vetorial + BM25
# ============================================================================

# ============================================================================
# LISTAR DOCUMENTOS DISPONï¿½?VEIS
# ============================================================================

def listar_documentos():
    """Lista todos os documentos disponÃ­veis"""
    all_data = db.get()
    sources = set(meta.get('source', 'Desconhecido') for meta in all_data['metadatas'])
    return sorted(sources)

documentos_disponiveis = listar_documentos()

print("ï¿½Y"" Documentos disponÃ­veis no sistema:")
for i, doc in enumerate(documentos_disponiveis, 1):
    print(f"   {i}. {doc}")

# ============================================================================
# PROMPT OTIMIZADO
# ============================================================================

# PROMPT importado de rag_pipeline.prompts

# ============================================================================
# FUNÃ‡ÃƒO DE BUSCA AVANÃ‡ADA
# ============================================================================

def buscar_avancada(query: str, filtrar_por_documento: str = None, 
                   ajustar_pesos: bool = False):
    """
    Busca avanÃ§ada com opÃ§Ãµes de filtro e ajuste
    
    Args:
        query: Pergunta
        filtrar_por_documento: Nome do arquivo (ex: "contrato.pdf")
        ajustar_pesos: Se True, prioriza BM25 (palavra-chave)
    """
    
    print(f"\n{'='*70}")
    print(f"ï¿½?" PERGUNTA: {query}")
    if filtrar_por_documento:
        print(f"ï¿½YZï¿½ Filtro: {filtrar_por_documento}")
    print('='*70)
    
    # Ajustar pesos se solicitado
    if ajustar_pesos:
        hybrid_retriever.weight_vector = 0.3
        hybrid_retriever.weight_bm25 = 0.7
        print("ï¿½s-ï¿½ï¿½?  Modo: Priorizando palavras-chave (70%)")
    else:
        hybrid_retriever.weight_vector = 0.5
        hybrid_retriever.weight_bm25 = 0.5
        print("ï¿½s-ï¿½ï¿½?  Modo: Balanceado (50% semÇ½ntica / 50% palavra-chave)")
    
    # Buscar documentos
    print("\nï¿½Y"? Buscando documentos relevantes...\n")
    
    scored_docs = hybrid_retriever.search(
        query,
        k=6,
        filter_source=filtrar_por_documento
    )
    
    # Mostrar resultados
    print("-"*70)
    for i, item in enumerate(scored_docs, 1):
        doc = item['doc'] if isinstance(item, dict) else item
        source = doc.metadata.get('source', '?')
        page = doc.metadata.get('page', '?')
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"[{i}] {source} (pÇ­g. {page})")
        print(f"    Preview: {preview}...")
    
    print("\n" + "-"*70)
    
    # ValidaÃ§Ã£o do usuÃ¡rio
    print("\nï¿½Y'ï¿½ Os documentos acima sÇœo relevantes?")
    print("   [s] Sim, gerar resposta")
    print("   [n] NÇœo, tentar com outro filtro")
    print("   [p] Priorizar palavras-chave")
    print("   [Enter] Continuar")
    
    resp = input(">>> ").strip().lower()
    
    if resp == 'n':
        print("\nï¿½Y'ï¿½ Experimente:")
        print("   1. Adicionar filtro por documento")
        print("   2. Reformular pergunta com termos do documento")
        print("   3. Usar busca literal (comando 'buscar')")
        return
    
    if resp == 'p':
        print("\nï¿½Y"" Repetindo busca priorizando palavras-chave...")
        return buscar_avancada(query, filtrar_por_documento, ajustar_pesos=True)
    
    # Gerar resposta
    print("\nï¿½Yï¿½- Gerando resposta (aguarde 10-30s)...\n")
    
    try:
        # Preparar contexto
        docs_for_llm = [item['doc'] if isinstance(item, dict) else item for item in scored_docs]
        context = "\n\n".join([
            f"[Documento: {doc.metadata.get('source')} - PÇ­gina {doc.metadata.get('page')}]\n{doc.page_content}"
            for doc in docs_for_llm
        ])
        
        # Gerar resposta
        prompt_text = PROMPT.format(context=context, question=query)
        resposta = llm.invoke(prompt_text)
        
        print("="*70)
        print("ï¿½Y"? RESPOSTA:")
        print("="*70)
        print(resposta)
        print("="*70)
        
        print("\nï¿½Y"s Fontes utilizadas (com scores):")
        for i, item in enumerate(scored_docs, 1):
            doc = item['doc'] if isinstance(item, dict) else item
            source = doc.metadata.get('source', '?')
            page = doc.metadata.get('page', '?')
            score = item.get('score') if isinstance(item, dict) else None
            if score is not None:
                print(f"[{i}] {source} (pÇ­g. {page}) - RelevÇ½ncia: {score:.3f}")
            else:
                print(f"[{i}] {source} (pÇ­g. {page})")
        
    except Exception as e:
        print(f"ï¿½?O Erro: {e}")

# ============================================================================
# INTERFACE INTERATIVA
# ============================================================================

print("\n" + "="*70)
print("ï¿½Y'ï¿½ SISTEMA RAG COM BUSCA Hï¿½?BRIDA")
print("="*70)

print("\nï¿½YZï¿½ Comandos especiais:")
print("   ï¿½?ï¿½ 'filtrar [num]' - Buscar apenas no documento [num]")
print("   ï¿½?ï¿½ 'buscar [termo]' - Busca literal")
print("   ï¿½?ï¿½ 'docs' - Listar documentos")
print("   ï¿½?ï¿½ 'sair' - Encerrar")

filtro_ativo = None

while True:
    print("\n" + "="*70)
    
    if filtro_ativo:
        print(f"ï¿½YZï¿½ Filtro ativo: {filtro_ativo}")
    
    query = input("ï¿½?" Pergunta: ").strip()
    
    if not query:
        continue
    
    if query.lower() in ['sair', 'exit', 'quit']:
        break
    
    # Comando: Listar documentos
    if query.lower() == 'docs':
        print("\nï¿½Y"" Documentos disponï¿½ï¿½veis:")
        documentos_disponiveis = listar_documentos()
        for i, doc in enumerate(documentos_disponiveis, 1):
            print(f"   {i}. {doc}")
        continue
    
    # Comando: Ativar filtro
    if query.lower().startswith('filtrar '):
        try:
            num = int(query.split()[1])
            if 1 <= num <= len(documentos_disponiveis):
                filtro_ativo = documentos_disponiveis[num-1]
                print(f"ï¿½o. Filtro ativado: {filtro_ativo}")
                print("   (Digite 'filtrar off' para desativar)")
            else:
                print("ï¿½?O NÇ§mero invÇ­lido")
        except:
            if 'off' in query.lower():
                filtro_ativo = None
                print("ï¿½o. Filtro desativado")
            else:
                print("ï¿½?O Uso: filtrar [nÇ§mero] ou filtrar off")
        continue
    
    # Comando: Busca literal
    if query.lower().startswith('buscar '):
        termo = query[7:].strip()
        print(f"\nï¿½Y"Z Buscando termo: '{termo}'")
        
        all_data = db.get()
        encontrados = []
        
        for text, metadata in zip(all_data['documents'], all_data['metadatas']):
            if termo.lower() in text.lower():
                if filtro_ativo:
                    if metadata.get('source') == filtro_ativo:
                        encontrados.append((text, metadata))
                else:
                    encontrados.append((text, metadata))
        
        if encontrados:
            print(f"ï¿½o. Encontrado em {len(encontrados)} trechos")
            for i, (text, meta) in enumerate(encontrados[:10], 1):
                source = meta.get('source', '?')
                page = meta.get('page', '?')
                idx = text.lower().find(termo.lower())
                preview = text[max(0, idx-50):idx+150]
                print(f"\n[{i}] {source} (pÇ­g. {page})")
                print(f"    ...{preview}...")
        else:
            print("ï¿½?O Termo nÇœo encontrado")
        continue
    
    # Busca normal
    buscar_avancada(query, filtrar_por_documento=filtro_ativo)

print("\nï¿½o. Encerrado!")

