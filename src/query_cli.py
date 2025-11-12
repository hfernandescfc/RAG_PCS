#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple CLI for querying the RAG vector store (hybrid search).

ASCII-only output to avoid encoding issues on Windows consoles.
"""

import os
import sys
import requests
from langchain_community.vectorstores import Chroma
from rag_pipeline.config import DB_PATH, get_embeddings
from rag_pipeline.prompts import PROMPT
from rag_pipeline.retrieval import HybridRetriever as SharedHybridRetriever
from langchain_community.llms import Ollama


def listar_documentos(db: Chroma):
    data = db.get()
    sources = set(m.get("source", "Desconhecido") for m in data.get("metadatas", []))
    return sorted(sources)


def buscar_avancada(query: str, db: Chroma, retriever: SharedHybridRetriever,
                    llm: Ollama, filtro_source: str | None = None,
                    priorizar_palavra: bool = False):
    print("\n" + "=" * 70)
    print("PERGUNTA:", query)
    if filtro_source:
        print("Filtro de documento:", filtro_source)
    print("=" * 70)

    if priorizar_palavra:
        retriever.weight_vector = 0.3
        retriever.weight_bm25 = 0.7
        print("Modo: Priorizar palavra-chave (70%)")
    else:
        retriever.weight_vector = 0.5
        retriever.weight_bm25 = 0.5
        print("Modo: Balanceado (50/50)")

    docs = retriever.search(query, k=6, filter_source=filtro_source)

    print("-" * 70)
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"[{i}] {source} (pag. {page})")
        print(f"    Preview: {preview}...")

    print("\n" + "-" * 70)
    print("Os documentos acima sao relevantes?")
    print("  [s] Sim, gerar resposta")
    print("  [n] Nao, tentar outro filtro")
    print("  [p] Priorizar palavra-chave")
    print("  [Enter] Continuar")

    resp = input(">>> ").strip().lower()
    if resp == "n":
        print("\nDicas:")
        print("  1) Adicione filtro por documento (comando 'filtrar N')")
        print("  2) Reformule a pergunta com termos do documento")
        print("  3) Use busca literal (comando 'buscar TERMO')")
        return
    if resp == "p":
        print("\nRepetindo com prioridade para palavra-chave...")
        return buscar_avancada(query, db, retriever, llm, filtro_source, priorizar_palavra=True)

    # Build LLM context
    context = "\n\n".join([
        f"[Documento: {doc.metadata.get('source')} - Pagina {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc in docs
    ])
    prompt_text = PROMPT.format(context=context, question=query)
    try:
        resposta = llm.invoke(prompt_text)
    except Exception as e:
        print("[ERRO] Falha ao gerar resposta:", e)
        return

    print("=" * 70)
    print("RESPOSTA:")
    print("=" * 70)
    print(resposta)
    print("=" * 70)


def main():
    print("Sistema RAG - Busca Hibrida (CLI)\n")

    # Verify DB path
    db_path = DB_PATH
    if not os.path.exists(db_path):
        print("[ERRO] Banco vetorial nao encontrado: {}".format(db_path))
        sys.exit(1)

    # Embeddings + Vector store
    print("Carregando embeddings e banco...")
    embeddings = get_embeddings()
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print("OK. Banco carregado.\n")

    # Verify Ollama
    print("Verificando Ollama...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            raise Exception()
        print("OK. Ollama respondendo.\n")
    except Exception:
        print("[ERRO] Ollama nao esta rodando. Execute: ollama serve")
        sys.exit(1)

    llm = Ollama(model="llama3.2:3b", temperature=0, num_ctx=4096, num_predict=1024)

    print("Documentos disponiveis:")
    documentos = listar_documentos(db)
    for i, doc in enumerate(documentos, 1):
        print(f"  {i}. {doc}")

    retriever = SharedHybridRetriever(db, weight_vector=0.5, weight_bm25=0.5)
    filtro_ativo = None

    print("\n" + "=" * 70)
    print("SISTEMA RAG - BUSCA HIBRIDA (CLI)")
    print("=" * 70)

    print("\nComandos:")
    print("  filtrar [num]   - buscar apenas no documento [num]")
    print("  buscar [termo]  - busca literal")
    print("  docs            - listar documentos")
    print("  sair            - encerrar")

    while True:
        print("\n" + "=" * 70)
        if filtro_ativo:
            print(f"Filtro ativo: {filtro_ativo}")
        query = input("Pergunta: ").strip()
        if not query:
            continue
        if query.lower() in ["sair", "exit", "quit"]:
            break

        # List docs
        if query.lower() == "docs":
            print("\nDocumentos disponiveis:")
            documentos = listar_documentos(db)
            for i, doc in enumerate(documentos, 1):
                print(f"  {i}. {doc}")
            continue

        # Toggle filter
        if query.lower().startswith("filtrar "):
            try:
                num = int(query.split()[1])
                documentos = listar_documentos(db)
                if 1 <= num <= len(documentos):
                    filtro_ativo = documentos[num - 1]
                    print(f"OK. Filtro ativado: {filtro_ativo}")
                    print("   (Digite 'filtrar off' para desativar)")
                else:
                    print("[ERRO] Numero invalido")
            except Exception:
                if "off" in query.lower():
                    filtro_ativo = None
                    print("OK. Filtro desativado")
                else:
                    print("[ERRO] Uso: filtrar [numero] ou filtrar off")
            continue

        # Literal search
        if query.lower().startswith("buscar "):
            termo = query[7:].strip()
            print(f"\nBuscando termo: '{termo}'")
            data = db.get()
            encontrados = []
            for text, metadata in zip(data.get("documents", []), data.get("metadatas", [])):
                if termo.lower() in (text or "").lower():
                    if filtro_ativo:
                        if metadata.get("source") == filtro_ativo:
                            encontrados.append((text, metadata))
                    else:
                        encontrados.append((text, metadata))
            if encontrados:
                print(f"OK. Encontrado em {len(encontrados)} trechos")
                for i, (text, meta) in enumerate(encontrados[:10], 1):
                    source = meta.get("source", "?")
                    page = meta.get("page", "?")
                    idx = (text or "").lower().find(termo.lower())
                    preview = (text or "")[max(0, idx - 50): idx + 150]
                    print(f"\n[{i}] {source} (pag. {page})")
                    print(f"    ...{preview}...")
            else:
                print("[ERRO] Termo nao encontrado")
            continue

        # Hybrid query
        buscar_avancada(query, db, retriever, llm, filtro_source=filtro_ativo)

    print("\nEncerrado.")


if __name__ == "__main__":
    main()

