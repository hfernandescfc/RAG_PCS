from typing import List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma


class HybridRetriever:
    """Combina busca vetorial (semântica) com BM25 (palavra‑chave)."""

    def __init__(self, vectorstore: Chroma, weight_vector: float = 0.5, weight_bm25: float = 0.5):
        self.vectorstore = vectorstore
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25

        # Carregar todos os documentos para BM25
        all_data = vectorstore.get()
        self.all_docs: List[Document] = []
        for text, metadata in zip(all_data["documents"], all_data["metadatas"]):
            self.all_docs.append(Document(page_content=text, metadata=metadata))

        tokenized_docs = [doc.page_content.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, k: int = 6, filter_source: Optional[str] = None) -> List[Document]:
        # 1) Vetorial
        if filter_source:
            vector_results = self.vectorstore.similarity_search(query, k=k * 2, filter={"source": filter_source})
        else:
            vector_results = self.vectorstore.similarity_search(query, k=k * 2)

        # 2) BM25
        bm25_scores = self.bm25.get_scores(query.lower().split())
        top_bm25_indices = np.argsort(bm25_scores)[::-1][: k * 2]
        bm25_results = [self.all_docs[i] for i in top_bm25_indices]
        if filter_source:
            bm25_results = [d for d in bm25_results if d.metadata.get("source") == filter_source]

        # 3) Combinar e ranquear
        doc_scores = {}
        for i, doc in enumerate(vector_results):
            doc_key = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))
            vector_score = 1.0 - (i / max(1, len(vector_results)))
            doc_scores[doc_key] = {"doc": doc, "vector": vector_score * self.weight_vector, "bm25": 0.0}

        max_bm25 = float(np.max(bm25_scores)) if len(bm25_scores) else 1.0
        if max_bm25 == 0:
            max_bm25 = 1.0
        for i, doc in enumerate(bm25_results):
            doc_key = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("page"))
            bm25_score = bm25_scores[top_bm25_indices[i]] / max_bm25
            if doc_key in doc_scores:
                doc_scores[doc_key]["bm25"] = bm25_score * self.weight_bm25
            else:
                doc_scores[doc_key] = {"doc": doc, "vector": 0.0, "bm25": bm25_score * self.weight_bm25}

        scored = []
        for scores in doc_scores.values():
            final_score = scores["vector"] + scores["bm25"]
            scored.append({"doc": scores["doc"], "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return [x["doc"] for x in scored[:k]]


def listar_documentos(vectorstore: Chroma) -> List[str]:
    data = vectorstore.get()
    sources = set(meta.get("source", "Desconhecido") for meta in data["metadatas"]) if data else set()
    return sorted(sources)

