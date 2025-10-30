import os
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# Defaults via env vars (override-friendly)
DB_PATH = os.getenv("DB_PATH", "./chroma_db_multiplos")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    name = model_name or EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def get_vectorstore(db_path: Optional[str] = None, embeddings: Optional[HuggingFaceEmbeddings] = None) -> Chroma:
    path = db_path or DB_PATH
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(persist_directory=path, embedding_function=embeddings)

