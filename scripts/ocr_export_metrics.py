#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta métricas de qualidade de OCR por página a partir do banco vetorial (Chroma).

Gera um CSV com as métricas produzidas por src.diagnostico_ocr._metrics e o score
de qualidade calculado por src.diagnostico_ocr._score_quality, além de metadados
relevantes por documento/página.

Uso:
  python scripts/ocr_export_metrics.py --out ocr_metrics.csv [--source NOME] [--contains SUBSTR]
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Garante que o layout src/ seja importável quando executado da raiz do projeto
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_pipeline.config import DB_PATH, get_embeddings
from langchain_community.vectorstores import Chroma

# Reutiliza as funções internas do módulo de diagnóstico
from diagnostico_ocr import _metrics, _score_quality  # type: ignore


def iter_chroma(vs: Chroma) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """Itera sobre (id, document, metadata) do Chroma (sem assumir filtros do backend)."""
    coll = vs._collection  # pyright: ignore [reportPrivateUsage]
    # 'ids' é sempre retornado por ChromaDB e não deve constar em include
    data = coll.get(include=["metadatas", "documents"])  # retorna tudo
    ids = data.get("ids") or []
    docs = data.get("documents") or []
    metas = data.get("metadatas") or []
    for i, doc, meta in zip(ids, docs, metas):
        yield str(i), doc or "", meta or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta métricas de OCR do Chroma para CSV")
    parser.add_argument("--out", default="ocr_metrics.csv", help="Arquivo CSV de saída")
    parser.add_argument("--source", help="Filtro: metadata.source exato")
    parser.add_argument("--contains", help="Filtro: substring em metadata.source (case-insensitive)")
    parser.add_argument("--preview", type=int, default=240, help="Tamanho do preview de texto")
    parser.add_argument(
        "--only-metodo",
        nargs="*",
        default=["ocr", "ocr_tabela"],
        help="Quais metodos incluir (padrão: ocr, ocr_tabela)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    embeddings = get_embeddings()
    vs = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    rows: List[Dict[str, Any]] = []
    for _id, doc, meta in iter_chroma(vs):
        source = (meta or {}).get("source")
        metodo = (meta or {}).get("metodo")
        if args.source and source != args.source:
            continue
        if args.contains and (not source or args.contains.lower() not in str(source).lower()):
            continue
        if args.only_metodo and metodo not in set(args.only_metodo):
            continue

        m = _metrics(doc or "", meta)
        score = _score_quality(m)
        preview = (doc or "").replace("\r", " ").replace("\n", " ")
        if len(preview) > args.preview:
            preview = preview[: args.preview] + "…"

        row = {
            "id": _id,
            "source": source,
            "metodo": metodo,
            "grid_score": (meta or {}).get("grid_score"),
            "page": (meta or {}).get("page"),
            **m,
            "score": score,
            "text_preview": preview,
        }
        rows.append(row)

    if not rows:
        print("Nenhum documento encontrado com os filtros fornecidos.")
        return

    # Ordena por score crescente (mais problemáticos primeiro)
    rows.sort(key=lambda r: (r.get("score", 0.0), r.get("source") or ""))

    fieldnames = [
        "id",
        "source",
        "metodo",
        "page",
        "grid_score",
        "chars",
        "words",
        "alpha_ratio",
        "digit_ratio",
        "space_ratio",
        "punct_ratio",
        "avg_word_len",
        "suspicious_count",
        "long_word_ratio",
        "vowel_ratio",
        "stop_ratio",
        "score",
        "text_preview",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Exportadas {len(rows)} linhas para {args.out}")


if __name__ == "__main__":
    main()
