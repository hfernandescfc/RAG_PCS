#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnóstico de qualidade de OCR no banco vetorial.

Analisa documentos indexados via OCR (metodo='ocr' ou 'ocr_tabela'),
calcula métricas heurísticas por página e agrega por arquivo (source).

Saídas:
- Relatório no console com ranking de arquivos mais problemáticos
- Opcional: grava JSON com resultados detalhados (ocr_report.json)
"""

import os
import json
import argparse
from collections import defaultdict, Counter
from statistics import mean

from rag_pipeline.config import DB_PATH, get_embeddings
from langchain_community.vectorstores import Chroma


PT_STOPWORDS = {
    # subconjunto suficiente para heuristicamente detectar português
    "de","a","o","que","e","do","da","em","um","para","é","com","não","uma",
    "os","no","se","na","por","mais","as","dos","como","mas","foi","ao","ele",
    "das","tem","à","seu","sua","ou","ser","quando","muito","há","nos","já","está",
    "eu","também","só","pelo","pela","até","isso","ela","entre","depois","sem","mesmo",
    "aos","seus","quem","nas","me","esse","eles","estão","você","tinha","foram","essa",
}


def _metrics(text: str, meta: dict | None = None) -> dict:
    if not text:
        return {
            "chars": 0,
            "words": 0,
            "alpha_ratio": 0.0,
            "digit_ratio": 0.0,
            "space_ratio": 0.0,
            "punct_ratio": 0.0,
            "avg_word_len": 0.0,
            "suspicious_count": 0,
        }

    chars = len(text)
    words_list = [w for w in text.split() if w]
    words = len(words_list)
    alpha = sum(1 for c in text if c.isalpha())
    digit = sum(1 for c in text if c.isdigit())
    space = text.count(" ")
    punct = sum(1 for c in text if c in ",.;:!?()[]{}-_/\\\'\"@#$%&*+=<>")
    suspicious = text.count("�") + text.count("□") + text.count("¤")
    avg_word_len = mean([len(w) for w in words_list]) if words_list else 0.0
    long_words = sum(1 for w in words_list if len(w) >= 20)
    vowels = sum(1 for c in text if c.lower() in "aeiouáàãâéêíóôõúü")
    vowel_ratio = vowels / alpha if alpha else 0.0
    stop_hits = sum(1 for w in words_list if w.lower().strip(".,;:!?()[]{}\"'") in PT_STOPWORDS)
    stop_ratio = stop_hits / words if words else 0.0
    grid_score = float((meta or {}).get("grid_score", 0.0))
    metodo = (meta or {}).get("metodo")

    return {
        "chars": chars,
        "words": words,
        "alpha_ratio": alpha / chars if chars else 0.0,
        "digit_ratio": digit / chars if chars else 0.0,
        "space_ratio": space / chars if chars else 0.0,
        "punct_ratio": punct / chars if chars else 0.0,
        "avg_word_len": avg_word_len,
        "suspicious_count": suspicious,
        "long_word_ratio": (long_words / words) if words else 0.0,
        "vowel_ratio": vowel_ratio,
        "stop_ratio": stop_ratio,
        "grid_score": grid_score,
        "metodo": metodo,
    }


def _score_quality(m: dict) -> float:
    """Heurística de 0 (ruim) a 1 (bom), mais sensível a ruído de OCR/tabular."""
    if m["chars"] < 80:
        return 0.15

    score = 1.0

    # Penalizações por distribuição de caracteres
    if m["alpha_ratio"] < 0.5:
        score -= 0.35
    if m["digit_ratio"] > 0.5:
        score -= 0.25
    if m["avg_word_len"] > 12:
        score -= 0.2
    if m["long_word_ratio"] > 0.1:
        score -= 0.2
    if m["suspicious_count"] > 2:
        score -= 0.15

    # Português “parece português”? (stopwords/vogais)
    if m["stop_ratio"] < 0.02 and m["words"] > 40:
        score -= 0.25
    if m["vowel_ratio"] < 0.25 and m["alpha_ratio"] > 0.2:
        score -= 0.2

    # Páginas com grade detectada, mas sem modo tabular estruturado
    if m.get("grid_score", 0.0) > 0.004 and m.get("metodo") in ("ocr", None):
        score -= 0.25

    return max(0.0, min(1.0, score))


def main():
    parser = argparse.ArgumentParser(description="Diagnóstico de qualidade de OCR no banco vetorial")
    parser.add_argument("--source", help="Nome exato do arquivo (campo 'source') para filtrar")
    parser.add_argument("--contains", help="Filtro por substring do 'source' (case-insensitive)")
    parser.add_argument("--top", type=int, default=15, help="Qtde de arquivos a exibir no ranking (padrão: 15)")
    parser.add_argument("--export", action="store_true", help="Exportar relatório JSON detalhado (ocr_report.json)")
    args = parser.parse_args()

    print("\n=== Diagnóstico de OCR no Banco Vetorial ===\n")

    if not os.path.exists(DB_PATH):
        print(f"[ERRO] Banco não encontrado em: {DB_PATH}")
        return

    embeddings = get_embeddings()
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    data = db.get()

    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    ids = data.get("ids", [])

    ocr_indices = [i for i, m in enumerate(metas) if m.get("metodo") in ("ocr", "ocr_tabela")]

    if not ocr_indices:
        print("Nenhum chunk com metodo 'ocr' ou 'ocr_tabela' foi encontrado.")
        return

    print(f"Chunks analisados (OCR): {len(ocr_indices)}\n")

    per_source = defaultdict(list)
    per_source_flags = defaultdict(int)
    per_source_counts = Counter()

    detailed = []

    for i in ocr_indices:
        text = docs[i] or ""
        meta = metas[i] or {}
        m = _metrics(text, meta)
        q = _score_quality(m)

        source = meta.get("source", "?")
        page = meta.get("page", "?")
        metodo = meta.get("metodo", "?")

        item = {
            "id": ids[i],
            "source": source,
            "page": page,
            "metodo": metodo,
            "metrics": m,
            "quality": q,
            "preview": text[:200].replace("\n", " ")
        }
        detailed.append(item)

        per_source[source].append(q)
        per_source_counts[source] += 1
        if q < 0.5:
            per_source_flags[source] += 1

    # Se solicitado, filtrar por um arquivo específico
    if args.source or args.contains:
        filtro = (args.source or args.contains).lower()
        def ok(name: str) -> bool:
            if args.source:
                return name == args.source
            return filtro in (name or "").lower()

        filtrados = [d for d in detailed if ok(d.get("source", ""))]
        if not filtrados:
            print("Nenhum chunk encontrado para o filtro fornecido.")
            return

        # Agrupar por source e relatar
        agrup = defaultdict(list)
        for d in filtrados:
            agrup[d["source"]].append(d)

        for src, itens in agrup.items():
            qualities = [it["quality"] for it in itens]
            media = mean(qualities) if qualities else 0.0
            ruins = sum(1 for it in itens if it["quality"] < 0.5)
            print(f"\nArquivo: {src}")
            print(f"Páginas OCR: {len(itens)} | média={media:.2f} | páginas <0.5: {ruins}")
            print("Piores páginas (top 10):")
            for it in sorted(itens, key=lambda x: x["quality"])[:10]:
                q = it["quality"]
                page = it.get("page", "?")
                metodo = it.get("metodo", "?")
                prev = it.get("preview", "")
            print(f"  - pág {page:>4} | q={q:.2f} | metodo={metodo} | stop={m['stop_ratio']:.3f} | grid={m['grid_score']:.4f} | {prev[:120]}...")
        # Exportar somente o subset filtrado, se pedido
        if args.export:
            out = {
                "db_path": DB_PATH,
                "filtered": True,
                "filters": {"source": args.source, "contains": args.contains},
                "details": filtrados,
            }
            with open("ocr_report_filtered.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print("\nRelatório filtrado salvo em: ocr_report_filtered.json")
        return

    # Ranking por pior média de qualidade (sem filtro)
    ranking = sorted(((src, mean(scores)) for src, scores in per_source.items()), key=lambda x: x[1])

    print(f"Top {args.top} arquivos com pior média de qualidade (0=ruim,1=bom):\n")
    for src, avgq in ranking[: args.top]:
        total = per_source_counts[src]
        ruins = per_source_flags[src]
        perc_ruins = 100.0 * ruins / total if total else 0.0
        print(f"- {src} | média={avgq:.2f} | paginas_ocr={total} | suspeitas<{0.5}={ruins} ({perc_ruins:.1f}%)")

    # Salvar relatório detalhado (opcional)
    if args.export:
        try:
            with open("ocr_report.json", "w", encoding="utf-8") as f:
                json.dump({
                    "db_path": DB_PATH,
                    "summary": [
                        {
                            "source": src,
                            "avg_quality": mean(per_source[src]),
                            "pages_ocr": per_source_counts[src],
                            "flagged_lt_0_5": per_source_flags[src],
                        } for src, _ in ranking
                    ],
                    "details": detailed,
                }, f, ensure_ascii=False, indent=2)
            print("\nRelatório detalhado salvo em: ocr_report.json")
        except Exception as e:
            print(f"\n[AVISO] Falha ao salvar ocr_report.json: {e}")

    # Dicas de correção
    print("\nSugestões:")
    print("- Reprocessar páginas com qualidade < 0.5 com DPI maior (ex.: 300→400).")
    print("- Testar config OCR --psm 6 quando houver muitas tabelas/colunas.")
    print("- Quando possível, preferir PDFs com texto (não escaneados) ou usar pdfplumber.")


if __name__ == "__main__":
    main()
