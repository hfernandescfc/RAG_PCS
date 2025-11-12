#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI simples para rotulagem humana de páginas exportadas em ocr_metrics.csv.

Lê um CSV de métricas (gerado por scripts/ocr_export_metrics.py), exibe um
preview com metadados e métricas principais e grava rótulos incrementais em
um arquivo CSV (por padrão, ocr_labels.csv) com colunas: id, label.

Comandos na interação:
  y -> marcar como ruim
  n -> marcar como ok
  s -> pular (skip)
  q -> sair
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional


def load_metrics(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id"]] = row["label"]
    return labels


def append_label(path: str, _id: str, label: str, extra: Dict[str, str]) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "label", "ts", "source", "metodo", "page", "score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "id": _id,
                "label": label,
                "ts": str(int(time.time())),
                "source": extra.get("source", ""),
                "metodo": extra.get("metodo", ""),
                "page": extra.get("page", ""),
                "score": extra.get("score", ""),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rotulador CLI para métricas de OCR")
    parser.add_argument("--metrics", default="ocr_metrics.csv", help="CSV de métricas exportadas")
    parser.add_argument("--out", default="ocr_labels.csv", help="CSV de saída de rótulos")
    parser.add_argument("--only-unlabeled", action="store_true", help="Mostrar apenas itens ainda não rotulados")
    parser.add_argument(
        "--uncertainty-first",
        action="store_true",
        help="Ordena por proximidade do score a 0.5 (incertos primeiro)",
    )
    parser.add_argument("--min-score", type=float, help="Filtra itens com score >= min-score")
    parser.add_argument("--max-score", type=float, help="Filtra itens com score <= max-score")
    parser.add_argument("--suggest-column", default="prelabel", help="Coluna de sugestão automática (se existir)")
    parser.add_argument("--suggest-threshold", type=float, help="Se não houver coluna, sugerir via score <= threshold => ruim")
    args = parser.parse_args()

    rows = load_metrics(args.metrics)
    if not rows:
        print("CSV de métricas vazio ou inexistente.")
        sys.exit(1)

    labels = load_labels(args.out)

    # Conversões e filtros leves
    def parse_float(x: Optional[str]) -> float:
        try:
            return float(x) if x is not None else 0.0
        except Exception:
            return 0.0

    items = []
    for r in rows:
        score = parse_float(r.get("score"))
        if args.min_score is not None and score < args.min_score:
            continue
        if args.max_score is not None and score > args.max_score:
            continue
        items.append((r, score))

    if args.uncertainty_first:
        items.sort(key=lambda t: abs(t[1] - 0.5))
    else:
        # mais problemáticos primeiro por padrão
        items.sort(key=lambda t: t[1])

    print("Instruções: y=ruim, n=ok, Enter=aceitar sugestão, s=pular, q=sair\n")
    total = len(items)
    processed = 0
    for r, score in items:
        processed += 1
        _id = r.get("id", "")
        already = labels.get(_id)
        if args.only_unlabeled and already:
            continue

        src = r.get("source", "")
        metodo = r.get("metodo", "")
        page = r.get("page", "")
        preview = r.get("text_preview", "")

        print("-" * 80)
        print(f"[{processed}/{total}] id={_id} | score={score:.3f} | metodo={metodo} | page={page}")
        print(f"source: {src}")
        if already:
            print(f"label existente: {already}")
        print("preview:")
        print(preview)
        print("métricas principais:")
        print(
            "chars={chars} words={words} alpha={alpha_ratio} digit={digit_ratio} stop={stop_ratio} vowels={vowel_ratio} suspicious={suspicious_count}".format(
                **{k: r.get(k, "") for k in (
                    "chars",
                    "words",
                    "alpha_ratio",
                    "digit_ratio",
                    "stop_ratio",
                    "vowel_ratio",
                    "suspicious_count",
                )}
            )
        )

        # sugestão automática
        suggest = None
        col = args.suggest_column
        if col and r.get(col):
            val = str(r.get(col)).strip().lower()
            if val in {"ruim", "ok"}:
                suggest = val
        if suggest is None and args.suggest_threshold is not None:
            suggest = "ruim" if score <= float(args.suggest_threshold) else "ok"
        if suggest:
            print(f"sugestão automática: {suggest}")

        while True:
            ans = input("rótulo? [y=ruim / n=ok / Enter=aceitar / s=skip / q=quit] ").strip().lower()
            if ans == "q":
                print("Saindo…")
                return
            if ans in {"y", "n", "s"}:
                break
            if ans == "":
                if suggest in {"ruim", "ok"}:
                    ans = "y" if suggest == "ruim" else "n"
                    break
                else:
                    print("Sem sugestão disponível; escolha y/n/s/q.")
            else:
                print("Entrada inválida. Use y, n, Enter, s ou q.")

        if ans == "s":
            continue

        label = "ruim" if ans == "y" else "ok"
        append_label(
            args.out,
            _id,
            label,
            {
                "source": src,
                "metodo": metodo,
                "page": page,
                "score": str(score),
            },
        )
        labels[_id] = label

    print("Concluído.")


if __name__ == "__main__":
    main()
