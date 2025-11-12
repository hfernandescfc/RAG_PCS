#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pré-rotula páginas usando o score heurístico atual.

Pode operar sobre o CSV completo de métricas ou sobre uma amostra (ocr_sample.csv).
Gera um CSV de rótulos (id,label) e/ou acrescenta a coluna 'prelabel' no CSV de entrada.

Uso:
  python scripts/ocr_prelabel.py --metrics ocr_sample.csv --threshold 0.35 --labels-out ocr_labels_prelabeled.csv --augment

Alternativa balanceada:
  python scripts/ocr_prelabel.py --metrics ocr_sample.csv --target-n 50 --labels-out ocr_labels_prelabeled.csv --augment
"""

import argparse
import csv
from typing import Dict, List, Tuple


def load_metrics(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Pré-rotulador baseado em score heurístico")
    parser.add_argument("--metrics", default="ocr_sample.csv", help="CSV de métricas (amostra ou completo)")
    parser.add_argument("--labels-out", default="ocr_labels_prelabeled.csv", help="CSV de rótulos de saída")
    parser.add_argument("--augment", action="store_true", help="Escrever coluna 'prelabel' no CSV de entrada")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--threshold", type=float, help="Score <= threshold => ruim, senão ok")
    group.add_argument("--target-n", type=int, help="Marcar N piores (menor score) como ruim e o restante como ok")
    args = parser.parse_args()

    rows = load_metrics(args.metrics)
    if not rows:
        print("CSV de métricas vazio ou inexistente.")
        return

    # define rótulos
    if args.target_n is not None:
        # escolhe N piores por score
        ordered = sorted(rows, key=lambda r: parse_float(r.get("score", "0")))
        cutoff_ids = set([r.get("id", "") for r in ordered[: max(0, args.target_n)]])
        labels = [(r.get("id", ""), "ruim" if r.get("id", "") in cutoff_ids else "ok") for r in rows]
    else:
        thr = args.threshold if args.threshold is not None else 0.35
        labels = []
        for r in rows:
            score = parse_float(r.get("score", "0"))
            label = "ruim" if score <= thr else "ok"
            labels.append((r.get("id", ""), label))

    # grava labels
    with open(args.labels_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"]) 
        writer.writeheader()
        for _id, lab in labels:
            writer.writerow({"id": _id, "label": lab})

    print(f"Pré-rotulagem salva em {args.labels_out} ({len(labels)} itens)")

    # opcional: acrescenta coluna prelabel no CSV de entrada
    if args.augment:
        by_id = {i: lab for i, lab in labels}
        fieldnames = list(rows[0].keys())
        if "prelabel" not in fieldnames:
            fieldnames.append("prelabel")
        with open(args.metrics, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                rid = r.get("id", "")
                r["prelabel"] = by_id.get(rid, "")
                writer.writerow(r)
        print(f"Coluna 'prelabel' escrita em {args.metrics}")


if __name__ == "__main__":
    main()

