#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cria uma amostra estratificada de páginas a partir do CSV de métricas.

Estratos usados por padrão:
- metodo: ocr, ocr_tabela, (vazio)
- score_bin: low(<=0.33), mid(0.33-0.66], high(>0.66)
- chars_bin: short(<200), mid(200-800], long(>800)

Uso:
  python scripts/ocr_sample.py --metrics ocr_metrics.csv --out ocr_sample.csv --n 100
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_metrics(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labeled_ids(path: str) -> set:
    if not path or not os.path.exists(path):
        return set()
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row.get("id", ""))
    return ids


def assign_bins(row: Dict[str, str]) -> Tuple[str, str, str]:
    metodo = (row.get("metodo") or "").strip() or "(vazio)"
    score = parse_float(row.get("score", "0"))
    chars = float(row.get("chars", "0") or 0)

    if score <= 0.33:
        score_bin = "low"
    elif score <= 0.66:
        score_bin = "mid"
    else:
        score_bin = "high"

    if chars < 200:
        chars_bin = "short"
    elif chars <= 800:
        chars_bin = "mid"
    else:
        chars_bin = "long"

    return metodo, score_bin, chars_bin


def stratified_sample(rows: List[Dict[str, str]], n: int, seed: int = 42) -> List[Dict[str, str]]:
    random.seed(seed)
    strata = defaultdict(list)
    for r in rows:
        key = assign_bins(r)
        strata[key].append(r)

    keys = list(strata.keys())
    if not keys:
        return []

    base = max(1, n // len(keys))
    selected: List[Dict[str, str]] = []

    # 1) coleta base por estrato
    for k in keys:
        pool = strata[k]
        take = min(base, len(pool))
        selected.extend(random.sample(pool, take))

    # 2) completa até n com o restante disponível (priorizando estratos com sobra)
    remaining_needed = max(0, n - len(selected))
    if remaining_needed > 0:
        leftover = []
        for k in keys:
            pool = [r for r in strata[k] if r not in selected]
            leftover.extend(pool)
        if leftover:
            take = min(remaining_needed, len(leftover))
            selected.extend(random.sample(leftover, take))

    # 3) corta se passou
    if len(selected) > n:
        selected = selected[:n]

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera amostra estratificada de métricas de OCR")
    parser.add_argument("--metrics", default="ocr_metrics.csv", help="CSV de métricas exportadas")
    parser.add_argument("--out", default="ocr_sample.csv", help="CSV de saída da amostra")
    parser.add_argument("--n", type=int, default=100, help="Tamanho da amostra")
    parser.add_argument("--seed", type=int, default=42, help="Seed para amostragem")
    parser.add_argument("--exclude-labeled", help="CSV de rótulos para excluir ids já rotulados")
    args = parser.parse_args()

    rows = load_metrics(args.metrics)
    if not rows:
        print("CSV de métricas vazio ou inexistente.")
        return

    labeled_ids = load_labeled_ids(args.exclude_labeled) if args.exclude_labeled else set()
    if labeled_ids:
        rows = [r for r in rows if r.get("id", "") not in labeled_ids]

    # prioriza diversidade de fontes: mantém no máximo K páginas por source antes da estratificação
    # (opcional: comentar se quiser todas as páginas)
    cap_per_source = 10
    by_source = defaultdict(list)
    for r in rows:
        by_source[(r.get("source") or "")] .append(r)
    capped_rows: List[Dict[str, str]] = []
    for src, items in by_source.items():
        if len(items) <= cap_per_source:
            capped_rows.extend(items)
        else:
            capped_rows.extend(random.sample(items, cap_per_source))

    sample = stratified_sample(capped_rows, args.n, seed=args.seed)
    if not sample:
        print("Não foi possível gerar amostra (dados insuficientes).")
        return

    # preserva cabeçalho do metrics
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample:
            writer.writerow(r)

    print(f"Amostra salva em {args.out} com {len(sample)} linhas.")


if __name__ == "__main__":
    main()

