#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avalia a heurística de qualidade de OCR a partir de métricas exportadas e rótulos humanos.

Entrada:
  - CSV de métricas (ocr_metrics.csv)
  - CSV de rótulos (ocr_labels.csv) com colunas: id, label

Saída:
  - Métricas agregadas: AUC-ROC, melhor F1 e threshold correspondente, precisão/recall
  - Matriz de confusão no threshold escolhido
  - Quebra por metodo (ocr, ocr_tabela)
  - Opcional: JSON com relatório detalhado
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_metrics(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels(path: str) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id"]] = row["label"].strip().lower()
    return labels


def parse_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def to_xy(rows: List[Dict[str, str]], labels: Dict[str, str]) -> Tuple[List[float], List[int]]:
    xs: List[float] = []
    ys: List[int] = []
    for r in rows:
        _id = r.get("id", "")
        if _id not in labels:
            continue
        y = 1 if labels[_id] == "ruim" else 0
        x = parse_float(r.get("score", "0"))
        xs.append(x)
        ys.append(y)
    return xs, ys


def sweep_threshold(xs: List[float], ys: List[int], step: float = 0.01):
    """Varre thresholds e calcula métricas binárias para classe positiva = ruim (ys=1).

    Retorna lista de dicts com threshold, tp, fp, tn, fn, precision, recall, f1.
    """
    n = len(xs)
    if n == 0:
        return []
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])  # score crescente (ruim primeiro)
    # Calcula prefixos eficientes
    total_pos = sum(ys)
    total_neg = n - total_pos
    results = []
    t = 0.0
    while t <= 1.000001:
        # Classe positiva: score <= t (pior ou igual ao threshold)
        tp = sum(1 for x, y in pairs if x <= t and y == 1)
        fp = sum(1 for x, y in pairs if x <= t and y == 0)
        fn = total_pos - tp
        tn = total_neg - fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        results.append({
            "threshold": round(t, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1,
        })
        t += step
    return results


def roc_auc(xs: List[float], ys: List[int]) -> float:
    """AUC-ROC via integração trapezoidal sobre TPR x FPR.
    Classe positiva: ys=1 corresponde a scores menores (pior). Inverte sinal do score.
    """
    if not xs:
        return 0.0
    # Usa thresholds únicos a partir dos scores
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])  # score crescente (ruim primeiro)
    total_pos = sum(ys)
    total_neg = len(ys) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.0

    # Pontos ROC: para cada threshold, calcula TPR (=recall) e FPR
    # Classe positiva: x <= t
    prev_t = None
    points = []  # (FPR, TPR)
    tp = fp = 0
    # Varre thresholds nos scores (crescentes). A cada novo score, atualiza contagens
    for s, y in pairs:
        if prev_t is None or s != prev_t:
            # registra ponto anterior
            tpr = tp / total_pos
            fpr = fp / total_neg
            points.append((fpr, tpr))
            prev_t = s
        # inclui este item na classe positiva (<= threshold corrente)
        if y == 1:
            tp += 1
        else:
            fp += 1
    # último ponto (threshold acima de todos)
    tpr = tp / total_pos
    fpr = fp / total_neg
    points.append((fpr, tpr))

    # integra
    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2.0
    return max(0.0, min(1.0, auc))


def evaluate(rows: List[Dict[str, str]], labels: Dict[str, str], step: float = 0.01, fixed_threshold: float = None):
    xs, ys = to_xy(rows, labels)
    if not xs:
        return {"error": "Sem interseção entre métricas e rótulos."}

    auc = roc_auc(xs, ys)
    sweep = sweep_threshold(xs, ys, step=step)
    best = max(sweep, key=lambda r: r["f1"]) if sweep else None
    chosen = None
    if fixed_threshold is not None:
        # escolhe o ponto cujo threshold está mais próximo do informado
        chosen = min(sweep, key=lambda r: abs(r["threshold"] - fixed_threshold)) if sweep else None
    else:
        chosen = best

    # quebras por metodo
    per_metodo = defaultdict(list)
    for r in rows:
        _id = r.get("id", "")
        if _id not in labels:
            continue
        per_metodo[r.get("metodo", "")] .append((parse_float(r.get("score", "0")), 1 if labels[_id] == "ruim" else 0))

    per_metodo_auc = {}
    for metodo, pairs in per_metodo.items():
        sx = [s for s, _ in pairs]
        sy = [y for _, y in pairs]
        per_metodo_auc[metodo] = roc_auc(sx, sy)

    return {
        "n_samples": len(xs),
        "pos": sum(ys),
        "neg": len(xs) - sum(ys),
        "auc_roc": auc,
        "best_f1": best,
        "chosen": chosen,
        "per_metodo_auc": per_metodo_auc,
        "sweep_step": step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliador de heurística de OCR")
    parser.add_argument("--metrics", default="ocr_metrics.csv", help="CSV de métricas")
    parser.add_argument("--labels", default="ocr_labels.csv", help="CSV de rótulos humanos")
    parser.add_argument("--step", type=float, default=0.01, help="Passo do threshold sweep")
    parser.add_argument("--threshold", type=float, help="Threshold fixo a avaliar (opcional)")
    parser.add_argument("--report-json", help="Salvar relatório JSON neste caminho")
    args = parser.parse_args()

    rows = load_metrics(args.metrics)
    labels = load_labels(args.labels)
    report = evaluate(rows, labels, step=args.step, fixed_threshold=args.threshold)

    if "error" in report:
        print(report["error"])
        return

    print("Resumo:")
    print(f" amostras rotuladas: {report['n_samples']} (pos={report['pos']}, neg={report['neg']})")
    print(f" AUC-ROC: {report['auc_roc']:.3f}")
    if report.get("best_f1"):
        b = report["best_f1"]
        print(
            " Melhor F1={f1:.3f} em threshold={threshold} | P={precision:.3f} R={recall:.3f} (tp={tp} fp={fp} tn={tn} fn={fn})".format(
                **b
            )
        )
    if report.get("chosen") and (args.threshold is not None):
        c = report["chosen"]
        print(
            " Threshold escolhido={threshold} | F1={f1:.3f} P={precision:.3f} R={recall:.3f} (tp={tp} fp={fp} tn={tn} fn={fn})".format(
                **c
            )
        )

    if report.get("per_metodo_auc"):
        print(" AUC por metodo:")
        for metodo, auc in report["per_metodo_auc"].items():
            print(f"  - {metodo or '(vazio)'}: {auc:.3f}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Relatório salvo em {args.report_json}")


if __name__ == "__main__":
    main()

