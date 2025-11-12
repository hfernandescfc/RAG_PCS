#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treina um modelo simples (Logistic Regression) para classificar páginas ok/ruim
com base nas métricas exportadas. Requer scikit-learn instalado.

Uso:
  python scripts/ocr_train_model.py --metrics ocr_metrics.csv --labels ocr_labels.csv --save models/ocr_lr.joblib

O script faz split por 'source' para evitar vazamento entre treino e teste.
"""

import argparse
import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


def load_metrics(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_labels(path: str) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lab = 1 if str(row.get("label", "")).strip().lower() == "ruim" else 0
            labels[row.get("id", "")] = lab
    return labels


def parse_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


FEATURES = [
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
    "grid_score",
]


def build_dataset(rows: List[Dict[str, str]], labels: Dict[str, int]):
    X = []
    y = []
    sources = []
    ids = []
    for r in rows:
        _id = r.get("id", "")
        if _id not in labels:
            continue
        xi = [parse_float(r.get(f, "0")) for f in FEATURES]
        X.append(xi)
        y.append(labels[_id])
        sources.append(r.get("source", ""))
        ids.append(_id)
    return X, y, sources, ids


def split_by_source(sources: List[str], test_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)
    unique = list({s for s in sources})
    random.shuffle(unique)
    n_test = max(1, int(len(unique) * test_ratio))
    test_src = set(unique[:n_test])
    train_idx, test_idx = [], []
    for i, s in enumerate(sources):
        (test_idx if s in test_src else train_idx).append(i)
    return train_idx, test_idx


def subset(lst, idx):
    return [lst[i] for i in idx]


def evaluate(y_true, y_prob, step: float = 0.01):
    # varre thresholds
    pairs = sorted(zip(y_prob, y_true), key=lambda t: t[0])  # prob positiva = ruim
    pos = sum(y_true)
    neg = len(y_true) - pos
    best = {"f1": 0.0}
    t = 0.0
    while t <= 1.000001:
        tp = sum(1 for p, y in pairs if p <= t and y == 1)
        fp = sum(1 for p, y in pairs if p <= t and y == 0)
        fn = pos - tp
        tn = neg - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        if f1 > best.get("f1", 0):
            best = {
                "threshold": round(t, 4),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": prec, "recall": rec, "f1": f1,
            }
        t += step
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina Logistic Regression para classificar páginas ok/ruim")
    parser.add_argument("--metrics", default="ocr_metrics.csv", help="CSV de métricas")
    parser.add_argument("--labels", default="ocr_labels.csv", help="CSV de rótulos humanos")
    parser.add_argument("--save", help="Caminho para salvar o modelo (joblib)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        import joblib
    except Exception as e:
        print("scikit-learn não encontrado. Instale com: pip install scikit-learn joblib")
        return

    rows = load_metrics(args.metrics)
    labels = load_labels(args.labels)
    X, y, sources, ids = build_dataset(rows, labels)
    if not X:
        print("Sem interseção entre métricas e rótulos.")
        return

    tr_idx, te_idx = split_by_source(sources, test_ratio=0.2, seed=args.seed)
    Xtr, ytr = subset(X, tr_idx), subset(y, tr_idx)
    Xte, yte = subset(X, te_idx), subset(y, te_idx)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=200, class_weight="balanced")),
    ])
    pipe.fit(Xtr, ytr)

    # prob positiva (ruim)
    yprob_tr = pipe.predict_proba(Xtr)[:, 0]  # classe 0 é 'ok' se classes_=[0,1]; precisamos da prob de 'ruim' => classe 1
    # Corrige: detecta índice da classe 1
    cls_idx = list(pipe.named_steps["lr"].classes_).index(1)
    yprob_tr = pipe.predict_proba(Xtr)[:, cls_idx]
    yprob_te = pipe.predict_proba(Xte)[:, cls_idx]

    best_tr = evaluate(ytr, yprob_tr)
    best_te = evaluate(yte, yprob_te)

    print("Treino:")
    print("  Melhor F1={f1:.3f} em threshold={threshold} | P={precision:.3f} R={recall:.3f} (tp={tp} fp={fp} tn={tn} fn={fn})".format(**best_tr))
    print("Teste:")
    print("  Melhor F1={f1:.3f} em threshold={threshold} | P={precision:.3f} R={recall:.3f} (tp={tp} fp={fp} tn={tn} fn={fn})".format(**best_te))

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        import joblib  # type: ignore
        joblib.dump({
            "model": pipe,
            "features": FEATURES,
        }, args.save)
        print(f"Modelo salvo em {args.save}")


if __name__ == "__main__":
    main()

