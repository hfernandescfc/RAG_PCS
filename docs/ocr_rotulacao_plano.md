Plano de Rotulagem – 3 Semanas

Semana 1 – Setup (≈2h)
- Exportar métricas: `python scripts/ocr_export_metrics.py --out ocr_metrics.csv`
- Amostra estratificada (100): `python scripts/ocr_sample.py --metrics ocr_metrics.csv --out ocr_sample.csv --n 100`
- Guidelines: ver `docs/ocr_labeling_guidelines.md`

Semana 2 – Rotulação Híbrida (≈4h)
- Pré-rotular amostra: `python scripts/ocr_prelabel.py --metrics ocr_sample.csv --threshold 0.35 --labels-out ocr_labels_prelabeled.csv --augment`
- Rotular revisando sugestão (Enter aceita): `python scripts/ocr_label_cli.py --metrics ocr_sample.csv --out ocr_labels.csv --only-unlabeled`
- Validação independente (10 itens): `python scripts/ocr_sample.py --metrics ocr_metrics.csv --out ocr_valid.csv --n 10 --exclude-labeled ocr_labels.csv`
- Rotular validação: `python scripts/ocr_label_cli.py --metrics ocr_valid.csv --out ocr_labels.csv`

Semana 3 – Análise (≈1h)
- Avaliar heurística: `python scripts/ocr_eval.py --metrics ocr_metrics.csv --labels ocr_labels.csv`
- Ajustar threshold operacional conforme melhor F1 (ou precisão alvo).
- (Opcional) Treinar modelo simples e comparar.

Notas
- Sempre mantenha páginas do mesmo `source` no mesmo split ao treinar/validar modelos.
- Repita a amostragem periodicamente para capturar novas distribuições de documentos.

