# Final report

Place the editable report source, selected figures, and final PDF here.

## Required format

- Maximum 3 pages.
- Minimum 10 pt Arial.
- 1-inch margins on every side.
- Final format: PDF.

## Suggested outline

1. Executive summary.
2. Problem formulation and methodology.
3. Generative-AI disclosure, if applicable.
4. Modeling approach and outcomes.
5. Geometry predictions, ground-truth comparison, and uncertainty.
6. Limitations and uncertainties.
7. Conclusion: how thermal-image behavior relates to final track variation.

Use visualizations only when they directly support a claim: explain what each figure shows and why it is included.

## Audited final

`FMRG_Final_Report_Audited.pdf` is the generated, three-page final report.
It uses 10 pt Arial or larger, one-inch margins, and only metrics from
`results/improved_submission/metrics.json`.

Rebuild it with:

```bash
python scripts/build_final_report.py
```
