"""Assemble the reviewed FMRG submission ZIP without raw data."""

from __future__ import annotations

import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deliverables/submission/FMRG_Final_Submission_Audited.zip"

FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "notebooks/03_final_submission_audited.ipynb",
    "deliverables/report/FMRG_Final_Report_Audited.pdf",
    "deliverables/presentation/FMRG_Final_Submission_Audited.pptx",
    "deliverables/presentation/FMRG_Final_Submission_Audited.pdf",
    "deliverables/presentation/FMRG_Final_Template_Starter.pptx",
    "deliverables/submission/REPOSITORY_URLS.txt",
    "deliverables/submission/SUBMISSION_MANIFEST.md",
    "results/final_submission/README.md",
    "results/final_submission/metrics.json",
    "results/final_submission/figures/track21_held_out_comparison.png",
    "results/final_submission/figures/track14_validation_comparison.png",
    "results/final_submission/figures/feature_importance.png",
    "results/final_submission/tables/metric_summary.csv",
    "results/final_submission/tables/track21_predictions.csv",
    "results/final_submission/tables/development_oof_predictions.csv",
    "results/final_submission/tables/feature_importance.csv",
    "scripts/run_final_analysis.py",
    "scripts/build_final_notebook.py",
    "scripts/build_final_report.py",
    "scripts/build_final_deck.mjs",
    "scripts/final_deck_template_inventory.ndjson",
    "src/nsf_fmrg_data.py",
    "src/fmrg_submission/__init__.py",
    "src/fmrg_submission/geometry.py",
    "src/fmrg_submission/modeling.py",
    "src/fmrg_submission/sem.py",
    "src/fmrg_submission/thermal.py",
    "tests/test_geometry.py",
    "tests/test_modeling.py",
    "tests/test_sem.py",
    "tests/test_thermal.py",
]


def main():
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing submission files: {missing}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in FILES:
            archive.write(ROOT / relative, arcname=f"FMRG_Final_Submission/{relative}")

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"{OUTPUT}\nsha256 {digest}")


if __name__ == "__main__":
    main()
