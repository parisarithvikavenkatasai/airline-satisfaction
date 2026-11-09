# Pipeline stages

| # | Script | Reads | Writes |
|---|---|---|---|
| 01 | `01_load_data.py` | raw csv | `artifacts/data_load_report.json` |
| 02 | `02_validate_data.py` | raw csv | `artifacts/baseline_validation.csv` (fails on errors) |
| 03 | `03_preprocessing.py` | raw csv | `data/processed/*`, `models/preprocessor.pkl` |
| 04 | `04_validate_outputs.py` | processed | `artifacts/preprocessing_summary_report.json` |
| 05 | `05_train_registry.py` | processed | `models/*.pkl`, `models/registry.json`, MLflow runs |
| 06 | `06_model_evaluation.py` | test arrays | `reports/evaluation_report.json`, `artifacts/confusion_matrix.png`, `roc_curve.png` |
| 07 | `07_quality_gate.py` | evaluation | `reports/quality_gate_report.json` (exit 1 on fail) |
| 08 | `08_automate_lifecycle.py` | gate + registry | `reports/lifecycle_report.json` |
| 09 | `09_inference.py` | production model | `logs/inference/inference_log.jsonl` |
| 10 | `10_failure_injection.py` | raw csv | `artifacts/failure_simulation_report.json`, `corrupted_simulation_errors.csv`, `reports/failure_report.json` |
| 11 | `11_prepare_deployment.py` | registry | `deployment/*`, `reports/deployment_automation_report.json` |
| 12 | `12_*.py` | see MONITORING.md | |

Lab wrappers: `pipelines/run_lab3_baseline.py`, `run_lab4_tracking.py` (MLflow), `run_lab5_pipeline.py` (01–04), `run_lab6_registry.py` (05–08).
