# Monitoring (stage 12)

| Script | Purpose | Output |
|---|---|---|
| `12_create_evolved_dataset.py` | Simulated covariate drift (older passengers, longer delays, lower wifi/food/seat scores, more disloyal, more Eco) | `data/monitoring/drift/evolved_*.csv` |
| `12_create_concept_drift_dataset.py` | Simulated label drift (business travellers harder to please) | `concept_drift_data.csv` |
| `12_create_updated_dataset.py` | Original train + drifted data | `post_drift_training_data.csv`, `data/processed/updated_training_data.csv` |
| `12_data_drift.py` | PSI + KS test per feature | `reports/data_drift_report.json`, `artifacts/data_drift_table.csv` |
| `12_prediction_drift.py` | Shift in predicted positive rate / probability PSI | `reports/prediction_drift_report.json` |
| `12_concept_drift.py` | F1 drop on labeled drifted data | `reports/concept_drift_report.json` |
| `12_performance_monitoring.py` / `12_performance_trend.py` | F1 on reference / evolved / concept data, history + plot | `reports/performance_report.json`, `artifacts/performance_trend.png` |
| `12_request_history.py` / `12_prediction_monitoring.py` | Summaries of real logged requests | `reports/request_history_report.json`, `reports/prediction_monitoring_report.json` |
| `12_drift_analysis.py` | Combined summary | `reports/drift_analysis_report.json` |
| `12_monitor_trigger.py` | Retrain decision | `reports/monitor_trigger_report.json` |
| `12_retrain_model.py` | New preprocessor + model (v2 artifacts), registered as `candidate` | `models/retrained_*_v2.pkl` |
| `12_model_comparison.py` | Production vs candidate on the same post-drift holdout | `reports/model_comparison_report.json` |
| `12_update_mlflow_registry.py` | Promote + redeploy if better | `reports/lifecycle_report.json` |
| `12_api_performance_monitoring.py` | Latency/throughput of the live API | `artifacts/api_performance_report.json` |

Thresholds live in `config.yaml` → `monitoring`. Retrain triggers: concept drift, ≥25% of features drifted, or prediction drift.
The drifted datasets are **synthetic simulations** built from the test split; replace them with real production data when available.
