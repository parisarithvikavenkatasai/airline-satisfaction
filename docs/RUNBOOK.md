# Runbook

| Situation | What to do |
|---|---|
| Fresh setup | `python -m venv .venv` → activate → `pip install -r requirements.txt` |
| Run everything | `python src/mlops_pipeline/airline_prediction_pipeline.py` |
| Run + drift monitoring | add `--with-monitoring` |
| Quality gate fails (stage 07) | Read `reports/quality_gate_report.json`; the previous production model stays live. Fix data/model and rerun 05–08. |
| Validation fails (stage 02) | Open `artifacts/baseline_validation.csv` for the exact errors. |
| API returns 503 | Deployment files missing → `python src/mlops_pipeline/11_prepare_deployment.py`, then `POST /reload`. |
| Drift trigger says retrain | `12_retrain_model.py` → `12_model_comparison.py` → `12_update_mlflow_registry.py` |
| Roll back | In `models/registry.json` set `aliases.production` to the earlier version, then `python deployment/deploy_production.py` and `POST /reload`. |
| Inspect registry | `python src/mlops_pipeline/check_mlflow_registry.py` |
| MLflow UI | `mlflow ui --backend-store-uri sqlite:///mlflow.db` |

## Pickle compatibility
`.pkl` files are tied to the scikit-learn version that created them. After changing Python/scikit-learn versions, rerun the pipeline to regenerate `models/` and `deployment/`.
