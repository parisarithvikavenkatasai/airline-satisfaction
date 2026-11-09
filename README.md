# Airline Passenger Satisfaction – MLOps Project

End-to-end MLOps pipeline (data validation → preprocessing → training/registry → quality gate → deployment → FastAPI → Docker/Kubernetes → drift monitoring & retraining), structured like the churn-prediction project.

Task: binary classification – will a passenger be **satisfied** (1) or **neutral/dissatisfied** (0)?
Data: `data/raw/airline.csv` (25,976 passengers, 22 features + target).

## Quick start (VS Code)
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python src/mlops_pipeline/airline_prediction_pipeline.py                    # stages 01-11
python src/mlops_pipeline/airline_prediction_pipeline.py --with-monitoring  # + drift stage 12
pytest -q

uvicorn api.main:app --reload --port 8000    # Swagger UI: http://localhost:8000/docs
```
Individual stages can be run on their own, e.g. `python src/mlops_pipeline/05_train_registry.py`.
Run `python src/mlops_pipeline/12_retrain_model.py`, `12_model_comparison.py`, `12_update_mlflow_registry.py` when the monitor trigger says retraining is required.

> The shipped `.pkl` files were created with scikit-learn 1.9 on Python 3.12. If loading fails or warns on your machine, rerun the pipeline once to regenerate them.

## Results (held-out 20% test split, from the shipped run)
| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic regression | 0.872 | 0.853 | 0.927 |
| Random forest | 0.955 | 0.949 | 0.992 |
| Hist. gradient boosting (production, v1) | 0.959 | 0.953 | 0.994 |

XGBoost is added automatically as a candidate when `xgboost` is installed (the sandbox this was built in did not have it, so those numbers are not included). Your own run will pick whichever candidate has the best validation F1.
In the shipped monitoring demo the drifted datasets are **simulated**; a retrained v2 model was promoted on a post-drift holdout with F1 0.909 vs 0.908 for v1, which is effectively a tie – tune `monitoring.min_improvement_to_promote` in `config.yaml` if you want a stricter rule.

## API
`POST /predict` (JSON keys = original column names, see `/docs` for an example), `POST /predict_batch`, `GET /health`, `GET /model-info`, `GET /metrics`, `POST /reload`.
Every prediction is appended to `logs/inference/inference_log.jsonl` for monitoring.

## Layout
```
.github/workflows/  CI (lab8), failure tests (lab9), scheduled monitoring (lab13)
api/                FastAPI app, schemas, model loader, request preprocessing, monitoring logger
artifacts/          validation/preprocessing/failure reports, confusion matrix, ROC, drift tables
data/raw|processed|monitoring/drift
deployment/         deploy_production.py + model.pkl, preprocessor.pkl, model_metadata.json
docs/               ARCHITECTURE, PIPELINE, MONITORING, RUNBOOK, DOCKER_K8S
k8s/                deployment.yaml, service.yaml
models/             all trained models, preprocessor, registry.json, retrained v2 artifacts
notebooks/          project_implementation.ipynb (your original EDA notebook)
pipelines/          lab 3-6 runnable wrappers
reports/            evaluation, quality gate, lifecycle, drift, performance reports
src/                common, preprocess, validate_data, train, evaluate, registry, monitoring_utils
src/mlops_pipeline/ numbered stages 01-12 + orchestrator
tests/              preprocessing, validation, artifacts, API, performance, container-vs-local
config.yaml  Dockerfile  requirements*.txt
```
Details: see `docs/`.
