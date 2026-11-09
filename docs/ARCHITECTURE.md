# Architecture

```
data/raw/airline.csv
   │  01 load → 02 validate → 03 preprocess (split, fit ColumnTransformer) → 04 validate outputs
   ▼
data/processed/*.npy + models/preprocessor.pkl
   │  05 train + register (candidate) → 06 evaluate → 07 quality gate → 08 lifecycle (promote → production)
   ▼
models/registry.json (+ mlflow.db when MLflow is installed)
   │  09 batch inference  ·  10 failure injection  ·  11 prepare deployment
   ▼
deployment/{model.pkl, preprocessor.pkl, model_metadata.json}  →  api/ (FastAPI)  →  Docker  →  k8s
   ▲                                                                   │ logs/inference/inference_log.jsonl
   └── 12_* monitoring: data / prediction / concept drift → trigger → retrain → compare → promote ◄─┘
```

## Key design decisions
- **Training/serving consistency**: `api/preprocessing.py` reuses `src/preprocess.py` (same cleaning, `TotalDelay`, `CompositeServiceScore`).
- **Registry**: `models/registry.json` is the source of truth (works offline); MLflow runs/aliases are mirrored when installed.
- **Aliases**: `candidate` (newest trained) and `production` (served). Stage 08 promotes only if the quality gate passes and F1 is not worse than the current production model.
- **Model selection** uses a validation split of the training data; the test set is only used in stage 06.
- **Missing data**: `Arrival Delay in Minutes` is median-imputed (median stored in metadata and reused at serving time).
- **Optional deps**: XGBoost is used when installed, otherwise `HistGradientBoostingClassifier` is the third candidate.

## Features
Raw: 22 columns (4 categorical, 18 numeric incl. 14 survey ratings). Engineered: `TotalDelay`, `CompositeServiceScore`.
Target: `satisfaction` → 1 = `satisfied`, 0 = `neutral or dissatisfied`.
