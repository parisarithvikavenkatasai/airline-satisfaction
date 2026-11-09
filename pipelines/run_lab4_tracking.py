"""Lab 4 - Experiment tracking with MLflow (sqlite backend: mlflow.db).

View runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    sys.exit("MLflow is not installed: pip install mlflow")

from sklearn.ensemble import RandomForestClassifier

from src.common import load_config, p
from src.evaluate import compute_metrics

cfg = load_config(); d = p(cfg["paths"]["processed_dir"])
Xtr, Xte, ytr, yte = (np.load(d / f) for f in ["X_train_final.npy", "X_test_final.npy", "y_train.npy", "y_test.npy"])
mlflow.set_tracking_uri(f"sqlite:///{p('mlflow.db').as_posix()}")
mlflow.set_experiment(cfg["mlflow"]["experiment_name"] + "-tracking")
for n_est, depth in [(50, 8), (100, 12), (200, 16)]:
    with mlflow.start_run(run_name=f"rf_{n_est}_{depth}"):
        m = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1).fit(Xtr, ytr)
        proba = m.predict_proba(Xte)[:, 1]
        met = compute_metrics(yte, (proba >= 0.5).astype(int), proba)
        mlflow.log_params({"n_estimators": n_est, "max_depth": depth})
        mlflow.log_metrics({k: v for k, v in met.items() if k != "confusion_matrix"})
        mlflow.sklearn.log_model(m, "model")
        print(f"rf n={n_est} depth={depth}: f1={met['f1']:.4f}")
