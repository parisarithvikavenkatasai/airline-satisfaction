"""Lab 3 - Baseline models (no tracking). Requires stage 03 outputs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.common import load_config, p
from src.evaluate import compute_metrics
from src.train import save_pickle

cfg = load_config(); d = p(cfg["paths"]["processed_dir"])
if not (d / "X_train_final.npy").exists():
    sys.exit("Run pipelines/run_lab5_pipeline.py (stages 01-04) first.")
Xtr, Xte, ytr, yte = (np.load(d / f) for f in ["X_train_final.npy", "X_test_final.npy", "y_train.npy", "y_test.npy"])
for name, model in {"decision_tree_baseline": DecisionTreeClassifier(max_depth=8, random_state=42),
                    "random_forest_baseline": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)}.items():
    model.fit(Xtr, ytr); proba = model.predict_proba(Xte)[:, 1]
    m = compute_metrics(yte, (proba >= 0.5).astype(int), proba)
    save_pickle(model, p(cfg["paths"]["models_dir"], f"{name}.pkl"))
    print(f"{name:26s} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
