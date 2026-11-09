
"""Stage 05 - Train candidate models, select best on validation, register version."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
try:  # MLflow is optional: the pipeline falls back to the local registry without it
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

from sklearn.model_selection import train_test_split
from sklearn.base import clone

from src.common import (
    load_config,
    p,
    write_json,
    get_logger,
    append_run_log,
    now_iso,
)
from src.train import get_models, train_all, save_pickle
from src.evaluate import compute_metrics
from src.registry import register_version, mlflow_available

log = get_logger("05_train_registry")


def best_threshold(y, proba):
    """Find the decision threshold with the best validation F1 score."""
    thresholds = np.linspace(0.3, 0.7, 41)

    f1_scores = [
        compute_metrics(
            y,
            (proba >= threshold).astype(int),
            proba
        )["f1"]
        for threshold in thresholds
    ]

    return float(thresholds[int(np.argmax(f1_scores))])


def main():
    cfg = load_config()

    d = p(cfg["paths"]["processed_dir"])
    models_dir = p(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load processed training data
    X = np.load(d / "X_train_final.npy")
    y = np.load(d / "y_train.npy")

    # Split training and validation data
    X_sub, X_val, y_sub, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=cfg["data"]["random_state"],
        stratify=y
    )

    # Initialize MLflow
    use_mlflow = mlflow_available()

    if use_mlflow:
        mlflow.set_tracking_uri(
            f"sqlite:///{p('mlflow.db').as_posix()}"
        )

        mlflow.set_experiment(
            cfg["mlflow"]["experiment_name"]
        )
    else:
        log.warning(
            "MLflow not installed - using local registry only"
        )

    # Train candidate models
    results = {}

    models = get_models(cfg["models"])

    trained_models = train_all(
        models,
        X_sub,
        y_sub
    )

    for name, model in trained_models.items():

        proba = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(
            y_val,
            (proba >= 0.5).astype(int),
            proba
        )

        results[name] = {
            "metrics": metrics,
            "run_id": None
        }

        log.info(
            "%-24s val F1=%.4f AUC=%.4f",
            name,
            metrics["f1"],
            metrics["roc_auc"]
        )

    # Select the best model based on validation F1
    best_name = max(
        results,
        key=lambda name: results[name]["metrics"]["f1"]
    )

    log.info("Selected best model: %s", best_name)

    # Refit selected model on the full training data
    best_model = clone(models[best_name]).fit(X, y)

    # Determine decision threshold
    proba_val = best_model.predict_proba(X_val)[:, 1]

    threshold = best_threshold(
        y_val,
        proba_val
    )

    # Specific types required by the trusted local model formats.
    # Trust these only for models trained by your own project.
    trusted_types = [
        "sklearn.tree._tree.Tree",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier"
    ]

    # Save all candidate models and log MLflow runs
    for name in results:

        model = clone(models[name]).fit(X, y)

        # Save model locally
        save_pickle(
            model,
            models_dir / f"{name}.pkl"
        )

        if use_mlflow:

            with mlflow.start_run(run_name=name) as run:

                mlflow.log_params({
                    key: str(value)
                    for key, value in cfg["models"].get(name, {}).items()
                })

                mlflow.log_metrics({
                    key: value
                    for key, value in results[name]["metrics"].items()
                    if key != "confusion_matrix"
                })

                # Log model with explicitly reviewed trusted types
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    skops_trusted_types=trusted_types
                )

                results[name]["run_id"] = run.info.run_id

                log.info(
                    "MLflow logging successful for %s",
                    name
                )

    # Save selected production model
    save_pickle(
        best_model,
        models_dir / "airline_model.pkl"
    )

    save_pickle(
        threshold,
        models_dir / "decision_threshold.pkl"
    )

    # Register selected model locally
    version = register_version(
        "models/airline_model.pkl",
        best_name,
        results[best_name]["metrics"],
        threshold,
        results[best_name]["run_id"]
    )

    # Register selected model in MLflow
    if use_mlflow and results[best_name]["run_id"]:

        try:
            model_uri = (
                f"runs:/{results[best_name]['run_id']}/model"
            )

            registered_model = mlflow.register_model(
                model_uri,
                cfg["mlflow"]["registered_model_name"]
            )

            log.info(
                "Registered in MLflow as version %s",
                registered_model.version
            )

        except Exception as exc:
            log.warning(
                "MLflow registration skipped: %s",
                exc
            )

    # Save training summary
    write_json(
        models_dir / "training_summary.json",
        {
            "timestamp": now_iso(),
            "best_model": best_name,
            "registry_version": version,
            "decision_threshold": threshold,
            "validation_metrics": {
                name: result["metrics"]
                for name, result in results.items()
            }
        }
    )

    # Record successful pipeline stage
    append_run_log(
        "05_train_registry",
        "SUCCESS",
        {
            "best": best_name,
            "version": version
        }
    )

    log.info(
        "Best model: %s -> registry version %d (threshold %.3f)",
        best_name,
        version,
        threshold
    )


if __name__ == "__main__":
    main()
