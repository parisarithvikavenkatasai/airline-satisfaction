"""Stage 06 - Evaluate the candidate model on the held-out test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.common import load_config, p, write_json, get_logger, append_run_log, now_iso
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_roc
from src.registry import get_alias, update_version
from src.train import load_pickle

log = get_logger("06_model_evaluation")


def main():
    cfg = load_config()
    d = p(cfg["paths"]["processed_dir"]); art = p(cfg["paths"]["artifacts_dir"])
    X, y = np.load(d / "X_test_final.npy"), np.load(d / "y_test.npy")
    cand = get_alias("candidate")
    model = load_pickle(p(cand["model_file"]))
    proba = model.predict_proba(X)[:, 1]
    m = compute_metrics(y, (proba >= cand["threshold"]).astype(int), proba)

    plot_confusion_matrix(m["confusion_matrix"], art / "confusion_matrix.png", f"Confusion Matrix - {cand['model_type']}")
    plot_roc(y, proba, art / "roc_curve.png", f"ROC - {cand['model_type']}")

    others = {}
    for name in ["logistic_regression", "random_forest", "xgboost", "hist_gradient_boosting"]:
        f = p(cfg["paths"]["models_dir"], f"{name}.pkl")
        if f.exists():
            mm = load_pickle(f); pr = mm.predict_proba(X)[:, 1]
            others[name] = compute_metrics(y, (pr >= 0.5).astype(int), pr)

    report = {"timestamp": now_iso(), "registry_version": cand["version"], "model_type": cand["model_type"],
              "threshold": cand["threshold"], "test_metrics": m, "all_models_test_metrics": others}
    update_version(cand["version"], test_metrics=m)
    write_json(p(cfg["paths"]["reports_dir"], "evaluation_report.json"), report)
    append_run_log("06_model_evaluation", "SUCCESS", {k: round(v, 4) for k, v in m.items() if k != "confusion_matrix"})
    log.info("Test: acc=%.4f f1=%.4f auc=%.4f recall=%.4f", m["accuracy"], m["f1"], m["roc_auc"], m["recall"])


if __name__ == "__main__":
    main()
