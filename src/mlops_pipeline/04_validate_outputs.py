"""Stage 04 - Validate preprocessing outputs (shapes, NaNs, label balance)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.common import load_config, p, write_json, get_logger, append_run_log, now_iso

log = get_logger("04_validate_outputs")


def main():
    cfg = load_config()
    d = p(cfg["paths"]["processed_dir"])
    Xtr, Xte = np.load(d / "X_train_final.npy"), np.load(d / "X_test_final.npy")
    ytr, yte = np.load(d / "y_train.npy"), np.load(d / "y_test.npy")
    errors = []
    if Xtr.shape[0] != ytr.shape[0]: errors.append("X_train/y_train row mismatch")
    if Xte.shape[0] != yte.shape[0]: errors.append("X_test/y_test row mismatch")
    if Xtr.shape[1] != Xte.shape[1]: errors.append("train/test feature count mismatch")
    nan_tr, nan_te = int(np.isnan(Xtr).sum()), int(np.isnan(Xte).sum())
    if nan_tr or nan_te: errors.append("NaNs present in processed matrices")
    if set(np.unique(ytr)) - {0, 1}: errors.append("non-binary labels")
    report = {
        "timestamp": now_iso(),
        "validation_status": "FAILED" if errors else "PASSED",
        "matrix_dimensions": {"X_train_shape": list(Xtr.shape), "X_test_shape": list(Xte.shape),
                              "y_train_shape": list(ytr.shape), "y_test_shape": list(yte.shape)},
        "data_quality": {"missing_values_X_train": nan_tr, "missing_values_X_test": nan_te},
        "class_balance": {"train_positive_rate": float(ytr.mean()), "test_positive_rate": float(yte.mean())},
        "errors": errors,
    }
    write_json(p(cfg["paths"]["artifacts_dir"], "preprocessing_summary_report.json"), report)
    append_run_log("04_validate_outputs", report["validation_status"], {"errors": errors})
    if errors:
        log.error("Output validation FAILED: %s", errors); sys.exit(1)
    log.info("Output validation PASSED")


if __name__ == "__main__":
    main()
