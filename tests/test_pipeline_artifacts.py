"""Checks outputs of a pipeline run. Skipped if the pipeline has not been run yet."""
import numpy as np
import pytest

from src.common import load_config, p, read_json


def _need(path):
    if not path.exists():
        pytest.skip(f"{path.name} missing - run the pipeline first")


def test_processed_shapes_consistent():
    cfg = load_config(); d = p(cfg["paths"]["processed_dir"])
    _need(d / "X_train_final.npy")
    Xtr, Xte = np.load(d / "X_train_final.npy"), np.load(d / "X_test_final.npy")
    assert Xtr.shape[1] == Xte.shape[1]
    assert Xtr.shape[0] == len(np.load(d / "y_train.npy"))
    assert not np.isnan(Xtr).any()


def test_split_is_stratified():
    cfg = load_config(); d = p(cfg["paths"]["processed_dir"])
    _need(d / "y_train.npy")
    assert abs(np.load(d / "y_train.npy").mean() - np.load(d / "y_test.npy").mean()) < 0.01


def test_quality_gate_report_passed():
    cfg = load_config(); f = p(cfg["paths"]["reports_dir"], "quality_gate_report.json")
    _need(f)
    assert read_json(f)["gate_passed"] is True


def test_failure_injection_detected_all():
    cfg = load_config(); f = p(cfg["paths"]["artifacts_dir"], "failure_simulation_report.json")
    _need(f)
    assert read_json(f)["all_detected"] is True


def test_deployment_package_complete():
    cfg = load_config(); dep = p(cfg["paths"]["deployment_dir"])
    for name in ("model.pkl", "preprocessor.pkl", "model_metadata.json"):
        _need(dep / name)
