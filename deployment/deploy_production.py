"""Copy the production model + preprocessor into deployment/ and write metadata.

Usage: python deployment/deploy_production.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config, now_iso, p, read_json, write_json  # noqa: E402
from src.registry import get_alias  # noqa: E402


def deploy() -> dict:
    cfg = load_config()
    prod = get_alias("production")
    if prod is None:
        raise RuntimeError("No production model in registry. Run stages 05-08 first.")
    dep = p(cfg["paths"]["deployment_dir"]); dep.mkdir(exist_ok=True)
    shutil.copy2(p(prod["model_file"]), dep / "model.pkl")
    shutil.copy2(p(prod.get("preprocessor_file", "models/preprocessor.pkl")), dep / "preprocessor.pkl")
    meta_in = read_json(p(prod["metadata_file"])) if prod.get("metadata_file") else read_json(p(cfg["paths"]["processed_dir"], "dataset_metadata.json"))
    metadata = {
        "deployed_at": now_iso(), "model_name": cfg["mlflow"]["registered_model_name"],
        "registry_version": prod["version"], "model_type": prod["model_type"],
        "decision_threshold": prod["threshold"],
        "metrics": prod.get("test_metrics", prod["metrics"]),
        "arrival_delay_median": meta_in["arrival_delay_median"],
        "numerical_features": meta_in["numerical_features"],
        "categorical_features": meta_in["categorical_features"],
        "positive_label": cfg["data"]["positive_label"], "negative_label": cfg["data"]["negative_label"],
    }
    write_json(dep / "model_metadata.json", metadata)
    write_json(p(cfg["paths"]["artifacts_dir"], "production_model_report.json"), metadata)
    return metadata


if __name__ == "__main__":
    m = deploy()
    print(f"Deployed model v{m['registry_version']} ({m['model_type']}) to deployment/")
