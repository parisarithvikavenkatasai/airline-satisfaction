"""Stage 09 - Batch inference with the production model (same preprocessing as the API)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import pandas as pd
from api.preprocessing import prepare_features
from src.common import load_config, p, read_json, ensure_dir, get_logger, append_run_log, now_iso
from src.registry import get_alias
from src.train import load_pickle

log = get_logger("09_inference")


def main(n: int = 200):
    cfg = load_config()
    prod = get_alias("production")
    if prod is None:
        log.error("No production model - run stages 05-08 first"); sys.exit(1)
    model = load_pickle(p(prod["model_file"])); pre = load_pickle(p(prod.get("preprocessor_file", "models/preprocessor.pkl")))
    meta = read_json(p(prod["metadata_file"])) if prod.get("metadata_file") else read_json(p(cfg["paths"]["processed_dir"], "dataset_metadata.json"))
    raw = pd.read_csv(p(cfg["paths"]["processed_dir"], "X_test_raw.csv")).head(n)
    X = prepare_features(raw, pre, meta["arrival_delay_median"])
    proba = model.predict_proba(X)[:, 1]
    log_path = p(cfg["paths"]["inference_log"]); ensure_dir(log_path.parent)
    with open(log_path, "a", encoding="utf-8") as f:
        for i, pr in enumerate(proba):
            f.write(json.dumps({"timestamp": now_iso(), "source": "batch", "row": i, "model_version": prod["version"],
                                "probability_satisfied": round(float(pr), 6),
                                "prediction": "satisfied" if pr >= prod["threshold"] else "neutral or dissatisfied"}) + "\n")
    append_run_log("09_inference", "SUCCESS", {"rows": len(proba)})
    log.info("Scored %d rows -> %s", len(proba), log_path)


if __name__ == "__main__":
    main()
