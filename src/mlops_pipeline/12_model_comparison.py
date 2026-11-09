"""Stage 12 - Compare production vs retrained candidate on the SAME post-drift holdout."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, write_json, get_logger, now_iso
from src.evaluate import compute_metrics
from src.monitoring_utils import predict_proba
from src.preprocess import encode_target
from src.registry import get_alias
from src.common import read_json
from src.train import load_pickle

log = get_logger("12_model_comparison")


def eval_version(v, df, cfg):
    model = load_pickle(p(v["model_file"])); pre = load_pickle(p(v.get("preprocessor_file", "models/preprocessor.pkl")))
    meta = read_json(p(v["metadata_file"])) if v.get("metadata_file") else read_json(p(cfg["paths"]["processed_dir"], "dataset_metadata.json"))
    y = encode_target(df[cfg["data"]["target_column"]], cfg["data"]["positive_label"])
    pr = predict_proba(df, model, pre, meta)
    return compute_metrics(y, (pr >= v["threshold"]).astype(int), pr)


def main():
    cfg = load_config(); hold = pd.read_csv(p(cfg["paths"]["drift_dir"], "updated_holdout.csv"))
    prod, cand = get_alias("production"), get_alias("candidate")
    mp, mc = eval_version(prod, hold, cfg), eval_version(cand, hold, cfg)
    better = mc["f1"] > mp["f1"] + cfg["monitoring"]["min_improvement_to_promote"] and cand["version"] != prod["version"]
    write_json(p(cfg["paths"]["reports_dir"], "model_comparison_report.json"), {
        "timestamp": now_iso(), "holdout_rows": len(hold),
        "production": {"version": prod["version"], "f1": mp["f1"], "accuracy": mp["accuracy"]},
        "candidate": {"version": cand["version"], "f1": mc["f1"], "accuracy": mc["accuracy"]},
        "promote_candidate": bool(better)})
    log.info("production v%d F1=%.4f | candidate v%d F1=%.4f | promote=%s", prod["version"], mp["f1"], cand["version"], mc["f1"], better)


if __name__ == "__main__":
    main()
