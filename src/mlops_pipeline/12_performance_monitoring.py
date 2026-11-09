"""Stage 12 - Production model performance on evolved (labeled) data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, write_json, get_logger, now_iso, append_run_log
from src.monitoring_utils import load_production, load_split, predict_proba
from src.preprocess import encode_target
from src.evaluate import compute_metrics

log = get_logger("12_performance_monitoring")


def main():
    cfg = load_config()
    model, pre, meta, thr, prod = load_production(cfg)
    t = cfg["data"]["target_column"]; pos = cfg["data"]["positive_label"]
    out = {}
    for name, df in {"reference_test": load_split(cfg, "test"),
                     "evolved": pd.read_csv(p(cfg["paths"]["drift_dir"], "evolved_labeled_data.csv")),
                     "concept_drift": pd.read_csv(p(cfg["paths"]["drift_dir"], "concept_drift_data.csv"))}.items():
        y = encode_target(df[t], pos); pr = predict_proba(df, model, pre, meta)
        out[name] = compute_metrics(y, (pr >= thr).astype(int), pr)
        log.info("%-14s acc=%.4f f1=%.4f", name, out[name]["accuracy"], out[name]["f1"])
    write_json(p(cfg["paths"]["reports_dir"], "performance_report.json"),
               {"timestamp": now_iso(), "model_version": prod["version"], "datasets": out})
    append_run_log("12_performance_monitoring", "SUCCESS", {k: round(v["f1"], 4) for k, v in out.items()})


if __name__ == "__main__":
    main()
