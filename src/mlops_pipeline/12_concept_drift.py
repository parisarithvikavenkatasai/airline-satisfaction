"""Stage 12 - Concept drift: accuracy/F1 of production model on labeled drifted data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from src.common import load_config, p, write_json, get_logger, now_iso
from src.monitoring_utils import load_production, load_split, predict_proba
from src.preprocess import encode_target

log = get_logger("12_concept_drift")


def score(df, cfg, model, pre, meta, thr):
    y = encode_target(df[cfg["data"]["target_column"]], cfg["data"]["positive_label"])
    pred = (predict_proba(df, model, pre, meta) >= thr).astype(int)
    return {"accuracy": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred))}


def main():
    cfg = load_config(); drop_thr = cfg["monitoring"]["performance_drop_threshold"]
    model, pre, meta, thr, prod = load_production(cfg)
    base = score(load_split(cfg, "test"), cfg, model, pre, meta, thr)
    cur = score(pd.read_csv(p(cfg["paths"]["drift_dir"], "concept_drift_data.csv")), cfg, model, pre, meta, thr)
    drop = base["f1"] - cur["f1"]
    report = {"timestamp": now_iso(), "model_version": prod["version"], "baseline": base, "current": cur,
              "f1_drop": round(drop, 4), "concept_drift_detected": drop > drop_thr}
    write_json(p(cfg["paths"]["reports_dir"], "concept_drift_report.json"), report)
    log.info("Concept drift: F1 %.4f -> %.4f (drop %.4f)", base["f1"], cur["f1"], drop)


if __name__ == "__main__":
    main()
