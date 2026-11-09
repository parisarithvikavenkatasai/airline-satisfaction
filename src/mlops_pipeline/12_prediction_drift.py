"""Stage 12 - Prediction drift: shift in predicted probabilities / positive rate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, write_json, get_logger, now_iso
from src.monitoring_utils import load_production, load_split, predict_proba, psi

log = get_logger("12_prediction_drift")


def main():
    cfg = load_config(); m = cfg["monitoring"]
    model, pre, meta, thr, prod = load_production(cfg)
    ref = load_split(cfg, "test"); cur = pd.read_csv(p(cfg["paths"]["drift_dir"], "evolved_data.csv"))
    pr, pc = predict_proba(ref, model, pre, meta), predict_proba(cur, model, pre, meta)
    ref_rate, cur_rate = float((pr >= thr).mean()), float((pc >= thr).mean())
    report = {"timestamp": now_iso(), "model_version": prod["version"],
              "reference_positive_rate": round(ref_rate, 4), "current_positive_rate": round(cur_rate, 4),
              "positive_rate_shift": round(abs(cur_rate - ref_rate), 4),
              "probability_psi": round(psi(pr, pc), 4),
              "prediction_drift_detected": abs(cur_rate - ref_rate) > m["prediction_drift_threshold"]}
    write_json(p(cfg["paths"]["reports_dir"], "prediction_drift_report.json"), report)
    log.info("Prediction drift: positive rate %.3f -> %.3f (psi %.3f)", ref_rate, cur_rate, report["probability_psi"])


if __name__ == "__main__":
    main()
