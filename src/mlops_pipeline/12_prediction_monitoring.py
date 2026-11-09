"""Stage 12 - Compare live prediction rate (inference log) with the training positive rate."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, read_json, write_json, get_logger, now_iso

log = get_logger("12_prediction_monitoring")


def main():
    cfg = load_config(); path = p(cfg["paths"]["inference_log"])
    if not path.exists() or path.stat().st_size == 0:
        log.warning("No inference log yet"); return
    df = pd.read_json(path, lines=True)
    meta = read_json(p(cfg["paths"]["processed_dir"], "dataset_metadata.json"))
    live = float((df["prediction"] == "satisfied").mean()); ref = meta["positive_rate_train"]
    rep = {"timestamp": now_iso(), "n_logged": int(len(df)), "training_positive_rate": round(ref, 4),
           "live_positive_rate": round(live, 4), "shift": round(abs(live - ref), 4),
           "alert": abs(live - ref) > cfg["monitoring"]["prediction_drift_threshold"] and len(df) >= 100}
    write_json(p(cfg["paths"]["reports_dir"], "prediction_monitoring_report.json"), rep)
    log.info("%s", rep)


if __name__ == "__main__":
    main()
