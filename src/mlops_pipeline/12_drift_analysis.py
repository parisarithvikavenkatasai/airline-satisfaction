"""Stage 12 - Aggregate data, prediction, concept drift and performance into one report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import load_config, p, read_json, write_json, get_logger, now_iso

log = get_logger("12_drift_analysis")


def main():
    cfg = load_config(); r = p(cfg["paths"]["reports_dir"])
    data, pred, conc = (read_json(r / f) for f in ["data_drift_report.json", "prediction_drift_report.json", "concept_drift_report.json"])
    perf = read_json(r / "performance_report.json")["datasets"]
    summary = {
        "timestamp": now_iso(),
        "data_drift": {"detected": data["data_drift_detected"], "drift_fraction": data["drift_fraction"], "features": data["drifted_features"]},
        "prediction_drift": {"detected": pred["prediction_drift_detected"], "shift": pred["positive_rate_shift"]},
        "concept_drift": {"detected": conc["concept_drift_detected"], "f1_drop": conc["f1_drop"]},
        "f1": {k: round(v["f1"], 4) for k, v in perf.items()},
    }
    write_json(r / "drift_analysis_report.json", summary)
    log.info("%s", summary)


if __name__ == "__main__":
    main()
