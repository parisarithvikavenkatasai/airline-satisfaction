"""Stage 12 - Decide whether retraining is needed. Exit code 0 always; decision is in the report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import load_config, p, read_json, write_json, get_logger, now_iso

log = get_logger("12_monitor_trigger")


def main():
    cfg = load_config(); a = read_json(p(cfg["paths"]["reports_dir"], "drift_analysis_report.json"))
    reasons = []
    if a["concept_drift"]["detected"]: reasons.append(f"concept drift (F1 drop {a['concept_drift']['f1_drop']})")
    if a["data_drift"]["drift_fraction"] >= 0.25: reasons.append(f"data drift in {a['data_drift']['drift_fraction']:.0%} of features")
    if a["prediction_drift"]["detected"]: reasons.append(f"prediction drift (shift {a['prediction_drift']['shift']})")
    trigger = bool(reasons)
    write_json(p(cfg["paths"]["reports_dir"], "monitor_trigger_report.json"),
               {"timestamp": now_iso(), "retrain_required": trigger, "reasons": reasons})
    log.info("RETRAIN REQUIRED: %s %s", trigger, reasons)
    print(f"retrain_required={str(trigger).lower()}")


if __name__ == "__main__":
    main()
