"""Stage 07 - Quality gate: block promotion if metrics are below thresholds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import load_config, p, read_json, write_json, get_logger, append_run_log, now_iso

log = get_logger("07_quality_gate")


def main():
    cfg = load_config()
    gate = cfg["quality_gate"]
    ev = read_json(p(cfg["paths"]["reports_dir"], "evaluation_report.json"))
    m = ev["test_metrics"]
    checks = {
        "accuracy": (m["accuracy"], gate["min_accuracy"]),
        "f1": (m["f1"], gate["min_f1"]),
        "roc_auc": (m["roc_auc"], gate["min_roc_auc"]),
        "recall": (m["recall"], gate["min_recall"]),
    }
    results = {k: {"value": round(v, 4), "threshold": t, "passed": v >= t} for k, (v, t) in checks.items()}
    passed = all(r["passed"] for r in results.values())
    write_json(p(cfg["paths"]["reports_dir"], "quality_gate_report.json"), {
        "timestamp": now_iso(), "registry_version": ev["registry_version"],
        "gate_passed": passed, "checks": results})
    append_run_log("07_quality_gate", "PASSED" if passed else "FAILED", results)
    for k, r in results.items():
        log.info("%-9s %.4f (>= %.2f) %s", k, r["value"], r["threshold"], "OK" if r["passed"] else "FAIL")
    if not passed:
        log.error("Quality gate FAILED - model will not be promoted"); sys.exit(1)
    log.info("Quality gate PASSED")


if __name__ == "__main__":
    main()
