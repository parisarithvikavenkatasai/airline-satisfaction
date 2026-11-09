"""Stage 08 - Promote candidate to production if gate passed and not worse than current."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import load_config, p, read_json, write_json, get_logger, append_run_log, now_iso
from src.registry import get_alias, set_alias, summary

log = get_logger("08_automate_lifecycle")


def main():
    cfg = load_config()
    gate = read_json(p(cfg["paths"]["reports_dir"], "quality_gate_report.json"))
    ev = read_json(p(cfg["paths"]["reports_dir"], "evaluation_report.json"))
    cand = get_alias("candidate"); prod = get_alias("production")
    decision, reason = "kept_current", ""
    if not gate["gate_passed"]:
        reason = "quality gate failed"
    elif prod is not None and prod["version"] == cand["version"]:
        reason = "candidate is already the production model"
    elif prod is None:
        set_alias("production", cand["version"]); decision, reason = "promoted", "no production model existed"
    else:
        prod_f1 = prod.get("test_metrics", prod["metrics"])["f1"]; new_f1 = ev["test_metrics"]["f1"]
        if new_f1 >= prod_f1 + cfg["monitoring"]["min_improvement_to_promote"]:
            set_alias("production", cand["version"]); decision, reason = "promoted", f"f1 {new_f1:.4f} >= {prod_f1:.4f}"
        else:
            reason = f"candidate f1 {new_f1:.4f} < production f1 {prod_f1:.4f}"
    write_json(p(cfg["paths"]["reports_dir"], "lifecycle_report.json"), {
        "timestamp": now_iso(), "decision": decision, "reason": reason,
        "candidate_version": cand["version"], "production_version": (get_alias("production") or {}).get("version"),
        "registry": summary()})
    append_run_log("08_automate_lifecycle", decision.upper(), {"reason": reason})
    log.info("Lifecycle decision: %s (%s)", decision, reason)


if __name__ == "__main__":
    main()
