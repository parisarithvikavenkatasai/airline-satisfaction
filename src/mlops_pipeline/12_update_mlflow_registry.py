"""Stage 12 - Promote the retrained model if the comparison says so, then redeploy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment.deploy_production import deploy
from src.common import load_config, p, read_json, write_json, get_logger, append_run_log, now_iso
from src.registry import get_alias, set_alias

log = get_logger("12_update_mlflow_registry")


def main():
    cfg = load_config(); cmp_ = read_json(p(cfg["paths"]["reports_dir"], "model_comparison_report.json"))
    if cmp_["promote_candidate"]:
        v = get_alias("candidate")["version"]; set_alias("production", v); meta = deploy()
        msg = f"promoted v{v} and redeployed"
    else:
        msg = "kept current production model"
    write_json(p(cfg["paths"]["reports_dir"], "lifecycle_report.json"),
               {"timestamp": now_iso(), "decision": "promoted" if cmp_["promote_candidate"] else "kept_current",
                "reason": msg, "comparison": cmp_})
    append_run_log("12_update_mlflow_registry", "PROMOTED" if cmp_["promote_candidate"] else "KEPT", {"msg": msg})
    log.info(msg)


if __name__ == "__main__":
    main()
