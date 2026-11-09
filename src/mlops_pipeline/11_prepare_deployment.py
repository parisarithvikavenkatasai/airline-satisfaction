"""Stage 11 - Package production model for serving (deployment/ folder)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment.deploy_production import deploy
from src.common import get_logger, append_run_log, load_config, p, write_json, now_iso

log = get_logger("11_prepare_deployment")


def main():
    meta = deploy()
    cfg = load_config()
    write_json(p(cfg["paths"]["reports_dir"], "deployment_automation_report.json"),
               {"timestamp": now_iso(), "status": "READY", "model_version": meta["registry_version"],
                "files": ["deployment/model.pkl", "deployment/preprocessor.pkl", "deployment/model_metadata.json"]})
    append_run_log("11_prepare_deployment", "SUCCESS", {"version": meta["registry_version"]})
    log.info("Deployment package ready (v%s)", meta["registry_version"])


if __name__ == "__main__":
    main()
