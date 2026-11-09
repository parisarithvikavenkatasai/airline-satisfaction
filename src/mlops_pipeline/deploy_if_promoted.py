"""Deploy only if stage 08 promoted a new model (used by CI)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common import load_config, p, read_json
from deployment.deploy_production import deploy

if __name__ == "__main__":
    cfg = load_config()
    rep = read_json(p(cfg["paths"]["reports_dir"], "lifecycle_report.json"))
    if rep["decision"] == "promoted":
        m = deploy(); print(f"Deployed v{m['registry_version']}")
    else:
        print(f"No deployment: {rep['reason']}")
