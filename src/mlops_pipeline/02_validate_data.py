"""Stage 02 - Validate raw data; fails the pipeline on schema/range errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, get_logger, append_run_log
from src.validate_data import validate_dataframe, has_errors

log = get_logger("02_validate_data")


def main():
    cfg = load_config()
    df = pd.read_csv(p(cfg["paths"]["raw_data"]))
    issues = validate_dataframe(df, cfg)
    out = p(cfg["paths"]["artifacts_dir"], "baseline_validation.csv")
    pd.DataFrame(issues, columns=["check", "severity", "detail"]).to_csv(out, index=False)
    for i in issues:
        log.info("[%s] %s: %s", i["severity"].upper(), i["check"], i["detail"])
    if has_errors(issues):
        append_run_log("02_validate_data", "FAILED", {"issues": len(issues)})
        log.error("Validation FAILED")
        sys.exit(1)
    append_run_log("02_validate_data", "SUCCESS", {"warnings": len(issues)})
    log.info("Validation PASSED (%d warnings)", len(issues))


if __name__ == "__main__":
    main()
