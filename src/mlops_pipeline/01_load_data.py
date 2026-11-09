"""Stage 01 - Load raw data and write a load report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, write_json, get_logger, append_run_log, now_iso

log = get_logger("01_load_data")


def main():
    cfg = load_config()
    raw = p(cfg["paths"]["raw_data"])
    if not raw.exists():
        raise FileNotFoundError(f"Raw data not found: {raw}")
    df = pd.read_csv(raw)
    target = cfg["data"]["target_column"]
    report = {
        "timestamp": now_iso(),
        "source": str(cfg["paths"]["raw_data"]),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[target].value_counts().to_dict() if target in df else None,
        "status": "SUCCESS",
    }
    write_json(p(cfg["paths"]["artifacts_dir"], "data_load_report.json"), report)
    append_run_log("01_load_data", "SUCCESS", {"rows": report["rows"]})
    log.info("Loaded %s rows x %s cols", *df.shape)


if __name__ == "__main__":
    main()
