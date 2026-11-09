"""Stage 12 - Build post-drift training data = original train + evolved + concept-drift rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, get_logger
from src.monitoring_utils import load_split

log = get_logger("12_create_updated_dataset")


def main():
    cfg = load_config(); dd = p(cfg["paths"]["drift_dir"])
    parts = [load_split(cfg, "train"), pd.read_csv(dd / "evolved_labeled_data.csv"), pd.read_csv(dd / "concept_drift_data.csv")]
    full = pd.concat(parts, ignore_index=True)
    full.to_csv(dd / "post_drift_training_data.csv", index=False)
    full.to_csv(p(cfg["paths"]["processed_dir"], "updated_training_data.csv"), index=False)
    log.info("Updated training data: %d rows", len(full))


if __name__ == "__main__":
    main()
