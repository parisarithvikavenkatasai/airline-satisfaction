"""Stage 12 - Simulate covariate (data) drift: new passenger mix + worse service/delays."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from src.common import load_config, p, ensure_dir, get_logger
from src.monitoring_utils import load_split, refresh_engineered

log = get_logger("12_create_evolved_dataset")


def main(n: int = 4000, seed: int = 7):
    cfg = load_config(); rng = np.random.default_rng(seed)
    df = load_split(cfg, "test").sample(n, replace=True, random_state=seed).reset_index(drop=True)
    target = cfg["data"]["target_column"]
    df["Age"] = (df["Age"] + rng.integers(4, 12, n)).clip(upper=85)
    df["Departure Delay in Minutes"] = (df["Departure Delay in Minutes"] * rng.uniform(1.5, 3.0, n)).round()
    df["Arrival Delay in Minutes"] = (df["Arrival Delay in Minutes"] * rng.uniform(1.5, 3.0, n)).round()
    for col in ["Inflight wifi service", "Food and drink", "Seat comfort"]:
        df[col] = (df[col] - rng.integers(0, 2, n)).clip(lower=0)
    flip = rng.random(n) < 0.25
    df.loc[flip, "Customer Type"] = "disloyal Customer"
    df.loc[rng.random(n) < 0.15, "Class"] = "Eco"
    df = refresh_engineered(df, cfg)
    out = ensure_dir(p(cfg["paths"]["drift_dir"]))
    df.to_csv(out / "evolved_labeled_data.csv", index=False)
    df.drop(columns=[target]).to_csv(out / "evolved_data.csv", index=False)
    log.info("Wrote evolved datasets (%d rows) to %s", n, out)


if __name__ == "__main__":
    main()
