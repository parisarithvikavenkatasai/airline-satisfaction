"""Stage 12 - Simulate concept drift: same features, but expectations changed so labels differ."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.common import load_config, p, ensure_dir, get_logger
from src.monitoring_utils import load_split

log = get_logger("12_create_concept_drift_dataset")


def main(n: int = 4000, seed: int = 11):
    cfg = load_config(); rng = np.random.default_rng(seed)
    t = cfg["data"]["target_column"]; pos, neg = cfg["data"]["positive_label"], cfg["data"]["negative_label"]
    df = load_split(cfg, "test").sample(n, replace=True, random_state=seed).reset_index(drop=True)
    # Business travellers become harder to please; wifi/online boarding now matter more
    harsh = (df["Type of Travel"] == "Business travel") & (df["CompositeServiceScore"] < 4.2) & (rng.random(n) < 0.6)
    lenient_low_wifi = (df["Inflight wifi service"] <= 2) & (df[t] == pos) & (rng.random(n) < 0.5)
    df.loc[harsh | lenient_low_wifi, t] = neg
    out = ensure_dir(p(cfg["paths"]["drift_dir"])) / "concept_drift_data.csv"
    df.to_csv(out, index=False)
    log.info("Wrote concept-drift dataset (%d rows, positive rate %.2f)", n, (df[t] == pos).mean())


if __name__ == "__main__":
    main()
