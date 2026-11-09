"""Stage 12 - Feature (covariate) drift: PSI + KS test, reference=train vs current=evolved_data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, write_json, get_logger, now_iso
from src.monitoring_utils import drift_table, load_split

log = get_logger("12_data_drift")


def main(current_csv=None):
    cfg = load_config(); m = cfg["monitoring"]
    ref = load_split(cfg, "train")
    cur = pd.read_csv(current_csv or p(cfg["paths"]["drift_dir"], "evolved_data.csv"))
    tbl = drift_table(ref, cur, cfg["features"]["numerical"], cfg["features"]["categorical"],
                      m["psi_threshold"], m["ks_pvalue_threshold"])
    tbl.to_csv(p(cfg["paths"]["artifacts_dir"], "data_drift_table.csv"), index=False)
    drifted = tbl[tbl["drift"]]["feature"].tolist()
    report = {"timestamp": now_iso(), "n_features": len(tbl), "n_drifted": len(drifted),
              "drift_fraction": round(len(drifted) / max(len(tbl), 1), 3), "drifted_features": drifted,
              "top_psi": tbl.sort_values("psi", ascending=False).head(5)[["feature", "psi"]].to_dict("records"),
              "data_drift_detected": len(drifted) > 0}
    write_json(p(cfg["paths"]["reports_dir"], "data_drift_report.json"), report)
    log.info("Data drift: %d/%d features drifted %s", len(drifted), len(tbl), drifted[:6])


if __name__ == "__main__":
    main(next((a for a in sys.argv[1:] if a.endswith(".csv")), None))
