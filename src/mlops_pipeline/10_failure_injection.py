"""Stage 10 - Inject data corruption and verify validation catches every scenario."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from src.common import load_config, p, write_json, get_logger, append_run_log, now_iso
from src.validate_data import validate_dataframe, has_errors

log = get_logger("10_failure_injection")


def scenarios(df: pd.DataFrame, rng) -> dict:
    out = {}
    out["missing_column"] = df.drop(columns=["Class"])
    d = df.copy(); d.loc[d.sample(frac=0.3, random_state=1).index, "Age"] = np.nan; out["heavy_missing_values"] = d
    d = df.copy(); d.loc[d.index[:50], "Inflight wifi service"] = 99; out["rating_out_of_range"] = d
    d = df.copy(); d.loc[d.index[:50], "Class"] = "First"; out["unknown_category"] = d
    d = df.copy(); d["Flight Distance"] = d["Flight Distance"].astype(object); d.loc[d.index[:20], "Flight Distance"] = "abc"; out["wrong_dtype"] = d
    d = df.copy(); d.loc[d.index[:30], "Departure Delay in Minutes"] = -100; out["negative_delay"] = d
    d = df.copy(); d.loc[d.index[:10], "satisfaction"] = "maybe"; out["bad_target_label"] = d
    out["empty_dataset"] = df.iloc[0:0]
    return out


def main():
    cfg = load_config()
    df = pd.read_csv(p(cfg["paths"]["raw_data"])).drop(columns=[c for c in cfg["data"]["id_columns"] if c in pd.read_csv(p(cfg["paths"]["raw_data"]), nrows=1).columns]).head(2000)
    rng = np.random.default_rng(42)
    rows, detected = [], 0
    for name, bad in scenarios(df, rng).items():
        issues = validate_dataframe(bad, cfg)
        caught = has_errors(issues); detected += caught
        rows.append({"scenario": name, "detected": caught, "n_issues": len(issues),
                     "first_issue": issues[0]["detail"] if issues else ""})
        log.info("%-22s detected=%s", name, caught)
    pd.DataFrame(rows).to_csv(p(cfg["paths"]["artifacts_dir"], "corrupted_simulation_errors.csv"), index=False)
    clean_ok = not has_errors(validate_dataframe(df, cfg))
    report = {"timestamp": now_iso(), "scenarios": len(rows), "detected": int(detected),
              "all_detected": detected == len(rows), "clean_data_passes": bool(clean_ok), "details": rows}
    write_json(p(cfg["paths"]["artifacts_dir"], "failure_simulation_report.json"), report)
    write_json(p(cfg["paths"]["reports_dir"], "failure_report.json"), report)
    ok = report["all_detected"] and clean_ok
    append_run_log("10_failure_injection", "SUCCESS" if ok else "FAILED", {"detected": detected, "total": len(rows)})
    if not ok:
        sys.exit(1)
    log.info("All %d failure scenarios detected; clean data passes", len(rows))


if __name__ == "__main__":
    main()
