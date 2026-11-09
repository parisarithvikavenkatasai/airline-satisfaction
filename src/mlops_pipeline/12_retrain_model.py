"""Stage 12 - Retrain on post-drift data with a freshly fitted preprocessor (v2 artifacts)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from src.common import load_config, p, write_json, get_logger, append_run_log, now_iso
from src.evaluate import compute_metrics
from src.preprocess import build_preprocessor, clean_raw_data, encode_target, engineer_features
from src.registry import register_version, get_alias
from src.train import get_models, save_pickle

log = get_logger("12_retrain_model")


def main():
    cfg = load_config(); t = cfg["data"]["target_column"]
    df = pd.read_csv(p(cfg["paths"]["drift_dir"], "post_drift_training_data.csv"))
    median = float(df["Arrival Delay in Minutes"].median())
    df = engineer_features(clean_raw_data(df, cfg["data"]["id_columns"], median), cfg["features"]["survey_columns"])
    y = encode_target(df[t], cfg["data"]["positive_label"]); X = df.drop(columns=[t])
    num = [c for c in cfg["features"]["numerical"] if c in X]; cat = [c for c in cfg["features"]["categorical"] if c in X]
    Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, random_state=cfg["data"]["random_state"], stratify=y)
    pre = build_preprocessor(cat, num); Xtr_f = pre.fit_transform(Xtr); Xho_f = pre.transform(Xho)

    prod = get_alias("production")
    model = clone(get_models(cfg["models"])[prod["model_type"]]).fit(Xtr_f, ytr)
    proba = model.predict_proba(Xho_f)[:, 1]
    metrics = compute_metrics(yho, (proba >= 0.5).astype(int), proba)

    md = p(cfg["paths"]["models_dir"])
    save_pickle(model, md / "retrained_model_v2.pkl"); save_pickle(pre, md / "retrained_preprocessor_v2.pkl")
    Xho.assign(**{t: yho.map({1: cfg["data"]["positive_label"], 0: cfg["data"]["negative_label"]})})         .to_csv(p(cfg["paths"]["drift_dir"], "updated_holdout.csv"), index=False)
    meta = {"arrival_delay_median": median, "numerical_features": num, "categorical_features": cat}
    write_json(md / "retrained_metadata_v2.json", meta)
    ver = register_version("models/retrained_model_v2.pkl", prod["model_type"], metrics, 0.5,
                           preprocessor_file="models/retrained_preprocessor_v2.pkl", source="drift_retraining")
    from src.registry import update_version
    update_version(ver, metadata_file="models/retrained_metadata_v2.json")
    append_run_log("12_retrain_model", "SUCCESS", {"version": ver, "f1": round(metrics["f1"], 4)})
    log.info("Retrained v%d on %d rows; holdout F1=%.4f", ver, len(df), metrics["f1"])


if __name__ == "__main__":
    main()
