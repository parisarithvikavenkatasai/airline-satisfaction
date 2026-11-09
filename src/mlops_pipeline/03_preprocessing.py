"""Stage 03 - Clean, engineer features, split, fit preprocessor, save arrays."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.common import load_config, p, ensure_dir, write_json, get_logger, append_run_log
from src.preprocess import (clean_raw_data, engineer_features, encode_target, build_preprocessor)
from src.train import save_pickle

log = get_logger("03_preprocessing")


def main():
    cfg = load_config()
    df = pd.read_csv(p(cfg["paths"]["raw_data"]))
    target = cfg["data"]["target_column"]
    id_col = "id" if "id" in df.columns else None
    ids = df[id_col] if id_col else pd.Series(df.index, name="id")

    arrival_median = float(df["Arrival Delay in Minutes"].median())
    df = clean_raw_data(df, cfg["data"]["id_columns"], arrival_median)
    df = engineer_features(df, cfg["features"]["survey_columns"])

    y = encode_target(df[target], cfg["data"]["positive_label"])
    X = df.drop(columns=[target])
    num_cols = [c for c in cfg["features"]["numerical"] if c in X.columns]
    cat_cols = [c for c in cfg["features"]["categorical"] if c in X.columns]

    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, ids, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"], stratify=y)

    pre = build_preprocessor(cat_cols, num_cols)
    X_tr_f = pre.fit_transform(X_tr)
    X_te_f = pre.transform(X_te)

    out = ensure_dir(p(cfg["paths"]["processed_dir"]))
    X_tr.to_csv(out / "X_train_raw.csv", index=False)
    X_te.to_csv(out / "X_test_raw.csv", index=False)
    np.save(out / "X_train_final.npy", X_tr_f)
    np.save(out / "X_test_final.npy", X_te_f)
    np.save(out / "y_train.npy", y_tr.to_numpy())
    np.save(out / "y_test.npy", y_te.to_numpy())
    id_te.to_csv(out / "test_passenger_ids.csv", index=False)

    models_dir = ensure_dir(p(cfg["paths"]["models_dir"]))
    save_pickle(pre, models_dir / "preprocessor.pkl")

    meta = {
        "dataset_name": "Airline Passenger Satisfaction",
        "train_shape": list(X_tr_f.shape), "test_shape": list(X_te_f.shape),
        "numerical_features": num_cols, "categorical_features": cat_cols,
        "feature_names_out": list(pre.get_feature_names_out()),
        "arrival_delay_median": arrival_median,
        "positive_rate_train": float(y_tr.mean()), "positive_rate_test": float(y_te.mean()),
    }
    write_json(out / "dataset_metadata.json", meta)
    append_run_log("03_preprocessing", "SUCCESS", {"train": meta["train_shape"], "test": meta["test_shape"]})
    log.info("Train %s | Test %s", X_tr_f.shape, X_te_f.shape)


if __name__ == "__main__":
    main()
