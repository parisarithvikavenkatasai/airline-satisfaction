"""Drift statistics shared by the stage-12 monitoring scripts."""
import numpy as np
import pandas as pd
from scipy import stats


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index for a numeric feature (quantile bins from `expected`)."""
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.dropna().unique()) | set(actual.dropna().unique()))
    e = expected.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-6)
    a = actual.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def ks_test(expected, actual):
    res = stats.ks_2samp(pd.Series(expected).dropna(), pd.Series(actual).dropna())
    return float(res.statistic), float(res.pvalue)


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, num_cols, cat_cols, psi_thr, ks_thr) -> pd.DataFrame:
    rows = []
    for c in num_cols:
        if c not in ref or c not in cur:
            continue
        stat, pval = ks_test(ref[c], cur[c])
        v = psi(ref[c], cur[c])
        rows.append({"feature": c, "type": "numeric", "psi": round(v, 4), "ks_stat": round(stat, 4),
                     "ks_pvalue": pval, "drift": bool(v > psi_thr or pval < ks_thr)})
    for c in cat_cols:
        if c not in ref or c not in cur:
            continue
        v = categorical_psi(ref[c], cur[c])
        rows.append({"feature": c, "type": "categorical", "psi": round(v, 4), "ks_stat": None,
                     "ks_pvalue": None, "drift": bool(v > psi_thr)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- helpers for stage 12
import numpy as np  # noqa: E402

from src.common import load_config, p, read_json  # noqa: E402
from src.preprocess import engineer_features  # noqa: E402
from src.registry import get_alias  # noqa: E402
from src.train import load_pickle  # noqa: E402


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    """Raw-format features + string target for the 'train' or 'test' split."""
    d = p(cfg["paths"]["processed_dir"])
    X = pd.read_csv(d / f"X_{split}_raw.csv")
    y = np.load(d / f"y_{split}.npy")
    X[cfg["data"]["target_column"]] = np.where(y == 1, cfg["data"]["positive_label"], cfg["data"]["negative_label"])
    return X


def load_production(cfg: dict):
    """Return (model, preprocessor, metadata dict, threshold, version dict) for production."""
    prod = get_alias("production")
    if prod is None:
        raise RuntimeError("No production model registered. Run the pipeline first.")
    model = load_pickle(p(prod["model_file"]))
    pre = load_pickle(p(prod.get("preprocessor_file", "models/preprocessor.pkl")))
    meta_path = prod.get("metadata_file") or str(p(cfg["paths"]["processed_dir"], "dataset_metadata.json"))
    meta = read_json(meta_path)
    return model, pre, meta, prod["threshold"], prod


def predict_proba(df: pd.DataFrame, model, pre, meta) -> np.ndarray:
    from api.preprocessing import prepare_features
    return model.predict_proba(prepare_features(df, pre, meta["arrival_delay_median"]))[:, 1]


def refresh_engineered(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return engineer_features(df, cfg["features"]["survey_columns"])
