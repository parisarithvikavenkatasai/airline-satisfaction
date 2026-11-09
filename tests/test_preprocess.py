import numpy as np
import pandas as pd

from src.common import load_config
from src.preprocess import build_preprocessor, clean_raw_data, encode_target, engineer_features


def _sample():
    return pd.DataFrame({
        "Gender": ["Male", "Female"], "Customer Type": ["Loyal Customer", "disloyal Customer"],
        "Age": [30, 45], "Type of Travel": ["Business travel", "Personal Travel"], "Class": ["Eco", "Business"],
        "Flight Distance": [500, 2000], "Inflight wifi service": [3, 5], "Departure/Arrival time convenient": [3, 4],
        "Ease of Online booking": [3, 4], "Gate location": [3, 3], "Food and drink": [3, 5], "Online boarding": [3, 4],
        "Seat comfort": [3, 5], "Inflight entertainment": [3, 5], "On-board service": [3, 4], "Leg room service": [3, 4],
        "Baggage handling": [3, 4], "Checkin service": [3, 4], "Inflight service": [3, 4], "Cleanliness": [3, 5],
        "Departure Delay in Minutes": [10, 0], "Arrival Delay in Minutes": [np.nan, 5.0],
        "satisfaction": ["neutral or dissatisfied", "satisfied"]})


def test_encode_target():
    y = encode_target(pd.Series(["satisfied", "neutral or dissatisfied", "satisfied"]))
    assert y.tolist() == [1, 0, 1]


def test_clean_imputes_arrival_delay_and_drops_ids():
    df = _sample(); df["id"] = [1, 2]
    out = clean_raw_data(df, ["id"], arrival_delay_median=7.0)
    assert "id" not in out.columns
    assert out["Arrival Delay in Minutes"].isna().sum() == 0
    assert out.loc[0, "Arrival Delay in Minutes"] == 7.0


def test_engineered_features():
    cfg = load_config()
    out = engineer_features(clean_raw_data(_sample(), arrival_delay_median=0.0), cfg["features"]["survey_columns"])
    assert out.loc[1, "TotalDelay"] == 5.0
    assert abs(out.loc[0, "CompositeServiceScore"] - 3.0) < 1e-6


def test_preprocessor_output_has_no_nan_and_handles_unknown_category():
    cfg = load_config()
    df = engineer_features(clean_raw_data(_sample(), arrival_delay_median=0.0), cfg["features"]["survey_columns"]).drop(columns=["satisfaction"])
    pre = build_preprocessor(cfg["features"]["categorical"], cfg["features"]["numerical"]).fit(df)
    unseen = df.copy(); unseen.loc[0, "Class"] = "First"
    out = pre.transform(unseen)
    assert not np.isnan(out).any()
    assert out.shape[1] == pre.transform(df).shape[1]
