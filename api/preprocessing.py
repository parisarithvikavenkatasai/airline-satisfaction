"""Request preprocessing shared by the API and the batch inference stage.

Uses the exact same cleaning + feature engineering code as training (src/preprocess.py)
to guarantee training/serving consistency.
"""
import pandas as pd

from src.preprocess import clean_raw_data, engineer_features

SURVEY_COLUMNS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
]


def prepare_features(df_raw: pd.DataFrame, preprocessor, arrival_delay_median: float):
    """Raw passenger records -> model-ready matrix."""
    df = clean_raw_data(df_raw, id_cols=["Unnamed: 0", "id", "satisfaction"], arrival_delay_median=arrival_delay_median)
    df = engineer_features(df, SURVEY_COLUMNS)
    return preprocessor.transform(df)
