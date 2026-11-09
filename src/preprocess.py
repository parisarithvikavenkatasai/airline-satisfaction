"""Cleaning, feature engineering and preprocessor construction."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def drop_id_columns(df: pd.DataFrame, id_cols) -> pd.DataFrame:
    return df.drop(columns=[c for c in id_cols if c in df.columns])


def clean_raw_data(df: pd.DataFrame, id_cols=None, arrival_delay_median=None) -> pd.DataFrame:
    """Drop ids and impute missing Arrival Delay (median)."""
    df = df.copy()
    if id_cols:
        df = drop_id_columns(df, id_cols)
    if "Arrival Delay in Minutes" in df.columns:
        med = arrival_delay_median if arrival_delay_median is not None else df["Arrival Delay in Minutes"].median()
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(med)
    return df


def engineer_features(df: pd.DataFrame, survey_cols) -> pd.DataFrame:
    """Add TotalDelay and CompositeServiceScore."""
    df = df.copy()
    if {"Departure Delay in Minutes", "Arrival Delay in Minutes"} <= set(df.columns):
        df["TotalDelay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    present = [c for c in survey_cols if c in df.columns]
    if present:
        df["CompositeServiceScore"] = np.round(df[present].mean(axis=1), 3)
    return df


def encode_target(y: pd.Series, positive="satisfied") -> pd.Series:
    return (y.astype(str).str.strip().str.lower() == positive.lower()).astype(int)


def build_preprocessor(categorical_cols, numerical_cols) -> ColumnTransformer:
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, numerical_cols), ("cat", cat, categorical_cols)], remainder="drop")


def get_feature_names(preprocessor: ColumnTransformer):
    return list(preprocessor.get_feature_names_out())
