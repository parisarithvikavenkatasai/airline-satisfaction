import pandas as pd

from src.common import load_config, p
from src.validate_data import has_errors, validate_dataframe


def _raw():
    return pd.read_csv(p("data", "raw", "airline.csv"), nrows=500)


def test_clean_data_passes():
    assert not has_errors(validate_dataframe(_raw(), load_config()))


def test_missing_column_detected():
    assert has_errors(validate_dataframe(_raw().drop(columns=["Class"]), load_config()))


def test_out_of_range_rating_detected():
    df = _raw(); df.loc[df.index[:5], "Cleanliness"] = 42
    assert has_errors(validate_dataframe(df, load_config()))


def test_unknown_category_detected():
    df = _raw(); df.loc[df.index[:5], "Gender"] = "Other"
    assert has_errors(validate_dataframe(df, load_config()))


def test_bad_target_label_detected():
    df = _raw(); df.loc[df.index[:5], "satisfaction"] = "maybe"
    assert has_errors(validate_dataframe(df, load_config()))


def test_empty_dataset_detected():
    assert has_errors(validate_dataframe(_raw().iloc[0:0], load_config()))
