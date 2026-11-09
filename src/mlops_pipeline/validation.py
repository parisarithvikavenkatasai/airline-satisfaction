"""Standalone validation CLI: python src/mlops_pipeline/validation.py <csv> [--no-target]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import pandas as pd
from src.common import load_config
from src.validate_data import validate_dataframe, has_errors

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--no-target", action="store_true")
    a = ap.parse_args()
    issues = validate_dataframe(pd.read_csv(a.csv), load_config(), require_target=not a.no_target)
    for i in issues: print(f"[{i['severity']}] {i['check']}: {i['detail']}")
    print("INVALID" if has_errors(issues) else "VALID"); sys.exit(1 if has_errors(issues) else 0)
