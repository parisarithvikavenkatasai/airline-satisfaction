"""Raw data validation rules (schema, ranges, categories, missing values)."""
import pandas as pd

REQUIRED_COLUMNS = [
    "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance",
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
    "Departure Delay in Minutes", "Arrival Delay in Minutes",
]


def validate_dataframe(df: pd.DataFrame, cfg: dict, require_target: bool = True) -> list[dict]:
    """Return a list of issue dicts. Empty list == valid."""
    issues = []
    vcfg = cfg["validation"]
    target = cfg["data"]["target_column"]

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append({"check": "schema", "severity": "error", "detail": f"missing columns: {missing_cols}"})
        return issues  # cannot continue safely
    if require_target and target not in df.columns:
        issues.append({"check": "schema", "severity": "error", "detail": f"missing target '{target}'"})

    if len(df) == 0:
        issues.append({"check": "rows", "severity": "error", "detail": "dataset is empty"})
        return issues

    # missing values
    frac = df[REQUIRED_COLUMNS].isna().mean()
    for col, f in frac[frac > 0].items():
        sev = "error" if f > vcfg["max_missing_fraction"] else "warning"
        issues.append({"check": "missing", "severity": sev, "detail": f"{col}: {f:.2%} missing"})

    # categories
    for col, allowed in vcfg["allowed_categories"].items():
        bad = df[col].dropna()[~df[col].dropna().isin(allowed)].unique().tolist()
        if bad:
            issues.append({"check": "category", "severity": "error", "detail": f"{col}: unexpected values {bad[:5]}"})

    # ratings
    lo, hi = vcfg["rating_range"]
    for col in cfg["features"]["survey_columns"]:
        s = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(((s < lo) | (s > hi)).sum() + (s.isna() & df[col].notna()).sum())
        if n_bad:
            issues.append({"check": "range", "severity": "error", "detail": f"{col}: {n_bad} values outside [{lo},{hi}]"})

    # age, distance, delays
    age = pd.to_numeric(df["Age"], errors="coerce")
    n_bad = int(((age < vcfg["age_range"][0]) | (age > vcfg["age_range"][1])).sum())
    if n_bad:
        issues.append({"check": "range", "severity": "error", "detail": f"Age: {n_bad} values outside {vcfg['age_range']}"})
    for col in ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
        s = pd.to_numeric(df[col], errors="coerce")
        if (s < 0).any():
            issues.append({"check": "range", "severity": "error", "detail": f"{col}: negative values"})
        if (s.isna() & df[col].notna()).any():
            issues.append({"check": "dtype", "severity": "error", "detail": f"{col}: non-numeric values"})

    # target labels
    if require_target and target in df.columns:
        allowed = {cfg["data"]["positive_label"], cfg["data"]["negative_label"]}
        bad = set(df[target].dropna().unique()) - allowed
        if bad:
            issues.append({"check": "target", "severity": "error", "detail": f"unexpected labels {sorted(map(str, bad))}"})
    return issues


def has_errors(issues) -> bool:
    return any(i["severity"] == "error" for i in issues)
