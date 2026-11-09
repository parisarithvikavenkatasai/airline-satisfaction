"""Stage 12 - Summarise logged prediction requests (batch + API) from inference_log.jsonl."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from src.common import load_config, p, write_json, get_logger, now_iso

log = get_logger("12_request_history")


def main():
    cfg = load_config(); path = p(cfg["paths"]["inference_log"])
    if not path.exists() or path.stat().st_size == 0:
        log.warning("No inference log yet"); return
    df = pd.read_json(path, lines=True)
    rep = {"timestamp": now_iso(), "total_requests": int(len(df)),
           "by_source": df["source"].value_counts().to_dict() if "source" in df else {},
           "positive_rate": round(float((df["prediction"] == "satisfied").mean()), 4),
           "mean_probability": round(float(df["probability_satisfied"].mean()), 4)}
    write_json(p(cfg["paths"]["reports_dir"], "request_history_report.json"), rep)
    log.info("%s", rep)


if __name__ == "__main__":
    main()
