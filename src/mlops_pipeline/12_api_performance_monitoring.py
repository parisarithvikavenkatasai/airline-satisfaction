"""Stage 12 - Measure live API latency/throughput. Requires the API to be running.

Usage: python src/mlops_pipeline/12_api_performance_monitoring.py [http://localhost:8000] [n_requests]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import statistics, time
import pandas as pd
import requests
from src.common import load_config, p, write_json, get_logger, now_iso

log = get_logger("12_api_performance_monitoring")
DROP = ["satisfaction", "TotalDelay", "CompositeServiceScore"]


def main(url="http://localhost:8000", n=100):
    cfg = load_config()
    rows = pd.read_csv(p(cfg["paths"]["processed_dir"], "X_test_raw.csv")).drop(columns=DROP).head(n).to_dict("records")
    lat, errors = [], 0
    t0 = time.perf_counter()
    for r in rows:
        s = time.perf_counter()
        try:
            resp = requests.post(f"{url}/predict", json=r, timeout=10); resp.raise_for_status()
            lat.append((time.perf_counter() - s) * 1000)
        except Exception:
            errors += 1
    total = time.perf_counter() - t0
    lat_sorted = sorted(lat)
    rep = {"timestamp": now_iso(), "requests": len(rows), "errors": errors,
           "mean_ms": round(statistics.mean(lat), 2) if lat else None,
           "p95_ms": round(lat_sorted[int(0.95 * (len(lat_sorted) - 1))], 2) if lat else None,
           "throughput_rps": round(len(lat) / total, 2)}
    write_json(p(cfg["paths"]["artifacts_dir"], "api_performance_report.json"), rep)
    log.info("%s", rep)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000", int(sys.argv[2]) if len(sys.argv) > 2 else 100)
