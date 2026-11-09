"""Append API predictions to logs/inference/inference_log.jsonl (consumed by stage 12 monitors)."""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(os.environ.get("AIRLINE_INFERENCE_LOG",
                               Path(__file__).resolve().parents[1] / "logs" / "inference" / "inference_log.jsonl"))
_lock = threading.Lock()


def log_prediction(result: dict, latency_ms: float, source: str = "api") -> None:
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "source": source,
           "latency_ms": round(latency_ms, 2), **result}
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock, open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass  # logging must never break serving
