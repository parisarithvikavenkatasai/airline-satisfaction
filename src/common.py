"""Shared helpers: project root, config loading, logging, JSON IO."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "config.yaml") -> dict:
    cfg_path = ROOT / path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(*parts) -> Path:
    """Path relative to project root."""
    return ROOT.joinpath(*parts)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(h)
        ensure_dir(ROOT / "logs")
        fh = logging.FileHandler(ROOT / "logs" / "pipeline.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
        logger.addHandler(fh)
    return logger


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_run_log(stage: str, status: str, details: dict | None = None) -> None:
    """Append a stage record to logs/pipeline_run_log.json."""
    log_path = ROOT / "logs" / "pipeline_run_log.json"
    runs = read_json(log_path) if log_path.exists() else []
    runs.append({"timestamp": now_iso(), "stage": stage, "status": status, "details": details or {}})
    write_json(log_path, runs)
