"""Stage 12 - Append performance snapshot to history and plot the trend."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common import load_config, p, read_json, write_json, get_logger, now_iso

log = get_logger("12_performance_trend")


def main():
    cfg = load_config()
    rep = read_json(p(cfg["paths"]["reports_dir"], "performance_report.json"))
    hist_path = p(cfg["paths"]["logs_dir"], "performance_history.json")
    hist = read_json(hist_path) if hist_path.exists() else []
    hist.append({"timestamp": now_iso(), "model_version": rep["model_version"],
                 **{k: v["f1"] for k, v in rep["datasets"].items()}})
    write_json(hist_path, hist)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for k in ["reference_test", "evolved", "concept_drift"]:
        ax.plot(range(len(hist)), [h[k] for h in hist], marker="o", label=k)
    ax.set_xlabel("monitoring run"); ax.set_ylabel("F1"); ax.set_title("F1 trend"); ax.legend()
    fig.tight_layout(); fig.savefig(p(cfg["paths"]["artifacts_dir"], "performance_trend.png"), dpi=150); plt.close(fig)
    log.info("History length: %d", len(hist))


if __name__ == "__main__":
    main()
