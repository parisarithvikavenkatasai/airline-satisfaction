"""End-to-end orchestrator: python src/mlops_pipeline/airline_prediction_pipeline.py [--with-monitoring]"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import runpy, time
from src.common import get_logger, write_json, p, now_iso

log = get_logger("pipeline")
CORE = ["01_load_data", "02_validate_data", "03_preprocessing", "04_validate_outputs", "05_train_registry",
        "06_model_evaluation", "07_quality_gate", "08_automate_lifecycle", "09_inference",
        "10_failure_injection", "11_prepare_deployment"]
MONITOR = ["12_create_evolved_dataset", "12_create_concept_drift_dataset", "12_create_updated_dataset",
           "12_data_drift", "12_prediction_drift", "12_concept_drift", "12_performance_monitoring",
           "12_performance_trend", "12_drift_analysis", "12_monitor_trigger"]


def run(stages):
    here = Path(__file__).parent; results = []
    for s in stages:
        t0 = time.time(); log.info("=== %s ===", s)
        try:
            runpy.run_path(str(here / f"{s}.py"), run_name="__main__"); status = "SUCCESS"
        except SystemExit as e:
            status = "SUCCESS" if e.code in (0, None) else "FAILED"
        except Exception:  # noqa: BLE001
            log.exception("Stage %s raised", s); status = "FAILED"
        results.append({"stage": s, "status": status, "seconds": round(time.time() - t0, 1)})
        if status == "FAILED":
            log.error("Stage %s failed - aborting", s); break
    write_json(p("logs", "airline_prediction_pipeline_run_log.json"), {"timestamp": now_iso(), "stages": results})
    return all(r["status"] == "SUCCESS" for r in results)


if __name__ == "__main__":
    stages = CORE + (MONITOR if "--with-monitoring" in sys.argv else [])
    sys.exit(0 if run(stages) else 1)
