"""Lab 6 - Train, evaluate, gate and register/promote: stages 05-08."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for s in ["05_train_registry", "06_model_evaluation", "07_quality_gate", "08_automate_lifecycle"]:
    runpy.run_path(str(ROOT / "src" / "mlops_pipeline" / f"{s}.py"), run_name="__main__")
