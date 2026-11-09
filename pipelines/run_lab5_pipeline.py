"""Lab 5 - Data pipeline: stages 01-04 (load, validate, preprocess, validate outputs)."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for s in ["01_load_data", "02_validate_data", "03_preprocessing", "04_validate_outputs"]:
    runpy.run_path(str(ROOT / "src" / "mlops_pipeline" / f"{s}.py"), run_name="__main__")
