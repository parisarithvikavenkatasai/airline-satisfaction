"""Print registry contents (local JSON registry, plus MLflow if installed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
from src.registry import summary, mlflow_available

if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
    if mlflow_available():
        import mlflow
        from src.common import p
        mlflow.set_tracking_uri(f"sqlite:///{p('mlflow.db').as_posix()}")
        for rm in mlflow.tracking.MlflowClient().search_registered_models():
            print("MLflow registered model:", rm.name, rm.aliases)
