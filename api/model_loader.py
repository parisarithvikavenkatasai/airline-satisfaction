"""Loads the deployed model + preprocessor from deployment/ and scores records."""
import json
import os
import threading
from pathlib import Path
from typing import List

import joblib
import pandas as pd

from api.preprocessing import prepare_features

DEPLOY_DIR = Path(os.environ.get("AIRLINE_DEPLOYMENT_DIR", Path(__file__).resolve().parents[1] / "deployment"))


class ModelNotLoaded(RuntimeError):
    pass


class ModelStore:
    def __init__(self, deploy_dir: Path = DEPLOY_DIR):
        self.deploy_dir = Path(deploy_dir)
        self.model = None
        self.preprocessor = None
        self.metadata: dict = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.preprocessor is not None

    def load(self) -> None:
        with self._lock:
            files = [self.deploy_dir / n for n in ("model.pkl", "preprocessor.pkl", "model_metadata.json")]
            missing = [f.name for f in files if not f.exists()]
            if missing:
                raise ModelNotLoaded(f"Deployment files missing in {self.deploy_dir}: {missing}. "
                                     "Run: python src/mlops_pipeline/11_prepare_deployment.py")
            self.model = joblib.load(files[0])
            self.preprocessor = joblib.load(files[1])
            self.metadata = json.loads(files[2].read_text(encoding="utf-8"))

    @property
    def version(self):
        return self.metadata.get("registry_version")

    def predict(self, records: List[dict]) -> List[dict]:
        if not self.loaded:
            raise ModelNotLoaded("Model not loaded")
        df = pd.DataFrame(records)
        X = prepare_features(df, self.preprocessor, self.metadata["arrival_delay_median"])
        proba = self.model.predict_proba(X)[:, 1]
        thr = self.metadata.get("decision_threshold", 0.5)
        pos, neg = self.metadata["positive_label"], self.metadata["negative_label"]
        return [{"prediction": pos if p >= thr else neg, "probability_satisfied": round(float(p), 6),
                 "model_version": int(self.version)} for p in proba]


store = ModelStore()
