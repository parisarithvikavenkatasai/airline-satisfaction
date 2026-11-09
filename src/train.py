"""Model construction and training helpers."""
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def get_models(model_cfg: dict) -> dict:
    models = {
        "logistic_regression": LogisticRegression(**model_cfg["logistic_regression"]),
        "random_forest": RandomForestClassifier(**model_cfg["random_forest"]),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(**model_cfg["xgboost"])
    except ImportError:  # xgboost optional: fall back to sklearn boosting
        models["hist_gradient_boosting"] = HistGradientBoostingClassifier(**model_cfg["gradient_boosting"])
    return models


def train_all(models: dict, X_train, y_train) -> dict:
    for name, m in models.items():
        m.fit(X_train, y_train)
    return models


def save_pickle(obj, path):
    joblib.dump(obj, path)


def load_pickle(path):
    return joblib.load(path)
