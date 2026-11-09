
"""Model registry: local JSON registry mirrored to MLflow when available.

The local registry (models/registry.json) is the source of truth used by
deployment. MLflow is used as an additional tracking and model registry layer.
"""

from src.common import ROOT, load_config, now_iso, read_json, write_json


REGISTRY_PATH = ROOT / "models" / "registry.json"


def mlflow_available() -> bool:
    """Check whether MLflow is installed."""
    try:
        import mlflow
        return True
    except ImportError:
        return False


def _load() -> dict:
    """Load the local model registry."""
    if REGISTRY_PATH.exists():
        return read_json(REGISTRY_PATH)

    return {
        "model_name": load_config()["mlflow"]["registered_model_name"],
        "versions": [],
        "aliases": {}
    }


def _save(reg: dict) -> None:
    """Save the local model registry."""
    write_json(REGISTRY_PATH, reg)


def register_version(
    model_file: str,
    model_type: str,
    metrics: dict,
    threshold: float,
    mlflow_run_id=None,
    preprocessor_file: str = "models/preprocessor.pkl",
    source: str = "initial_training"
) -> int:
    """Register a new model version in the local registry."""

    reg = _load()

    version = len(reg["versions"]) + 1

    reg["versions"].append({
        "version": version,
        "model_file": model_file,
        "model_type": model_type,
        "metrics": metrics,
        "threshold": threshold,
        "mlflow_run_id": mlflow_run_id,
        "preprocessor_file": preprocessor_file,
        "source": source,
        "created_at": now_iso(),
        "stage": "None"
    })

    reg["aliases"]["candidate"] = version

    _save(reg)

    return version


def get_version(version: int) -> dict:
    """Get a model version from the local registry."""

    for model in _load()["versions"]:
        if model["version"] == version:
            return model

    raise KeyError(f"version {version} not found")


def update_version(version: int, **fields) -> None:
    """Update fields of an existing model version."""

    reg = _load()

    for model in reg["versions"]:
        if model["version"] == version:
            model.update(fields)
            break

    _save(reg)


def get_alias(alias: str):
    """Get the model assigned to an alias."""

    reg = _load()
    version = reg["aliases"].get(alias)

    return get_version(version) if version else None


def set_alias(alias: str, version: int) -> None:
    """Set a local registry alias and mirror it to MLflow."""

    reg = _load()

    reg["aliases"][alias] = version

    for model in reg["versions"]:
        if alias == "production":
            if model["version"] == version:
                model["stage"] = "Production"
            elif model["stage"] == "Production":
                model["stage"] = "Archived"

    _save(reg)

    _mirror_alias_to_mlflow(alias, version)


def _mirror_alias_to_mlflow(alias: str, version: int) -> None:
    """Mirror a local alias to the matching MLflow model version.

    Local registry version numbers and MLflow version numbers are
    independent. Therefore, the MLflow version is identified using
    the MLflow run ID stored in the local registry record.
    """

    if not mlflow_available():
        return

    try:
        import mlflow

        cfg = load_config()
        model_name = cfg["mlflow"]["registered_model_name"]

        mlflow.set_tracking_uri(
            f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
        )

        client = mlflow.tracking.MlflowClient()

        # Find the local registry record.
        local_model = get_version(version)
        run_id = local_model.get("mlflow_run_id")

        if not run_id:
            print(
                f"[registry] MLflow alias mirror skipped: "
                f"no MLflow run ID for local version {version}"
            )
            return

        # Find the MLflow version associated with this run ID.
        mlflow_versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        matching_version = next(
            (
                model_version
                for model_version in mlflow_versions
                if model_version.run_id == run_id
            ),
            None
        )

        if matching_version is None:
            print(
                f"[registry] MLflow alias mirror skipped: "
                f"no MLflow version found for run ID {run_id}"
            )
            return

        # Assign the alias using the actual MLflow version.
        client.set_registered_model_alias(
            model_name,
            alias,
            str(matching_version.version)
        )

        print(
            f"[registry] MLflow alias '{alias}' mapped: "
            f"local version {version} -> "
            f"MLflow version {matching_version.version}"
        )

    except Exception as exc:
        # MLflow mirroring must never break the pipeline.
        print(
            f"[registry] MLflow alias mirror skipped: {exc}"
        )


def summary() -> dict:
    """Return the complete local registry."""
    return _load()
