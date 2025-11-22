import os
import mlflow
from mlflow.tracking import MlflowClient
from functools import wraps
from rich.console import Console
from src.console import colored_text, input_with_default

console = Console()
TRACK_EXP = True if os.getenv("TRACK_EXPERIMENT") else False

def _short_circuit(f):
    @wraps(f)
    def wrapper(*args, **kw):
        if not TRACK_EXP:
            return None
        return f(*args, **kw)
    return wrapper


#=========================================================================================


@_short_circuit
def connect_server(uri, experiment_name):
    with console.status("Connecting to MLFlow..."):
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
    print(colored_text("Connected to MLFLow server"))


@_short_circuit
def start_run(run_name=None):
    run_name = input_with_default(prompt="Enter run name", default="Test")
    print(colored_text(f"Run name set to: {run_name}"))
    return mlflow.start_run(run_name=run_name)


@_short_circuit
def log_params(params: dict):
    with console.status("Logging parameters..."):
        mlflow.log_params(params=params)


@_short_circuit
def log_artifact(artifact_path: str, run_id: str | None = None):
    """Logs an artifact to an existing run (or to the active run if run_id is None)."""

    with console.status("Logging artifact..."):
        if run_id is None:                       # usual mode: inside an active run
            mlflow.log_artifact(artifact_path)
        else:                                    # post-hoc: use the client to log to a past run
            client = MlflowClient()
            client.log_artifact(run_id, artifact_path)
    print(colored_text("The artifact is registered"))


@_short_circuit
def log_model(model, name: str, signature, run_id: str | None = None):
    """Registers a model in an existing run (or in the active run if run_id is None)."""

    with console.status("Logging model..."):
        if run_id is None:                       # inside an active run
            mlflow.pytorch.log_model(model, name, signature=signature)
        else:                                    # post-hoc
            with mlflow.start_run(run_id=run_id):  # open the run in write-only mode
                mlflow.pytorch.log_model(model, name, signature=signature)
    print(colored_text("The model is registered"))

@_short_circuit
def end_run():
    return mlflow.end_run()