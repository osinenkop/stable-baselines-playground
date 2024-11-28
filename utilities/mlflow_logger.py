import inspect
from datetime import datetime
import mlflow
import numpy as np
import os
import sys
from typing import Dict, Any, Tuple, Union

from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger



class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


def add_ml_logger(model):
    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )
    model.set_logger(loggers)


def mlflow_monotoring(func):
    def inner1(*args, **kwargs):
        experiment_name = os.path.basename(inspect.stack()[1].filename).split(".")[0]
        run_name = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
            
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            func(*args, **kwargs, use_mlflow=True)

    return inner1
