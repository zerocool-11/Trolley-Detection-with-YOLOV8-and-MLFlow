import mlflow
from typing import Any
import pandas as pd

def create_mlflow_experiment(experiment_name: str, artifact_location: str, tags: dict[str, Any]) -> str:
    try:
        x = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except Exception as e:
        print("exception handled:",e)
        print(f"Experiment {experiment_name} already exists.")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    return experiment_id


def get_mlflow_experiment(experiment_name: str = None , experiment_id: str = None) -> str:
    if experiment_name:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    elif experiment_id:
        experiment  = mlflow.get_experiment(experiment_id)
    else:   
        raise ValueError("Either experiment_name or experiment_id should be provided.")
    return experiment
    
def delete_mlflow_experiment(experiment_id: str):
    try:
        mlflow.delete_experiment(experiment_id,)
    except:
        print(f"Experiment {experiment_id} does not exist.")
        
