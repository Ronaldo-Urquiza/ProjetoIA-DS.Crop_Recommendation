import mlflow

experiment_name = "exp_projeto_ciclo_2_V2"
entry_point = "MlFlow2.py"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name
)