import mlflow

experiment_name = "exp_projeto_ciclo_2_V1"
entry_point = "MlFlow1.py"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name
)