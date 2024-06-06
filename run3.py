import mlflow

experiment_name = "exp_projeto_ciclo_2_V3"
entry_point = "MlFlow3.py"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name
)