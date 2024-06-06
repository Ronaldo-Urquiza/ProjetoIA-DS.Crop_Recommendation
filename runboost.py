import mlflow

experiment_name = "exp_teste_modelos_de_boost"
entry_point = "Boosts.py"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name
)