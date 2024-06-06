import mlflow

experiment_name = "Crop_Recommendation_Experiment"
entry_point = "MlFlow2.py"

mlflow.projects.run(
    uri=".",
    entry_point=entry_point,
    experiment_name=experiment_name
)