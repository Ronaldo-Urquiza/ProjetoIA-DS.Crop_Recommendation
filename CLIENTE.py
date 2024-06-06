import mlflow
from mlflow.tracking import MlflowClient
from mlflow_functions import get_top_models, register_models

# Defina a URI de rastreamento, se necessário
mlflow.set_tracking_uri("file:///C:/Users/ronal/PycharmProjects/MachineLearning%20-%20MlFlow/mlruns")

# Crie uma instância do MlflowClient
client = MlflowClient()

# Obtenha todos os experimentos
experiments = client.list_experiments()

# Lista para armazenar as acurácias de todos os modelos
all_accuracies = []

# Iterar sobre os experimentos
for experiment in experiments:
    experiment_name = experiment.name
    # Verifique se o nome do experimento corresponde ao padrão desejado
    if "exp_projeto_ciclo" in experiment_name:
        # Obtenha os melhores modelos e suas métricas para este experimento
        top_models = get_top_models(experiment_name, num_models=3)
        # Adicione as acurácias à lista geral
        all_accuracies.extend(top_models["metrics.accuracy"].tolist())

# Ordenar os modelos com base nas acurácias
top_runs = sorted(all_accuracies, reverse=True)[:3]

# Registre os melhores modelos
register_models(top_runs)