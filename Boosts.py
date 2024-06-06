import numpy as np
from sklearn.metrics import accuracy_score
import mlflow.sklearn
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Importar funções dos arquivos auxiliares
from conversao_normalizacao import converte_normaliza
from k_folds import cross_validation

# Caminho para o dataset
ds_path = 'Crop_Recommendation.csv'
folds = 10

# Carregar e pré-processar os dados
X_train_folds, X_test_folds, y_train_folds, y_test_folds = cross_validation(ds_path, folds)

# Lista dos modelos com parâmetros diferentes
models = {
    "XGBoost": XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100),
    "LightGBM": LGBMClassifier(max_depth=5, learning_rate=0.05, n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(max_depth=4, learning_rate=0.1, n_estimators=100)
}

# Criar o experimento
mlflow.create_experiment("exp_teste_modelos_de_boost")

# Definir o experimento atual
mlflow.set_experiment("exp_teste_modelos_de_boost")

# Loop sobre os modelos
for model_name, model in models.items():
    accuracies = []
    for i in range(folds):
        X_train_fold = np.array(X_train_folds[i])
        X_test_fold = np.array(X_test_folds[i])
        y_train_fold = np.array(y_train_folds[i])
        y_test_fold = np.array(y_test_folds[i])

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        accuracy = accuracy_score(y_test_fold, y_pred)
        accuracies.append(accuracy)

    # Calcular a média das acurácias dos folds
    mean_accuracy = np.mean(accuracies)

    # Logar o resultado final no MLflow
    with mlflow.start_run(run_name=f"{model_name}_Final"):
        mlflow.log_metric("Mean Accuracy", mean_accuracy)
        mlflow.sklearn.log_model(model, f"{model_name}_Final_Model")

print("Experimentos concluídos!")