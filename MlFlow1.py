import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from deslib.dcs import OLA, LCA, MCB
from deslib.des import KNORAE, KNORAU
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import mlflow
import mlflow.sklearn

# Funções de conversão, normalização e cross_validation
from conversao_normalizacao import converte_normaliza
from k_folds import cross_validation

# Caminho para o dataset
ds_path = 'Crop_Recommendation.csv'
folds = 10

# Carregar e pré-processar os dados
DS = converte_normaliza(ds_path)  # conversão e normalização
y = DS['Crop']  # extrai a última coluna, que é o label
X = DS.iloc[:, :-1]  # extrai as características (todas as colunas exceto a última)

X_train, X_test, y_train, y_test = cross_validation(ds_path, folds)

# Inicializa listas para armazenar os resultados de cada fold para seleções dinâmicas
resultKNU = []
resultKNE = []
resultLCA = []
resultOLA = []
resultMCB = []

# Inicializa dicionários para armazenar as métricas de cada classificador
metrics_knn = {"accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": []}
metrics_svc = {"accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": []}
metrics_rf = {"accuracy": [], "precision": [], "recall": [], "f1_score": [], "roc_auc": []}

# Criar o experimento
mlflow.create_experiment("exp_projeto_ciclo_2_V1")

# Definir o experimento atual
mlflow.set_experiment("exp_projeto_ciclo_2_V1")

for i in range(folds):
    # Divisão de treino e validação
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_train[i], y_train[i], test_size=0.5, random_state=None)

    with mlflow.start_run(run_name=f"Fold_{i + 1}"):
        # Criação e treinamento dos modelos KNN e RandomForest com Bagging e do SVM com AdaBoost
        knn = BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=50)
        svc = AdaBoostClassifier(estimator=SVC(kernel='linear', probability=True, C=1, gamma='auto'), n_estimators=50, learning_rate=1, algorithm='SAMME')
        rf = BaggingClassifier(estimator=RandomForestClassifier(n_estimators=10), n_estimators=50)

        # Treinamento dos classificadores
        knn.fit(X_treino, y_treino)
        svc.fit(X_treino, y_treino)
        rf.fit(X_treino, y_treino)

        # Avaliação dos modelos de seleção dinâmica
        # KNORA-U
        knu = KNORAU(pool_classifiers=[knn, svc, rf], k=5)
        knu.fit(X_treino, y_treino)
        resultKNU.append(knu.score(X_validacao, y_validacao))

        # KNORA-E
        kne = KNORAE(pool_classifiers=[knn, svc, rf], k=5)
        kne.fit(X_treino, y_treino)
        resultKNE.append(kne.score(X_validacao, y_validacao))

        # LCA
        lca = LCA(pool_classifiers=[knn, svc, rf], k=5)
        lca.fit(X_treino, y_treino)
        resultLCA.append(lca.score(X_validacao, y_validacao))

        # OLA
        ola = OLA(pool_classifiers=[knn, svc, rf], k=5)
        ola.fit(X_treino, y_treino)
        resultOLA.append(ola.score(X_validacao, y_validacao))

        # MCB
        mcb = MCB(pool_classifiers=[knn, svc, rf], k=5)
        mcb.fit(X_treino, y_treino)
        resultMCB.append(mcb.score(X_validacao, y_validacao))

        # Função para avaliar e logar as métricas
        def evaluate_and_log(model, X_validacao, y_validacao, metrics_dict, model_name):
            result = model.predict(X_validacao)
            acc = accuracy_score(y_validacao, result)
            metrics_dict["accuracy"].append(acc)

            report = classification_report(y_validacao, result, output_dict=True, zero_division=0)

            precision_values = [report[str(label)]['precision'] for label in np.unique(y) if str(label) in report]
            recall_values = [report[str(label)]['recall'] for label in np.unique(y) if str(label) in report]
            f1_score_values = [report[str(label)]['f1-score'] for label in np.unique(y) if str(label) in report]
            y_validacao_bin = label_binarize(y_validacao, classes=np.unique(y))
            result_bin = label_binarize(result, classes=np.unique(y))

            metrics_dict["precision"].append(np.mean(precision_values))
            metrics_dict["recall"].append(np.mean(recall_values))
            metrics_dict["f1_score"].append(np.mean(f1_score_values))

            if y_validacao_bin.shape == result_bin.shape:
                roc_auc = roc_auc_score(y_validacao_bin, result_bin, average="macro")
                metrics_dict["roc_auc"].append(roc_auc)
                mlflow.log_metric(f"{model_name} ROC AUC", roc_auc)

            mlflow.log_metric(f"{model_name} Accuracy", acc)
            mlflow.log_metric(f"{model_name} Precision", np.mean(precision_values))
            mlflow.log_metric(f"{model_name} Recall", np.mean(recall_values))
            mlflow.log_metric(f"{model_name} F1 Score", np.mean(f1_score_values))

            mlflow.log_params({f"{model_name} Parameters": model.get_params()})

        # Avaliar e logar métricas para KNN, SVC e RF
        evaluate_and_log(knn, X_validacao, y_validacao, metrics_knn, "KNN")
        evaluate_and_log(svc, X_validacao, y_validacao, metrics_svc, "SVC")
        evaluate_and_log(rf, X_validacao, y_validacao, metrics_rf, "RF")

        # Log models
        mlflow.sklearn.log_model(knn, "knn_model")
        mlflow.sklearn.log_model(svc, "svc_model")
        mlflow.sklearn.log_model(rf, "rf_model")

# Printar resultados médios das métricas
def print_metrics(metrics_dict, model_name):
    print(f"\nMétricas para {model_name}:")
    print(f"Acurácia: {np.mean(metrics_dict['accuracy']) * 100:.2f}%")
    print(f"Precisão: {np.mean(metrics_dict['precision']) * 100:.2f}%")
    print(f"Recall: {np.mean(metrics_dict['recall']) * 100:.2f}%")
    print(f"F1 Score: {np.mean(metrics_dict['f1_score']) * 100:.2f}%")
    if metrics_dict["roc_auc"]:
        print(f"ROC AUC: {np.mean(metrics_dict['roc_auc']) * 100:.2f}%")

print_metrics(metrics_knn, "KNN")
print_metrics(metrics_svc, "SVC")
print_metrics(metrics_rf, "RF")

with mlflow.start_run(run_name="Resultados gerais das métricas dos modelos de Machine Learning"):
    mlflow.log_metric("KNN Mean Accuracy", np.mean(metrics_knn["accuracy"]) * 100)
    mlflow.log_metric("KNN Mean Precision", np.mean(metrics_knn["precision"]) * 100)
    mlflow.log_metric("KNN Mean Recall", np.mean(metrics_knn["recall"]) * 100)
    mlflow.log_metric("KNN Mean F1 Score", np.mean(metrics_knn["f1_score"]) * 100)
    if metrics_knn["roc_auc"]:
        mlflow.log_metric("KNN Mean ROC AUC", np.mean(metrics_knn["roc_auc"]) * 100)

    mlflow.log_metric("SVC Mean Accuracy", np.mean(metrics_svc["accuracy"]) * 100)
    mlflow.log_metric("SVC Mean Precision", np.mean(metrics_svc["precision"]) * 100)
    mlflow.log_metric("SVC Mean Recall", np.mean(metrics_svc["recall"]) * 100)
    mlflow.log_metric("SVC Mean F1 Score", np.mean(metrics_svc["f1_score"]) * 100)
    if metrics_svc["roc_auc"]:
        mlflow.log_metric("SVC Mean ROC AUC", np.mean(metrics_svc["roc_auc"]) * 100)

    mlflow.log_metric("RF Mean Accuracy", np.mean(metrics_rf["accuracy"]) * 100)
    mlflow.log_metric("RF Mean Precision", np.mean(metrics_rf["precision"]) * 100)
    mlflow.log_metric("RF Mean Recall", np.mean(metrics_rf["recall"]) * 100)
    mlflow.log_metric("RF Mean F1 Score", np.mean(metrics_rf["f1_score"]) * 100)
    if metrics_rf["roc_auc"]:
        mlflow.log_metric("RF Mean ROC AUC", np.mean(metrics_rf["roc_auc"]) * 100)

# Print and log dynamic selection results
print("\n","Resultado das seleções dinâmicas:", "\n")
print("KNORA-U: {:.2f}%".format(np.mean(resultKNU) * 100))
print("KNORA-E: {:.2f}%".format(np.mean(resultKNE) * 100))
print("LCA: {:.2f}%".format(np.mean(resultLCA) * 100))
print("OLA: {:.2f}%".format(np.mean(resultOLA) * 100))
print("MCB: {:.2f}%".format(np.mean(resultMCB) * 100))

with mlflow.start_run(run_name="Resultado dos algoritimos de seleção dinâmica"):
    mlflow.log_metric("KNORA-U", np.mean(resultKNU) * 100)
    mlflow.log_metric("KNORA-E", np.mean(resultKNE) * 100)
    mlflow.log_metric("LCA", np.mean(resultLCA) * 100)
    mlflow.log_metric("OLA", np.mean(resultOLA) * 100)
    mlflow.log_metric("MCB", np.mean(resultMCB) * 100)