import numpy as np
import os
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

# Funções de conversão, normalização e cross_validation
from conversao_normalizacao import converte_normaliza
from k_folds import cross_validation

# Caminho para o dataset
ds_path = os.path.join('..', 'Crop_Recommendation.csv')
folds = 2

# Carregar e pré-processar os dados
DS = converte_normaliza(ds_path)  # conversão e normalização
y = DS['Crop']  # extrai a última coluna, que é o label
X = DS.iloc[:, :-1]  # extrai as características (todas as colunas exceto a última)

X_train, X_test, y_train, y_test = cross_validation(ds_path, folds)

# Inicializa listas para armazenar os resultados de cada fold
resultKNU = []
resultKNE = []
resultLCA = []
resultOLA = []
resultMCB = []

# Inicializa listas para armazenar as métricas de cada classificador
accuracies_knn = []
precisions_knn = []
recalls_knn = []
f1_scores_knn = []
roc_aucs_knn = []

accuracies_svc = []
precisions_svc = []
recalls_svc = []
f1_scores_svc = []
roc_aucs_svc = []

accuracies_rf = []
precisions_rf = []
recalls_rf = []
f1_scores_rf = []
roc_aucs_rf = []

for i in range(folds):
    # Divisão de treino e validação
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_train[i], y_train[i], test_size=0.5, random_state=None)

    # Criação e treinamento dos modelos KNN e RandomForest com Bagging e do SVM com AdaBoost
    knn = BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=5), n_estimators=50)
    svc = AdaBoostClassifier(estimator=SVC(kernel='rbf', probability=True), n_estimators=50, algorithm='SAMME')
    rf = BaggingClassifier(estimator=RandomForestClassifier(n_estimators=100), n_estimators=50)

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

    # Avaliação das métricas para cada classificador individualmente

    # KNN
    result_knn = knn.predict(X_validacao)

    acc_knn = accuracy_score(y_validacao, result_knn)
    accuracies_knn.append(acc_knn)
    report_knn = classification_report(y_validacao, result_knn, output_dict=True, zero_division=0)

    precision_values_knn = [report_knn[str(label)]['precision'] for label in np.unique(y) if str(label) in report_knn]
    precisions_knn.append(np.mean(np.array(precision_values_knn)))
    recall_values_knn = [report_knn[str(label)]['recall'] for label in np.unique(y) if str(label) in report_knn]
    recalls_knn.append(np.mean(np.array(recall_values_knn)))
    f1_score_values_knn = [report_knn[str(label)]['f1-score'] for label in np.unique(y) if str(label) in report_knn]
    f1_scores_knn.append(np.mean(np.array(f1_score_values_knn)))
    y_validacao_bin_knn = label_binarize(y_validacao, classes=np.unique(y))
    result_bin_knn = label_binarize(result_knn, classes=np.unique(y))
    if y_validacao_bin_knn.shape == result_bin_knn.shape:
        roc_auc_knn = roc_auc_score(y_validacao_bin_knn, result_bin_knn, average="macro")
        roc_aucs_knn.append(roc_auc_knn)

    # SVC
    result_svc = svc.predict(X_validacao)

    acc_svc = accuracy_score(y_validacao, result_svc)
    accuracies_svc.append(acc_svc)
    report_svc = classification_report(y_validacao, result_svc, output_dict=True, zero_division=0)

    precision_values_svc = [report_svc[str(label)]['precision'] for label in np.unique(y) if str(label) in report_svc]
    precisions_svc.append(np.mean(np.array(precision_values_svc)))
    recall_values_svc = [report_svc[str(label)]['recall'] for label in np.unique(y) if str(label) in report_svc]
    recalls_svc.append(np.mean(np.array(recall_values_svc)))
    f1_score_values_svc = [report_svc[str(label)]['f1-score'] for label in np.unique(y) if str(label) in report_svc]
    f1_scores_svc.append(np.mean(np.array(f1_score_values_svc)))
    y_validacao_bin_svc = label_binarize(y_validacao, classes=np.unique(y))
    result_bin_svc = label_binarize(result_svc, classes=np.unique(y))
    if y_validacao_bin_svc.shape == result_bin_svc.shape:
        roc_auc_svc = roc_auc_score(y_validacao_bin_svc, result_bin_svc, average="macro")
        roc_aucs_svc.append(roc_auc_svc)

    # RandomForest
    result_rf = rf.predict(X_validacao)

    acc_rf = accuracy_score(y_validacao, result_rf)
    accuracies_rf.append(acc_rf)
    report_rf = classification_report(y_validacao, result_rf, output_dict=True, zero_division=0)

    precision_values_rf = [report_rf[str(label)]['precision'] for label in np.unique(y) if str(label) in report_rf]
    precisions_rf.append(np.mean(np.array(precision_values_rf)))
    recall_values_rf = [report_rf[str(label)]['recall'] for label in np.unique(y) if str(label) in report_rf]
    recalls_rf.append(np.mean(np.array(recall_values_rf)))
    f1_score_values_rf = [report_rf[str(label)]['f1-score'] for label in np.unique(y) if str(label) in report_rf]
    f1_scores_rf.append(np.mean(np.array(f1_score_values_rf)))
    y_validacao_bin_rf = label_binarize(y_validacao, classes=np.unique(y))
    result_bin_rf = label_binarize(result_rf, classes=np.unique(y))
    if y_validacao_bin_rf.shape == result_bin_rf.shape:
        roc_auc_rf = roc_auc_score(y_validacao_bin_rf, result_bin_rf, average="macro")
        roc_aucs_rf.append(roc_auc_rf)

# Calcular e imprimir os resultados médios das acurácias para todos os modelos
print("\n","Resultado das seleções dinâmicas:","\n")
print("KNORA-U: {:.2f}%".format(np.mean(resultKNU) * 100))
print("KNORA-E: {:.2f}%".format(np.mean(resultKNE) * 100))
print("LCA: {:.2f}%".format(np.mean(resultLCA) * 100))
print("OLA: {:.2f}%".format(np.mean(resultOLA) * 100))
print("MCB: {:.2f}%".format(np.mean(resultMCB) * 100))

# Média das métricas para KNN
mean_accuracy_knn = np.mean(accuracies_knn) * 100
mean_precision_knn = np.mean(precisions_knn) * 100
mean_recall_knn = np.mean(recalls_knn) * 100
mean_f1_score_knn = np.mean(f1_scores_knn) * 100
mean_roc_auc_knn = np.mean(roc_aucs_knn) * 100

print("\n","Métricas do K-Nearst-Neighbour:","\n")
print("KNN - Média das acurácias: {:.2f}%".format(mean_accuracy_knn))
print("KNN - Média das precisões: {:.2f}%".format(mean_precision_knn))
print("KNN - Média dos recalls: {:.2f}%".format(mean_recall_knn))
print("KNN - Média dos F1-scores: {:.2f}%".format(mean_f1_score_knn))
print("KNN - Média das ROC AUCs: {:.2f}%".format(mean_roc_auc_knn))

# Média das métricas para SVC
mean_accuracy_svc = np.mean(accuracies_svc) * 100
mean_precision_svc = np.mean(precisions_svc) * 100
mean_recall_svc = np.mean(recalls_svc) * 100
mean_f1_score_svc = np.mean(f1_scores_svc) * 100
mean_roc_auc_svc = np.mean(roc_aucs_svc) * 100

print("\n","Métricas do Support Vector Classifier:","\n")
print("SVC - Média das acurácias: {:.2f}%".format(mean_accuracy_svc))
print("SVC - Média das precisões: {:.2f}%".format(mean_precision_svc))
print("SVC - Média dos recalls: {:.2f}%".format(mean_recall_svc))
print("SVC - Média dos F1-scores: {:.2f}%".format(mean_f1_score_svc))
print("SVC - Média das ROC AUCs: {:.2f}%".format(mean_roc_auc_svc))

# Média das métricas para RandomForest
mean_accuracy_rf = np.mean(accuracies_rf) * 100
mean_precision_rf = np.mean(precisions_rf) * 100
mean_recall_rf = np.mean(recalls_rf) * 100
mean_f1_score_rf = np.mean(f1_scores_rf) * 100
mean_roc_auc_rf = np.mean(roc_aucs_rf) * 100

print("\n","Métricas do Random Forest:","\n")
print("RandomForest - Média das acurácias: {:.2f}%".format(mean_accuracy_rf))
print("RandomForest - Média das precisões: {:.2f}%".format(mean_precision_rf))
print("RandomForest - Média dos recalls: {:.2f}%".format(mean_recall_rf))
print("RandomForest - Média dos F1-scores: {:.2f}%".format(mean_f1_score_rf))
print("RandomForest - Média das ROC AUCs: {:.2f}%".format(mean_roc_auc_rf))