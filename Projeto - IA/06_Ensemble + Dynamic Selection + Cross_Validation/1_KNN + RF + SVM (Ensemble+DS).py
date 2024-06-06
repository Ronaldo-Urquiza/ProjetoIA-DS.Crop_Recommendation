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

resultKNU = []
resultKNE = []
resultLCA = []
resultOLA = []
resultMCB = []

for i in range(folds):
    # Divisão de treino e validação
    X_treino, X_validacao, y_treino, y_validacao = train_test_split(X_train[i], y_train[i], test_size=0.5,random_state=None)

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

# Calcular e imprimir os resultados médios das acurácias para todos os modelos
print("KNORA-U: {:.2f}%".format(np.mean(resultKNU) * 100))
print("KNORA-E: {:.2f}%".format(np.mean(resultKNE) * 100))
print("LCA: {:.2f}%".format(np.mean(resultLCA) * 100))
print("OLA: {:.2f}%".format(np.mean(resultOLA) * 100))
print("MCB: {:.2f}%".format(np.mean(resultMCB) * 100))
