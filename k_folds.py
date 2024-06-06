import numpy as np
from sklearn.model_selection import StratifiedKFold
from conversao_normalizacao import converte_normaliza  # Importação do código de conversão e normalização
def cross_validation(ds_path, folds=10):
    # Carregando e pré-processando os dados
    DS = converte_normaliza(ds_path)  # Conversão e normalização

    # Extrai a última coluna
    y = DS['Crop']  # Extrai a última coluna, que é o label

    # Extrai as características (todas as colunas exceto a última)
    X = DS.iloc[:, :-1]

    # Transforma para Array NumPy
    X = np.array(X)
    y = np.array(y)

    kf = StratifiedKFold(n_splits=folds)

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for train_index, test_index in kf.split(X, y):
        X_train.append(X[train_index])
        X_test.append(X[test_index])

        y_train.append(y[train_index])
        y_test.append(y[test_index])

    return X_train, X_test, y_train, y_test