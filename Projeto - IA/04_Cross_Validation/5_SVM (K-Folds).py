import numpy as np
import os
from conversao_normalizacao import converte_normaliza # Importação do código de conversão e normalização
from k_folds import cross_validation
from sklearn.svm import SVC
from sklearn import metrics

# Caminho para o dataset
ds_path = os.path.join('..', 'Crop_Recommendation.csv')
folds=100

# Carregando e pré-processa os dados
DS = converte_normaliza(ds_path) #conversão e normalização
#print(DS) #Teste para saber se a importação do código de conversão e normalização deu certo

# Determina o número de colunas
columns = len(DS.columns)
#print(columns)

# Extrai a última coluna
y = DS['Crop'] # extrai a última coluna, que é o label
#print(y)

# Extrai as características (todas as colunas exceto a última)
X = DS.iloc[:, :-1]
#print(X)

X_train, X_test, y_train, y_test = cross_validation(ds_path,folds)

results = []

for i in range(folds):  # Certifique-se de que "folds" está definido antes deste loop
    model = model = SVC(kernel='rbf')
    model = model.fit(X_train[i], y_train[i])

    result = model.predict(X_test[i])
    acc = metrics.accuracy_score(result, y_test[i]) * 100  # Converte precisão para porcentagem
    acc = round(acc, 2)  # Arredonda a precisão para duas casas decimais
    results.append(acc)

print("Acurácia de cada Support Vector Classifier ->",results)
show = round(np.mean(results), 2)  # Arredonda a média para duas casas decimais
print("Média dos modelos SVC (100 folds): {}%".format(show))