import numpy as np
import os
from conversao_normalizacao import converte_normaliza # Importação do código de conversão e normalização
from k_folds import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

# Caminho para o dataset
ds_path = os.path.join('..', 'Crop_Recommendation.csv')

folds=10 #Folds reduzidos devido a complexidade e demora de processamento do modelo
#Teste com 10 folds (tempo) -> 02:15 - 02:17 (3 minutos)

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
    # Aqui estamos criando um modelo de classificação de rede neural usando MLPClassifier.
    # O modelo terá duas camadas ocultas, com 20 e 10 neurônios, respectivamente.
    # A função de ativação usada nas camadas ocultas será a tangente hiperbólica ('tanh').
    # Vamos permitir que o treinamento continue por um máximo de 6000 iterações.
    model = MLPClassifier(hidden_layer_sizes=(20, 10), activation='tanh', max_iter=6000)
    model = model.fit(X_train[i], y_train[i])

    result = model.predict(X_test[i])
    acc = metrics.accuracy_score(result, y_test[i]) * 100  # Converte precisão para porcentagem
    acc = round(acc, 2)  # Arredonda a precisão para duas casas decimais
    results.append(acc)

print("Acurácia de cada Multi-Layer-Perceptron ->",results)
show = round(np.mean(results), 2)  # Arredonda a média para duas casas decimais
print("Média das acurácias dos modelos MLP (10 folds): {}%".format(show))