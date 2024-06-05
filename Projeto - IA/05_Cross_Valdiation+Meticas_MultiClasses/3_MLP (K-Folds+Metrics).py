import numpy as np
import os
from conversao_normalizacao import converte_normaliza  # Importação do código de conversão e normalização
from k_folds import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

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

# Inicializa listas para armazenar os resultados de cada fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []

for i in range(folds):  # Certifique-se de que "folds" está definido antes deste loop
    # Aqui estamos criando um modelo de classificação de rede neural usando MLPClassifier.
    # O modelo terá duas camadas ocultas, com 20 e 10 neurônios, respectivamente.
    # A função de ativação usada nas camadas ocultas será a tangente hiperbólica ('tanh').
    # Vamos permitir que o treinamento continue por um máximo de 6000 iterações.
    model = MLPClassifier(hidden_layer_sizes=(20, 10), activation='tanh', max_iter=6000)
    model = model.fit(X_train[i], y_train[i])

    result = model.predict(X_test[i])

    # Acurácia
    acc = accuracy_score(y_test[i], result)
    accuracies.append(acc)

    # Relatório de Classificação
    report = classification_report(y_test[i], result, output_dict=True, zero_division=0)
    #print(report)
    precision_values = [report[str(label)]['precision'] for label in np.unique(y) if str(label) in report]
    precisions.append(np.mean(np.array(precision_values)))
    recall_values = [report[str(label)]['recall'] for label in np.unique(y) if str(label) in report]
    recalls.append(np.mean(np.array(recall_values)))
    f1_score_values = [report[str(label)]['f1-score'] for label in np.unique(y) if str(label) in report]
    f1_scores.append(np.mean(np.array(f1_score_values)))

    # ROC AUC
    y_test_bin = label_binarize(y_test[i], classes=np.unique(y))
    result_bin = label_binarize(result, classes=np.unique(y))
    if y_test_bin.shape == result_bin.shape:  # Verifica se a binarização está correta
        roc_auc = roc_auc_score(y_test_bin, result_bin, average="macro")
        roc_aucs.append(roc_auc)

# Média das métricas
mean_accuracy = np.mean(accuracies) * 100
mean_precision = np.mean(precisions) * 100
mean_recall = np.mean(recalls) * 100
mean_f1_score = np.mean(f1_scores) * 100
mean_roc_auc = np.mean(roc_aucs) * 100

print("Média das acurácias dos modelos MLP (10 folds): {:.2f}%".format(mean_accuracy))
print("Média das precisões: {:.2f}%".format(mean_precision))
print("Média dos recalls: {:.2f}%".format(mean_recall))
print("Média dos F1-scores: {:.2f}%".format(mean_f1_score))
print("Média das ROC AUCs: {:.2f}%".format(mean_roc_auc))