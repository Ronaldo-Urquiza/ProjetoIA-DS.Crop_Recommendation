import numpy as np
import os
from conversao_normalizacao import converte_normaliza # Importação do código de conversão e normalização
from k_folds import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

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

# Inicializa listas para armazenar os resultados de cada fold
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []

for i in range(folds):  # Certifique-se de que "folds" está definido antes deste loop
    model = DecisionTreeClassifier(criterion="entropy")
    model = model.fit(X_train[i], y_train[i])

    result = model.predict(X_test[i])

    # Acurácia
    acc = accuracy_score(y_test[i], result)
    accuracies.append(acc)

    # Relatório de Classificação
    report = classification_report(y_test[i], result, output_dict=True, zero_division=0)
    #print(report)
    precision_values = [report[str(label)]['precision'] for label in np.unique(y) if str(label) in report]
    #print(precision_values)
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

#print("Precisão:",precisions)
#print("Tamanho:",len(precisions))
#print("Recalls:",recalls)
#print("Tamanho:",len(recalls))
#print("F1_scores:",f1_scores)
#print("Tamanho:",len(f1_scores))
#print("Acurácias:",accuracies)
#print("Tamanho:",len(accuracies))
#print("ROC:",roc_aucs)
#print("Tamanho:",len(roc_aucs))

# Média das métricas
mean_accuracy = np.mean(accuracies) * 100
mean_precision = np.mean(precisions) * 100
mean_recall = np.mean(recalls) * 100
mean_f1_score = np.mean(f1_scores) * 100
mean_roc_auc = np.mean(roc_aucs) * 100

print("Média das acurácias das Árvores de Decisão (100 folds): {:.2f}%".format(mean_accuracy))
print("Média das precisões: {:.2f}%".format(mean_precision))
print("Média dos recalls: {:.2f}%".format(mean_recall))
print("Média dos F1-scores: {:.2f}%".format(mean_f1_score))
print("Média das ROC AUCs: {:.2f}%".format(mean_roc_auc))
