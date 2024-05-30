import os
import seaborn as sns
import matplotlib.pyplot as plt
from conversao_normalizacao import converte_normaliza # Importação do código de conversão e normalização
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Caminho para o dataset
ds_path = os.path.join('..', 'Crop_Recommendation.csv')

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
print(X,"\n")

# Calcula a matriz de correlação
matriz_de_correlacao = X.corr()

# Configura a aparência da matriz
plt.figure(figsize=(12, 10))                  #cores frias     #duas casas decimais
sns.heatmap(matriz_de_correlacao, annot=True, cmap='coolwarm', fmt='.2f')
                                  #valores numéricos de correlação serão exibidos diretamente nas células do gráfico

# Mostra o heatmap
plt.title('Heatmap de Correlação das Features')
plt.show()

# O SelectKBest pode ser usado com várias funções, aqui usamos a f_classif (dados numéricos e variável alvo categórica)
f_classifteste = SelectKBest(score_func=f_classif,k=7)

modelofeat = f_classifteste.fit(X,y)

# Obtendo os nomes das features
feature_names = modelofeat.get_feature_names_out()
# Obtendo os scores das features
feature_scores = modelofeat.scores_

# Iterando sobre as features e imprimindo os nomes e scores em linhas alternadas para melhor visualização
for name, score in zip(feature_names, feature_scores):
    print(f"Feature: {name}")
    print(f"Score: {score}\n")