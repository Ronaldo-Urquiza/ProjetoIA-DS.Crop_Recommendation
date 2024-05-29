import os
from conversao_normalizacao import converte_normaliza # Importação do código de conversão e normalização
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

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
#print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, stratify=y) #Separa variáveis para treinamento (X_train,y_train) e variáveis para teste(X_test,y_test) através do split puxando o X e y de cima

                                                    #tamanho de teste = 20% então treinamento = 80% (podes coloar train_size=0.8 mas já está implícito)

                                                    #random_state, que controla a aleatoriedade na divisão dos dados, none = totalmente aleatório

                                                    #80% das amostras pertencem à y_train e 20% pertencem à y_test, ao usar stratify=y, a função train_test_split tentará manter essa proporção de 80-20 entre as classes y_train, y_test.

                                                    #X (colunas representando os atributos), y (coluna(s) representando o target/alvo)

# Aqui estamos criando um modelo de classificação usando o algoritmo RandomForestClassifier.
# RandomForestClassifier é um modelo de aprendizado de máquina baseado em árvores de decisão, que constrói várias árvores de decisão durante o treinamento e combina suas previsões para obter um resultado final mais robusto.
model = RandomForestClassifier()

# Treina o modelo RF com os dados de treinamento
model.fit(X_train, y_train)

# Predição e Resultados
result = model.predict(X_test) #irá retornar uma DataFrame com os dados que o seu modelo tentou predizer (20% do dataframe)

# Esse bloquinho é so para mostrar o resultado do quanto sua árvore foi eficiente -=-=-=-=-=-
acc = metrics.accuracy_score(result, y_test) #obtém o resultado da comparação entre sua árvore e a classe real do dataframe
show = round(acc * 100)  #transforma no valor visual de porcentagem
print("\nRandom Forest:\n")
print("Acurácia: {}%".format(show))
print("O que o modelo tentou predizer ->  ",list(result))
print("O que o modelo era para predizer ->",list(y_test))
# fim do bloquinho -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

