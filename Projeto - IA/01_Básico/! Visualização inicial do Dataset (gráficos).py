import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Obtendo caminho onde o dataset está
ds_path = '../Crop_Recommendation.csv'

# Lendo o arquivo CSV da mesma pasta onde o script está localizado
DS = pd.read_csv(ds_path)

#Ajusta a largura máxima de exibição para garantir que as colunas sejam exibidas sem quebra de linha no terminal
pd.set_option('display.width', 200)

# Definir a opção para exibir todas as colunas
pd.set_option('display.max_columns', None)

# Mostrando as 5 primeiras linhas do DataFrame para verificar a leitura
print(DS.head())

# Lista das colunas a serem plotadas
colunas = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall']

# Loop sobre cada coluna para gerar os histogramas
for coluna in colunas:
    # Plotando o histograma
    sns.histplot(x=coluna, data=DS, kde=True)
    # Adicionando título ao histograma
    plt.title(f'Histograma da coluna {coluna}')
    # Exibindo o histograma
    plt.show()

# Loop sobre cada coluna para gerar os boxplots
for coluna in colunas:
    # Plotando o boxplot
    sns.boxplot(x=coluna, data=DS)
    # Adicionando título ao boxplot
    plt.title(f'Boxplot da coluna {coluna}')
    # Exibindo o boxplot
    plt.show()

# Define a paleta de core através do Seaborn
colors = sns.color_palette('pastel')[0:5]

#Cria gráfico de barras
plt.figure(figsize=(12, 6), facecolor='#FFFFFF')
plt.barh(DS['Crop'].unique(), DS['Crop'].value_counts(), color=colors)
plt.xlabel('Contagem')
plt.ylabel('Planta/Grão')
plt.show()

# Cria gráfico de setores/pizza
plt.pie(DS['Crop'].value_counts(), labels=DS['Crop'].unique(), colors=colors, autopct='%.0f%%')
plt.show()