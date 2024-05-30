import pandas as pd

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

print()
print("Tamanho do dataset (linhas, colunas) ->", DS.shape)

print("\nFeature Nitrogênio:\n")
print("Tipo dos valores que aparecem nessa feature:", DS.Nitrogen.dtype)
print("Valores NaN (Not a Number) da feature:", DS.Nitrogen.isnull().sum())
print("Valores vazios da feature:", DS.Nitrogen.isna().sum())
print("Média dos valores da feature:", DS.Nitrogen.mean())
print("Valor máximo da feature:", DS.Nitrogen.max())
print("Valor mínimo da feature:", DS.Nitrogen.min())
print("Contagem de cada valor de", DS.Nitrogen.value_counts(),"\n")

print("\nFeature Fósforo:\n")
print("Tipo dos valores que aparecem nessa feature:", DS.Phosphorus.dtype)
print("Valores NaN (Not a Number) da feature:", DS.Phosphorus.isnull().sum())
print("Valores vazios da feature:", DS.Phosphorus.isna().sum())
print("Média dos valores da feature:", DS.Phosphorus.mean())
print("Valor máximo da feature:", DS.Phosphorus.max())
print("Valor mínimo da feature:", DS.Phosphorus.min())
print("Contagem de cada valor de", DS.Phosphorus.value_counts(),"\n")

print("\nFeature Potássio:\n")
print("Tipo dos valores que aparecem nessa feature:", DS.Potassium.dtype)
print("Valores NaN (Not a Number) da feature:", DS.Potassium.isnull().sum())
print("Valores vazios da feature:", DS.Potassium.isna().sum())
print("Média dos valores da feature:", DS.Potassium.mean())
print("Valor máximo da feature:", DS.Potassium.max())
print("Valor mínimo da feature:", DS.Potassium.min())
print("Contagem de cada valor de", DS.Potassium.value_counts(),"\n")
