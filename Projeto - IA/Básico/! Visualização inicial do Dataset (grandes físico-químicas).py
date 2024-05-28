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

print("\nFeature Temperatura:\n")
print("Tipo dos valores que aparecem nessa feature:", DS.Temperature.dtype)
print("Valores NaN (Not a Number) da feature:", DS.Temperature.isnull().sum())
print("Valores vazios da feature:", DS.Temperature.isna().sum())
print("Média dos valores da feature:", DS.Temperature.mean())
print("Valor máximo da feature:", DS.Temperature.max())
print("Valor mínimo da feature:", DS.Temperature.min())
print("Contagem de cada valor de", DS.Temperature.value_counts(),"\n")

print("\nFeature Umidade:\n")
print("Tipo dos valores que aparecem nessa feature:", DS.Humidity.dtype)
print("Valores NaN (Not a Number) da feature:", DS.Humidity.isnull().sum())
print("Valores vazios da feature:", DS.Humidity.isna().sum())
print("Média dos valores da feature:", DS.Humidity.mean())
print("Valor máximo da feature:", DS.Humidity.max())
print("Valor mínimo da feature:", DS.Humidity.min())
print("Contagem de cada valor de", DS.Humidity.value_counts(),"\n")

print("\nFeature pH:\n")
print("Tipo dos valores que aparecem nessa feature:", DS.pH_Value.dtype)
print("Valores NaN (Not a Number) da feature:", DS.pH_Value.isnull().sum())
print("Valores vazios da feature:", DS.pH_Value.isna().sum())
print("Média dos valores da feature:", DS.pH_Value.mean())
print("Valor máximo da feature:", DS.pH_Value.max())
print("Valor mínimo da feature:", DS.pH_Value.min())
print("Contagem de cada valor de", DS.pH_Value.value_counts(),"\n")

print("\nFeature Chuva em mm:\n")
print("Tipo dos valores que aparecem nessa feature:", DS.Rainfall.dtype)
print("Valores NaN (Not a Number) da feature:", DS.Rainfall.isnull().sum())
print("Valores vazios da feature:", DS.Rainfall.isna().sum())
print("Média dos valores da feature:", DS.Rainfall.mean())
print("Valor máximo da feature:", DS.Rainfall.max())
print("Valor mínimo da feature:", DS.Rainfall.min())
print("Contagem de cada valor de", DS.Rainfall.value_counts(),"\n")
