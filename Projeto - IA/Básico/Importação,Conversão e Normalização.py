import pandas as pd

#Obtendo caminho onde o dataset está
ds_path = '../Crop_Recommendation.csv'

# Lendo o arquivo CSV
DS = pd.read_csv(ds_path)

# Mostrando as primeiras linhas do DataFrame para verificar a leitura
print(DS.head())

from sklearn import preprocessing  # Importa o módulo de pré-processamento do scikit-learn

label_encoder = preprocessing.LabelEncoder()  # Inicializa o Label Encoder para TARGETS

DS['Crop'] = label_encoder.fit_transform(DS['Crop'])  # Aplica a codificação de rótulos à coluna de 'Crop' do DataFrame df

#-----------------------------------------------------------------------------------------------------------------------
# Configura pandas para exibir o dataset inteiro
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Imprime o DataFrame inteiro
#print(DS.to_string())
#-----------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.preprocessing import MinMaxScaler

DS['Nitrogen'] = MinMaxScaler().fit_transform(np.array(DS['Nitrogen']).reshape(-1,1))
DS['Phosphorus'] = MinMaxScaler().fit_transform(np.array(DS['Phosphorus']).reshape(-1,1))
DS['Potassium'] = MinMaxScaler().fit_transform(np.array(DS['Potassium']).reshape(-1,1))
DS['Temperature'] = MinMaxScaler().fit_transform(np.array(DS['Temperature']).reshape(-1,1))
DS['Humidity'] = MinMaxScaler().fit_transform(np.array(DS['Humidity']).reshape(-1,1))
DS['pH_Value'] = MinMaxScaler().fit_transform(np.array(DS['pH_Value']).reshape(-1,1))
DS['Rainfall'] = MinMaxScaler().fit_transform(np.array(DS['Rainfall']).reshape(-1,1))

#-----------------------------------------------------------------------------------------------------------------------
# Imprime o DataFrame inteiro
#print(DS.to_string())
#-----------------------------------------------------------------------------------------------------------------------