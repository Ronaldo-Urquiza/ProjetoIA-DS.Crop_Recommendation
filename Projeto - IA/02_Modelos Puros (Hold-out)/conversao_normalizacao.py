import pandas as pd
from sklearn import preprocessing  # Importa o módulo de pré-processamento do scikit-learn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def converte_normaliza (ds_path):
    DS = pd.read_csv(ds_path) # Lendo o arquivo CSV
    label_encoder = preprocessing.LabelEncoder()  # Inicializa o Label Encoder para TARGETS
    DS['Crop'] = label_encoder.fit_transform(DS['Crop'])  # Aplica a codificação de rótulos à coluna de 'Crop' do DataFrame df

    #-------------------------------------------------------------------------------------------------------------------
    # Configura pandas para exibir o dataset inteiro
    #pd.set_option('display.max_columns', None)

    # Imprime o DataFrame inteiro
    #print(DS.to_string())
    #-------------------------------------------------------------------------------------------------------------------

    DS['Nitrogen'] = MinMaxScaler().fit_transform(np.array(DS['Nitrogen']).reshape(-1,1))
    DS['Phosphorus'] = MinMaxScaler().fit_transform(np.array(DS['Phosphorus']).reshape(-1,1))
    DS['Potassium'] = MinMaxScaler().fit_transform(np.array(DS['Potassium']).reshape(-1,1))
    DS['Temperature'] = MinMaxScaler().fit_transform(np.array(DS['Temperature']).reshape(-1,1))
    DS['Humidity'] = MinMaxScaler().fit_transform(np.array(DS['Humidity']).reshape(-1,1))
    DS['pH_Value'] = MinMaxScaler().fit_transform(np.array(DS['pH_Value']).reshape(-1,1))
    DS['Rainfall'] = MinMaxScaler().fit_transform(np.array(DS['Rainfall']).reshape(-1,1))

    #-------------------------------------------------------------------------------------------------------------------
    # Imprime o DataFrame inteiro
    #print(DS.to_string())
    #-------------------------------------------------------------------------------------------------------------------

    return DS
