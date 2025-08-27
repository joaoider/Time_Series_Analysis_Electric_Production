"""
Módulo para carregamento e preparação dos dados
"""

import pandas as pd
import kaggle
import os
from config import DATA_PATH, KAGGLE_DATASET


class DataLoader:
    """Classe para carregamento e preparação dos dados"""
    
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.df = None
        
    def download_kaggle_data(self):
        """Download do dataset do Kaggle"""
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                KAGGLE_DATASET, 
                path='./data', 
                unzip=True
            )
            print("Dataset baixado com sucesso!")
        except Exception as e:
            print(f"Erro ao baixar dataset: {e}")
            
    def load_data(self):
        """Carrega os dados do arquivo CSV"""
        try:
            self.df = pd.read_csv(
                self.data_path, 
                index_col='DATE', 
                parse_dates=True
            )
            # Renomear coluna para 'value'
            self.df.columns = ['value']
            print(f"Dados carregados com sucesso! Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return None
    
    def get_data_info(self):
        """Retorna informações básicas sobre os dados"""
        if self.df is None:
            print("Dados não carregados!")
            return None
            
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'index_min': self.df.index.min(),
            'index_max': self.df.index.max(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        return info
    
    def split_data(self, test_size=0.25):
        """Divide os dados em treino e teste"""
        if self.df is None:
            print("Dados não carregados!")
            return None, None
            
        split_idx = int(len(self.df) * (1 - test_size))
        train_data = self.df.iloc[:split_idx]
        test_data = self.df.iloc[split_idx:]
        
        return train_data, test_data
    
    def split_by_date(self, split_date='2016-12-31'):
        """Divide os dados por data específica"""
        if self.df is None:
            print("Dados não carregados!")
            return None, None
            
        train_data = self.df.loc[self.df.index <= split_date]
        test_data = self.df.loc[self.df.index > split_date]
        
        return train_data, test_data
    
    def prepare_lagged_data(self, data, lag=1):
        """Prepara dados com variáveis defasadas para modelos de ML"""
        if data is None:
            return None, None
            
        # Criar variável target (próximo valor)
        data_copy = data.copy()
        data_copy['target'] = data_copy['value'].shift(-lag)
        
        # Remover valores NaN
        data_copy = data_copy.dropna()
        
        # Separar features e target
        X = data_copy[['value']].values
        y = data_copy['target'].values
        
        return X, y
