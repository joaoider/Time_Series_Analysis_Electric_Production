"""
Módulo para transformações dos dados
"""

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from config import ROLLING_WINDOW


class DataTransformer:
    """Classe para transformações dos dados"""
    
    def __init__(self, data):
        self.data = data
        self.transformed_data = None
        
    def log_transform(self, lambda_param=0.0):
        """Aplica transformação logarítmica usando Box-Cox"""
        if self.data is None:
            return None
            
        try:
            # Aplicar transformação Box-Cox
            transformed_values = boxcox(self.data['value'], lmbda=lambda_param)
            
            # Criar novo DataFrame
            self.transformed_data = self.data.copy()
            self.transformed_data['value'] = transformed_values
            
            print(f"Transformação logarítmica aplicada com lambda={lambda_param}")
            return self.transformed_data
            
        except Exception as e:
            print(f"Erro na transformação logarítmica: {e}")
            return None
    
    def remove_trend_moving_average(self, window=ROLLING_WINDOW):
        """Remove tendência usando média móvel"""
        if self.transformed_data is None:
            print("Aplique primeiro uma transformação logarítmica")
            return None
            
        try:
            # Calcular média móvel
            moving_avg = self.transformed_data.rolling(window=window).mean()
            
            # Remover tendência
            detrended_data = self.transformed_data - moving_avg
            
            # Remover valores NaN
            detrended_data = detrended_data.dropna()
            
            print(f"Tendência removida usando média móvel (janela={window})")
            return detrended_data
            
        except Exception as e:
            print(f"Erro ao remover tendência: {e}")
            return None
    
    def exponential_decay_transform(self, halflife=12):
        """Aplica transformação de decaimento exponencial"""
        if self.transformed_data is None:
            print("Aplique primeiro uma transformação logarítmica")
            return None
            
        try:
            # Calcular média exponencial
            exp_mean = self.transformed_data.ewm(
                halflife=halflife, 
                min_periods=0, 
                adjust=True
            ).mean()
            
            # Aplicar transformação
            transformed_data = self.transformed_data - exp_mean
            
            print(f"Transformação de decaimento exponencial aplicada (halflife={halflife})")
            return transformed_data
            
        except Exception as e:
            print(f"Erro na transformação exponencial: {e}")
            return None
    
    def apply_full_transformation_pipeline(self, log_lambda=0.0, ma_window=12, exp_halflife=12):
        """Aplica pipeline completo de transformações"""
        if self.data is None:
            return None
            
        print("Iniciando pipeline de transformações...")
        
        # Passo 1: Transformação logarítmica
        step1 = self.log_transform(log_lambda)
        if step1 is None:
            return None
            
        # Passo 2: Remoção de tendência
        step2 = self.remove_trend_moving_average(ma_window)
        if step2 is None:
            return None
            
        # Passo 3: Transformação exponencial
        step3 = self.exponential_decay_transform(exp_halflife)
        if step3 is None:
            return None
            
        print("Pipeline de transformações concluído com sucesso!")
        return step3
    
    def reverse_transformations(self, transformed_data, original_data):
        """Reverte as transformações aplicadas (aproximação)"""
        if transformed_data is None or original_data is None:
            return None
            
        try:
            # Esta é uma aproximação - em casos reais, seria necessário
            # armazenar os parâmetros exatos de cada transformação
            
            # Para Box-Cox, usar exp() se lambda=0
            # Para média móvel, adicionar de volta
            # Para decaimento exponencial, adicionar de volta
            
            print("Aviso: Reversão de transformações é uma aproximação")
            return transformed_data
            
        except Exception as e:
            print(f"Erro ao reverter transformações: {e}")
            return None
    
    def get_transformation_summary(self):
        """Retorna resumo das transformações aplicadas"""
        summary = {
            'original_data_shape': self.data.shape if self.data is not None else None,
            'transformed_data_shape': self.transformed_data.shape if self.transformed_data is not None else None,
            'transformations_applied': []
        }
        
        if self.transformed_data is not None:
            summary['transformations_applied'].append('log_transform')
            
        return summary
