"""
Módulo para análise exploratória dos dados
"""

import pandas as pd
import numpy as np
from config import ROLLING_WINDOW


class DataAnalyzer:
    """Classe para análise exploratória dos dados"""
    
    def __init__(self, data):
        self.data = data
        
    def calculate_rolling_statistics(self, window=ROLLING_WINDOW):
        """Calcula estatísticas móveis"""
        if self.data is None:
            return None, None
            
        rolling_mean = self.data.rolling(window=window).mean()
        rolling_std = self.data.rolling(window=window).std()
        
        return rolling_mean, rolling_std
    
    def calculate_basic_statistics(self):
        """Calcula estatísticas básicas dos dados"""
        if self.data is None:
            return None
            
        stats = {
            'count': len(self.data),
            'mean': self.data['value'].mean(),
            'std': self.data['value'].std(),
            'min': self.data['value'].min(),
            'max': self.data['value'].max(),
            'median': self.data['value'].median(),
            'skewness': self.data['value'].skew(),
            'kurtosis': self.data['value'].kurtosis()
        }
        
        return stats
    
    def check_stationarity_by_splitting(self, split_point=200):
        """Verifica estacionariedade dividindo os dados em duas partes"""
        if self.data is None or len(self.data) < split_point * 2:
            return None
            
        part1 = self.data.iloc[:split_point]
        part2 = self.data.iloc[split_point:]
        
        stats = {
            'part1': {
                'mean': part1['value'].mean(),
                'variance': part1['value'].var()
            },
            'part2': {
                'mean': part2['value'].mean(),
                'variance': part2['value'].var()
            }
        }
        
        # Verificar se há diferenças significativas
        mean_diff = abs(stats['part1']['mean'] - stats['part2']['mean'])
        var_diff = abs(stats['part1']['variance'] - stats['part2']['variance'])
        
        stats['stationarity_check'] = {
            'mean_difference': mean_diff,
            'variance_difference': var_diff,
            'likely_stationary': mean_diff < 10 and var_diff < 50  # Thresholds arbitrários
        }
        
        return stats
    
    def get_data_summary(self):
        """Retorna um resumo completo dos dados"""
        if self.data is None:
            return None
            
        summary = {
            'basic_stats': self.calculate_basic_statistics(),
            'stationarity_check': self.check_stationarity_by_splitting(),
            'data_range': {
                'start_date': self.data.index.min(),
                'end_date': self.data.index.max(),
                'total_periods': len(self.data)
            }
        }
        
        return summary
