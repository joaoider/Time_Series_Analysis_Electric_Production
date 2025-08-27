"""
Módulo para testes de estacionariedade
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from config import ADF_SIGNIFICANCE_LEVEL


class StationarityTester:
    """Classe para testes de estacionariedade"""
    
    def __init__(self, data):
        self.data = data
        
    def adfuller_test(self, window=12, show_plot=False):
        """
        Teste Augmented Dickey-Fuller para verificar estacionariedade
        
        Args:
            window: Janela para estatísticas móveis
            show_plot: Se deve mostrar gráfico das estatísticas móveis
            
        Returns:
            dict: Resultados do teste ADF
        """
        if self.data is None:
            return None
            
        # Calcular estatísticas móveis
        moving_average = self.data.rolling(window=window).mean()
        moving_std = self.data.rolling(window=window).std()
        
        # Realizar teste ADF
        adf_result = adfuller(self.data['value'], autolag='AIC')
        
        # Interpretar resultados
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        # Verificar se rejeita hipótese nula
        is_stationary = adf_statistic < critical_values['5%']
        
        results = {
            'adf_statistic': round(adf_statistic, 3),
            'p_value': round(p_value, 3),
            'critical_values': {k: round(v, 3) for k, v in critical_values.items()},
            'is_stationary': is_stationary,
            'interpretation': self._interpret_adf_results(adf_statistic, p_value, critical_values),
            'moving_stats': {
                'mean': moving_average,
                'std': moving_std
            }
        }
        
        return results
    
    def _interpret_adf_results(self, adf_statistic, p_value, critical_values):
        """Interpreta os resultados do teste ADF"""
        interpretation = []
        
        # Verificar estatística ADF vs valores críticos
        if adf_statistic < critical_values['1%']:
            interpretation.append("ADF < 1% critical value: Strong evidence against null hypothesis")
        elif adf_statistic < critical_values['5%']:
            interpretation.append("ADF < 5% critical value: Evidence against null hypothesis")
        elif adf_statistic < critical_values['10%']:
            interpretation.append("ADF < 10% critical value: Weak evidence against null hypothesis")
        else:
            interpretation.append("ADF > all critical values: No evidence against null hypothesis")
        
        # Verificar p-value
        if p_value < ADF_SIGNIFICANCE_LEVEL:
            interpretation.append(f"p-value ({p_value}) < {ADF_SIGNIFICANCE_LEVEL}: Reject null hypothesis")
        else:
            interpretation.append(f"p-value ({p_value}) >= {ADF_SIGNIFICANCE_LEVEL}: Fail to reject null hypothesis")
        
        # Conclusão final
        if adf_statistic < critical_values['5%'] and p_value < ADF_SIGNIFICANCE_LEVEL:
            interpretation.append("CONCLUSION: Time series is STATIONARY")
        else:
            interpretation.append("CONCLUSION: Time series is NON-STATIONARY")
        
        return interpretation
    
    def check_multiple_windows(self, windows=[6, 12, 24]):
        """Testa estacionariedade com diferentes janelas"""
        if self.data is None:
            return None
            
        results = {}
        for window in windows:
            results[f'window_{window}'] = self.adfuller_test(window=window)
        
        return results
    
    def get_stationarity_summary(self):
        """Retorna um resumo da análise de estacionariedade"""
        if self.data is None:
            return None
            
        # Teste principal com janela padrão
        main_test = self.adfuller_test()
        
        # Teste com múltiplas janelas
        multiple_tests = self.check_multiple_windows()
        
        summary = {
            'main_test': main_test,
            'multiple_windows': multiple_tests,
            'overall_conclusion': self._get_overall_conclusion(main_test, multiple_tests)
        }
        
        return summary
    
    def _get_overall_conclusion(self, main_test, multiple_tests):
        """Determina conclusão geral sobre estacionariedade"""
        if main_test is None:
            return "Unable to determine stationarity"
        
        # Contar quantos testes indicam estacionariedade
        stationary_count = 0
        total_tests = 1  # main_test
        
        if main_test['is_stationary']:
            stationary_count += 1
        
        for test_name, test_result in multiple_tests.items():
            if test_result and test_result['is_stationary']:
                stationary_count += 1
            total_tests += 1
        
        stationary_percentage = (stationary_count / total_tests) * 100
        
        if stationary_percentage >= 75:
            return f"Likely STATIONARY ({stationary_percentage:.1f}% of tests indicate stationarity)"
        elif stationary_percentage >= 50:
            return f"Possibly STATIONARY ({stationary_percentage:.1f}% of tests indicate stationarity)"
        else:
            return f"Likely NON-STATIONARY ({stationary_percentage:.1f}% of tests indicate stationarity)"
