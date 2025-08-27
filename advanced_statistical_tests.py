"""
Módulo para testes estatísticos avançados de séries temporais
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import coint
from arch import arch_model
from config import ADF_SIGNIFICANCE_LEVEL


class AdvancedStatisticalTests:
    """Classe para testes estatísticos avançados"""
    
    def __init__(self, data):
        self.data = data
        self.test_results = {}
        
    def comprehensive_stationarity_tests(self):
        """Executa bateria completa de testes de estacionariedade"""
        results = {}
        
        # 1. Augmented Dickey-Fuller (ADF)
        results['adf'] = self._adf_test()
        
        # 2. Kwiatkowski-Phillips-Schmidt-Shin (KPSS)
        results['kpss'] = self._kpss_test()
        
        # 3. Zivot-Andrews (para quebras estruturais)
        results['zivot_andrews'] = self._zivot_andrews_test()
        
        # 4. Phillips-Perron
        results['phillips_perron'] = self._phillips_perron_test()
        
        # 5. Teste de raiz unitária com quebras estruturais
        results['structural_breaks'] = self._structural_breaks_test()
        
        return results
    
    def _adf_test(self):
        """Teste ADF com múltiplas especificações"""
        specifications = [
            {'autolag': 'AIC', 'regression': 'c'},
            {'autolag': 'BIC', 'regression': 'c'},
            {'autolag': 'AIC', 'regression': 'ct'},
            {'autolag': 'BIC', 'regression': 'ct'},
            {'autolag': 'AIC', 'regression': 'ctt'},
            {'autolag': 'BIC', 'regression': 'ctt'}
        ]
        
        results = {}
        for i, spec in enumerate(specifications):
            try:
                adf_result = adfuller(self.data['value'], **spec)
                results[f'spec_{i+1}'] = {
                    'statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'critical_values': adf_result[4],
                    'specification': spec,
                    'is_stationary': adf_result[0] < adf_result[4]['5%']
                }
            except Exception as e:
                results[f'spec_{i+1}'] = {'error': str(e)}
        
        return results
    
    def _kpss_test(self):
        """Teste KPSS"""
        try:
            from statsmodels.tsa.stattools import kpss
            kpss_result = kpss(self.data['value'], regression='c')
            
            return {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[0] < kpss_result[3]['5%']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _zivot_andrews_test(self):
        """Teste Zivot-Andrews para quebras estruturais"""
        try:
            za_result = zivot_andrews(self.data['value'], regression='ct')
            
            return {
                'statistic': za_result.stat,
                'p_value': za_result.pvalue,
                'critical_values': za_result.critvalues,
                'break_date': za_result.break_date,
                'is_stationary': za_result.stat < za_result.critvalues['5%']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _phillips_perron_test(self):
        """Teste Phillips-Perron"""
        try:
            from statsmodels.tsa.stattools import PhillipsPerron
            pp_result = PhillipsPerron(self.data['value'])
            
            return {
                'statistic': pp_result.stat,
                'p_value': pp_result.pvalue,
                'critical_values': pp_result.critvalues,
                'is_stationary': pp_result.stat < pp_result.critvalues['5%']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _structural_breaks_test(self):
        """Teste para quebras estruturais"""
        try:
            from statsmodels.tsa.regime_switching import MarkovRegression
            
            # Teste simples de quebras estruturais
            n = len(self.data)
            mid_point = n // 2
            
            # Dividir em dois períodos
            period1 = self.data.iloc[:mid_point]['value']
            period2 = self.data.iloc[mid_point:]['value']
            
            # Teste de igualdade de médias
            t_stat, p_value = stats.ttest_ind(period1, period2)
            
            # Teste de igualdade de variâncias (Levene)
            levene_stat, levene_p = stats.levene(period1, period2)
            
            return {
                't_test': {
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < ADF_SIGNIFICANCE_LEVEL
                },
                'levene_test': {
                    'statistic': levene_stat,
                    'p_value': levene_p,
                    'significant_difference': levene_p < ADF_SIGNIFICANCE_LEVEL
                },
                'break_point': mid_point,
                'period1_mean': period1.mean(),
                'period2_mean': period2.mean(),
                'period1_var': period1.var(),
                'period2_var': period2.var()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def normality_tests(self):
        """Bateria de testes de normalidade"""
        data = self.data['value'].dropna()
        
        tests = {}
        
        # 1. Shapiro-Wilk
        try:
            shapiro_stat, shapiro_p = stats.shapiro(data)
            tests['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > ADF_SIGNIFICANCE_LEVEL
            }
        except Exception as e:
            tests['shapiro_wilk'] = {'error': str(e)}
        
        # 2. Anderson-Darling
        try:
            anderson_result = stats.anderson(data)
            tests['anderson_darling'] = {
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values,
                'significance_levels': anderson_result.significance_level,
                'is_normal': anderson_result.statistic < anderson_result.critical_values[2]  # 5%
            }
        except Exception as e:
            tests['anderson_darling'] = {'error': str(e)}
        
        # 3. Kolmogorov-Smirnov
        try:
            # Gerar dados normais com mesma média e desvio padrão
            normal_data = np.random.normal(data.mean(), data.std(), len(data))
            ks_stat, ks_p = stats.ks_2samp(data, normal_data)
            
            tests['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > ADF_SIGNIFICANCE_LEVEL
            }
        except Exception as e:
            tests['kolmogorov_smirnov'] = {'error': str(e)}
        
        # 4. Jarque-Bera
        try:
            jb_stat, jb_p = stats.jarque_bera(data)
            tests['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > ADF_SIGNIFICANCE_LEVEL
            }
        except Exception as e:
            tests['jarque_bera'] = {'error': str(e)}
        
        return tests
    
    def heteroscedasticity_tests(self):
        """Testes de heterocedasticidade"""
        try:
            # Preparar dados para teste
            data = self.data['value'].dropna()
            X = np.arange(len(data)).reshape(-1, 1)
            y = data.values
            
            # Teste de Breusch-Pagan
            bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(y, X)
            
            # Teste de White (simplificado)
            residuals = y - np.mean(y)
            white_stat = len(data) * np.corrcoef(X.flatten(), residuals**2)[0, 1]**2
            white_p = 1 - stats.chi2.cdf(white_stat, 1)
            
            return {
                'breusch_pagan': {
                    'statistic': bp_stat,
                    'p_value': bp_p,
                    'f_statistic': bp_f,
                    'f_p_value': bp_f_p,
                    'is_heteroscedastic': bp_p < ADF_SIGNIFICANCE_LEVEL
                },
                'white_test': {
                    'statistic': white_stat,
                    'p_value': white_p,
                    'is_heteroscedastic': white_p < ADF_SIGNIFICANCE_LEVEL
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def autocorrelation_tests(self):
        """Testes de autocorrelação"""
        try:
            data = self.data['value'].dropna()
            
            # Teste de Ljung-Box
            ljung_result = acorr_ljungbox(data, lags=[10, 20, 30], return_df=True)
            
            # Teste de Durbin-Watson
            residuals = data - data.shift(1)
            dw_stat = np.sum(residuals[1:]**2) / np.sum(residuals[1:]**2)
            
            return {
                'ljung_box': {
                    'statistics': ljung_result['lb_stat'].tolist(),
                    'p_values': ljung_result['lb_pvalue'].tolist(),
                    'lags': [10, 20, 30],
                    'significant_autocorr': any(p < ADF_SIGNIFICANCE_LEVEL for p in ljung_result['lb_pvalue'])
                },
                'durbin_watson': {
                    'statistic': dw_stat,
                    'interpretation': self._interpret_durbin_watson(dw_stat)
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _interpret_durbin_watson(self, dw_stat):
        """Interpreta estatística de Durbin-Watson"""
        if dw_stat < 1.5:
            return "Positive autocorrelation"
        elif dw_stat > 2.5:
            return "Negative autocorrelation"
        else:
            return "No significant autocorrelation"
    
    def cointegration_tests(self, other_series=None):
        """Testes de cointegração"""
        if other_series is None:
            return {'error': 'No other series provided for cointegration test'}
        
        try:
            # Teste de Engle-Granger
            coint_stat, p_value, critical_values = coint(self.data['value'], other_series)
            
            return {
                'engle_granger': {
                    'statistic': coint_stat,
                    'p_value': p_value,
                    'critical_values': critical_values,
                    'is_cointegrated': p_value < ADF_SIGNIFICANCE_LEVEL
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def arch_effects_test(self):
        """Teste para efeitos ARCH (heterocedasticidade condicional)"""
        try:
            data = self.data['value'].dropna()
            
            # Modelo ARCH(1)
            arch_model_1 = arch_model(data, vol='ARCH', p=1)
            arch_result_1 = arch_model_1.fit(disp='off')
            
            # Modelo ARCH(2)
            arch_model_2 = arch_model(data, vol='ARCH', p=2)
            arch_result_2 = arch_model_2.fit(disp='off')
            
            # Teste de significância dos parâmetros ARCH
            arch1_significant = any(p < ADF_SIGNIFICANCE_LEVEL for p in arch_result_1.pvalues[1:])
            arch2_significant = any(p < ADF_SIGNIFICANCE_LEVEL for p in arch_result_2.pvalues[1:])
            
            return {
                'arch_1': {
                    'aic': arch_result_1.aic,
                    'bic': arch_result_1.bic,
                    'parameters': arch_result_1.params.to_dict(),
                    'p_values': arch_result_1.pvalues.to_dict(),
                    'significant_arch': arch1_significant
                },
                'arch_2': {
                    'aic': arch_result_2.aic,
                    'bic': arch_result_2.bic,
                    'parameters': arch_result_2.params.to_dict(),
                    'p_values': arch_result_2.pvalues.to_dict(),
                    'significant_arch': arch2_significant
                },
                'best_model': 'ARCH(1)' if arch_result_1.aic < arch_result_2.aic else 'ARCH(2)'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def generate_comprehensive_report(self):
        """Gera relatório completo de todos os testes"""
        report = {
            'stationarity_tests': self.comprehensive_stationarity_tests(),
            'normality_tests': self.normality_tests(),
            'heteroscedasticity_tests': self.heteroscedasticity_tests(),
            'autocorrelation_tests': self.autocorrelation_tests(),
            'arch_effects': self.arch_effects_test()
        }
        
        # Resumo executivo
        summary = self._generate_executive_summary(report)
        report['executive_summary'] = summary
        
        return report
    
    def _generate_executive_summary(self, report):
        """Gera resumo executivo dos testes"""
        summary = {
            'overall_stationarity': 'Unknown',
            'normality_assessment': 'Unknown',
            'heteroscedasticity': 'Unknown',
            'autocorrelation': 'Unknown',
            'arch_effects': 'Unknown',
            'recommendations': []
        }
        
        # Avaliar estacionariedade
        stationarity_results = report['stationarity_tests']
        if 'adf' in stationarity_results:
            adf_tests = [r for r in stationarity_results['adf'].values() 
                        if isinstance(r, dict) and 'is_stationary' in r]
            if adf_tests:
                stationary_count = sum(1 for test in adf_tests if test['is_stationary'])
                total_tests = len(adf_tests)
                if stationary_count / total_tests >= 0.7:
                    summary['overall_stationarity'] = 'Likely Stationary'
                elif stationary_count / total_tests >= 0.3:
                    summary['overall_stationarity'] = 'Possibly Stationary'
                else:
                    summary['overall_stationarity'] = 'Likely Non-Stationary'
        
        # Avaliar normalidade
        normality_results = report['normality_tests']
        normal_tests = [r for r in normality_results.values() 
                       if isinstance(r, dict) and 'is_normal' in r]
        if normal_tests:
            normal_count = sum(1 for test in normal_tests if test['is_normal'])
            if normal_count / len(normal_tests) >= 0.7:
                summary['normality_assessment'] = 'Likely Normal'
            else:
                summary['normality_assessment'] = 'Likely Non-Normal'
        
        # Gerar recomendações
        if summary['overall_stationarity'] == 'Likely Non-Stationary':
            summary['recommendations'].append('Apply differencing or transformations')
        
        if summary['normality_assessment'] == 'Likely Non-Normal':
            summary['recommendations'].append('Consider non-parametric methods or transformations')
        
        if 'heteroscedasticity_tests' in report and 'error' not in report['heteroscedasticity_tests']:
            het_results = report['heteroscedasticity_tests']
            if any('is_heteroscedastic' in test and test['is_heteroscedastic'] 
                   for test in het_results.values()):
                summary['heteroscedasticity'] = 'Present'
                summary['recommendations'].append('Use robust standard errors or GARCH models')
            else:
                summary['heteroscedasticity'] = 'Absent'
        
        return summary
