"""
Módulo para modelos de machine learning e séries temporais
"""

import pandas as pd
import numpy as np
from pandas import DataFrame, concat
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from xgboost import XGBRegressor
from config import ARIMA_ORDER, SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER, XGB_PARAMS


class TimeSeriesModels:
    """Classe para modelos de séries temporais"""
    
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.predictions = {}
        
    def persistence_model(self, test_size=10):
        """Modelo de persistência (baseline)"""
        if self.data is None:
            return None
            
        try:
            # Preparar dados
            values = DataFrame(self.data['value'].values)
            persistence_df = concat([values.shift(1), values], axis=1)
            persistence_df.columns = ['t-1', 't+1']
            per_values = persistence_df.values
            
            # Dividir em treino e teste
            train = per_values[1:len(per_values)-test_size]
            test = per_values[len(per_values)-test_size:]
            
            X_train, y_train = train[:, 0], train[:, 1]
            X_test, y_test = test[:, 0], test[:, 1]
            
            # Fazer previsões
            predictions = X_test  # Modelo de persistência
            
            # Calcular métricas
            mse = mean_squared_error(y_test, predictions)
            
            results = {
                'model_type': 'Persistence',
                'mse': round(mse, 4),
                'predictions': predictions,
                'actual': y_test,
                'train_data': train,
                'test_data': test
            }
            
            self.models['persistence'] = results
            self.predictions['persistence'] = predictions
            
            return results
            
        except Exception as e:
            print(f"Erro no modelo de persistência: {e}")
            return None
    
    def autoregression_model(self, order=(2, 1, 0), test_size=10):
        """Modelo de autoregressão (AR)"""
        if self.data is None:
            return None
            
        try:
            # Preparar dados
            ar_values = self.data['value'].values
            train = ar_values[1:len(ar_values)-test_size]
            test = ar_values[len(ar_values)-test_size:]
            
            # Treinar modelo
            model = ARIMA(train, order=order)
            ar_model = model.fit()
            
            # Fazer previsões
            predictions = ar_model.predict(
                start=len(train), 
                end=len(train)+len(test)-1, 
                dynamic=False
            )
            
            # Calcular métricas
            mse = mean_squared_error(test, predictions)
            
            results = {
                'model_type': 'Autoregression',
                'order': order,
                'mse': round(mse, 4),
                'predictions': predictions,
                'actual': test,
                'model': ar_model,
                'train_data': train,
                'test_data': test
            }
            
            self.models['autoregression'] = results
            self.predictions['autoregression'] = predictions
            
            return results
            
        except Exception as e:
            print(f"Erro no modelo de autoregressão: {e}")
            return None
    
    def moving_average_model(self, order=(0, 1, 2), test_size=10):
        """Modelo de média móvel (MA)"""
        if self.data is None:
            return None
            
        try:
            # Preparar dados
            ma_values = self.data['value'].values
            train = ma_values[1:len(ma_values)-test_size]
            test = ma_values[len(ma_values)-test_size:]
            
            # Treinar modelo
            model = ARIMA(train, order=order)
            ma_model = model.fit()
            
            # Fazer previsões
            predictions = ma_model.predict(
                start=len(train), 
                end=len(train)+len(test)-1, 
                dynamic=False
            )
            
            # Calcular métricas
            mse = mean_squared_error(test, predictions)
            
            results = {
                'model_type': 'Moving Average',
                'order': order,
                'mse': round(mse, 4),
                'predictions': predictions,
                'actual': test,
                'model': ma_model,
                'train_data': train,
                'test_data': test
            }
            
            self.models['moving_average'] = results
            self.predictions['moving_average'] = predictions
            
            return results
            
        except Exception as e:
            print(f"Erro no modelo de média móvel: {e}")
            return None
    
    def arima_model(self, order=ARIMA_ORDER, test_size=10):
        """Modelo ARIMA"""
        if self.data is None:
            return None
            
        try:
            # Preparar dados
            arima_values = self.data['value'].values
            train = arima_values[1:len(arima_values)-test_size]
            test = arima_values[len(arima_values)-test_size:]
            
            # Treinar modelo
            model = ARIMA(train, order=order)
            arima_model = model.fit()
            
            # Fazer previsões
            predictions = arima_model.predict(
                start=len(train), 
                end=len(train)+len(test)-1, 
                dynamic=False
            )
            
            # Calcular métricas
            mse = mean_squared_error(test, predictions)
            
            results = {
                'model_type': 'ARIMA',
                'order': order,
                'mse': round(mse, 4),
                'predictions': predictions,
                'actual': test,
                'model': arima_model,
                'train_data': train,
                'test_data': test
            }
            
            self.models['arima'] = results
            self.predictions['arima'] = predictions
            
            return results
            
        except Exception as e:
            print(f"Erro no modelo ARIMA: {e}")
            return None
    
    def sarimax_model(self, order=SARIMAX_ORDER, seasonal_order=SARIMAX_SEASONAL_ORDER, test_size=10):
        """Modelo SARIMAX"""
        if self.data is None:
            return None
            
        try:
            # Preparar dados
            sarimax_values = self.data['value'].values
            train = sarimax_values[1:len(sarimax_values)-test_size]
            test = sarimax_values[len(sarimax_values)-test_size:]
            
            # Treinar modelo
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
            sarimax_model = model.fit()
            
            # Fazer previsões
            predictions = sarimax_model.predict(
                start=len(train), 
                end=len(train)+len(test)-1, 
                dynamic=False
            )
            
            # Calcular métricas
            mse = mean_squared_error(test, predictions)
            
            results = {
                'model_type': 'SARIMAX',
                'order': order,
                'seasonal_order': seasonal_order,
                'mse': round(mse, 4),
                'predictions': predictions,
                'actual': test,
                'model': sarimax_model,
                'train_data': train,
                'test_data': test
            }
            
            self.models['sarimax'] = results
            self.predictions['sarimax'] = predictions
            
            return results
            
        except Exception as e:
            print(f"Erro no modelo SARIMAX: {e}")
            return None
    
    def auto_arima_model(self, seasonal=True, m=6):
        """Modelo AutoARIMA"""
        if self.data is None:
            return None
            
        try:
            # Treinar modelo
            auto_model = auto_arima(
                self.data, 
                d=1, 
                start_p=1, 
                start_q=1, 
                max_p=3, 
                max_q=3, 
                seasonal=seasonal, 
                m=m, 
                D=1, 
                start_P=1, 
                start_Q=1, 
                max_P=2, 
                max_Q=2, 
                information_criterion='aic', 
                trace=True, 
                error_action='ignore', 
                stepwise=True
            )
            
            results = {
                'model_type': 'AutoARIMA',
                'best_order': auto_model.order,
                'best_seasonal_order': auto_model.seasonal_order,
                'aic': auto_model.aic(),
                'model': auto_model
            }
            
            self.models['auto_arima'] = results
            
            return results
            
        except Exception as e:
            print(f"Erro no modelo AutoARIMA: {e}")
            return None
    
    def xgboost_model(self, train_data, test_data):
        """Modelo XGBoost"""
        if train_data is None or test_data is None:
            return None
            
        try:
            # Preparar dados
            X_train, y_train = self._prepare_lagged_data(train_data)
            X_test, y_test = self._prepare_lagged_data(test_data)
            
            if X_train is None or X_test is None:
                return None
            
            # Treinar modelo
            model = XGBRegressor(**XGB_PARAMS)
            xgb_model = model.fit(X_train, y_train)
            
            # Fazer previsões
            predictions = xgb_model.predict(X_test)
            
            # Calcular métricas
            mse = mean_squared_error(y_test, predictions)
            
            results = {
                'model_type': 'XGBoost',
                'mse': round(mse, 4),
                'predictions': predictions,
                'actual': y_test,
                'model': xgb_model,
                'train_data': (X_train, y_train),
                'test_data': (X_test, y_test)
            }
            
            self.models['xgboost'] = results
            self.predictions['xgboost'] = predictions
            
            return results
            
        except Exception as e:
            print(f"Erro no modelo XGBoost: {e}")
            return None
    
    def _prepare_lagged_data(self, data, lag=1):
        """Prepara dados com variáveis defasadas"""
        try:
            data_copy = data.copy()
            data_copy['target'] = data_copy['value'].shift(-lag)
            data_copy = data_copy.dropna()
            
            X = data_copy[['value']].values
            y = data_copy['target'].values
            
            return X, y
        except Exception as e:
            print(f"Erro ao preparar dados defasados: {e}")
            return None, None
    
    def get_all_models_summary(self):
        """Retorna resumo de todos os modelos treinados"""
        summary = {}
        
        for model_name, model_results in self.models.items():
            if model_results and 'mse' in model_results:
                summary[model_name] = {
                    'mse': model_results['mse'],
                    'model_type': model_results['model_type']
                }
        
        return summary
    
    def get_best_model(self):
        """Retorna o melhor modelo baseado no MSE"""
        summary = self.get_all_models_summary()
        
        if not summary:
            return None
        
        # Ordenar por MSE (menor é melhor)
        best_model = min(summary.items(), key=lambda x: x[1]['mse'])
        
        return {
            'model_name': best_model[0],
            'mse': best_model[1]['mse'],
            'model_type': best_model[1]['model_type']
        }
